import sys
import gensim
from gensim.models import Word2Vec
from sklearn.externals import joblib
from gensim import utils, matutils
import scipy
import numpy as np
import fastcluster
import scipy.cluster.hierarchy
import scipy.cluster.hierarchy as sch 
import string
from pprint import pprint
from configobj import ConfigObj
import traceback
from tsne import tsne
import re, os, ast
import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
from sklearn import datasets, linear_model, cross_validation
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from word2vec_srl_utils import DocumentFeatures
from text_extract_doc import Sentences
import numpy as Math
import pylab as Plot
from senna_py import srl_extract

def cleanse_data(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

#	print "cleansing"
	temp = re.sub(r'[^\x00-\x7F]+',' ', text)
	temp = re.sub(r'(\d+(\s)?(yrs|year|years|Yrs|Years|Year|yr))'," TIME ",temp)
	temp = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",temp)
	temp = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',temp)
	temp = re.sub(r"[\r\n]+[\s\t]+",'\n',temp)	
	wF = set(string.punctuation) - set(["+"])
	for c in wF:
        	temp =temp.replace(c," ")	

	return temp


class TrainData():

	def __init__(self):
		pass

	def load_w2vmodel(self,model):
		return gensim.models.Word2Vec.load(model)

	def get_tfidf_model(self, dirname):
		
		data = Sentences(dirname)
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix_train = tfidf_vectorizer.fit_transform(data)
		return tfidf_vectorizer
	
	def train_model(self, dirname, w2v_model_path,ndim):
		
		
		tfidf_model = self.get_tfidf_model(dirname)
		w2v_model = self.load_w2vmodel(w2v_model_path)
		trd = DocumentFeatures()
		wt_vect_data = []
		label_data = []
		fn = []
		for fname in os.listdir(dirname):
			f = open(os.path.join(dirname, fname))
			text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(text)	
			print "processsing ::" + fname
			VA0, VA1 =srl_extract(text)
			sent_vect = trd.get_sent_circconv_vec(text, w2v_model, ndim, 'tfidf', tfidf_model,VA0, VA1)
			if fname[-10:-4] == 'accept':
				label = 1
			else:
				label = 0
			fn.append(fname)
			wt_vect_data.append(sent_vect[0])
			label_data.append(label)

		return wt_vect_data, label_data, fn



class PredictData():

	def __init__(self):
		pass

	def load_w2vmodel(self,model):
		return gensim.models.Word2Vec.load(model)

	def get_tfidf_model(self, dirname):
		
		data = Sentences(dirname)
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix_train = tfidf_vectorizer.fit_transform(data)
		return tfidf_vectorizer
	
	def train_model(self, dirname, w2v_model_path,ndim):
		
		
		tfidf_model = self.get_tfidf_model(dirname)
		w2v_model = self.load_w2vmodel(w2v_model_path)
		trd = DocumentFeatures()
		wt_vect_data = []
	#	label_data = []
		fn = []
		for fname in os.listdir(dirname):
			f = open(os.path.join(dirname, fname))
			text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(text)	
			print "processsing ::" + fname
			VA0, VA1 =srl_extract(text)
			sent_vect = trd.get_sent_circconv_vec(text, w2v_model, ndim, 'tfidf', tfidf_model,VA0, VA1)
	#		if fname[-10:-4] == 'accept':
	#			label = 1
	#		else:
	#			label = 0
			fn.append(fname)
			wt_vect_data.append(sent_vect[0])
	#		label_data.append(label)

		return wt_vect_data,  fn

		
		
class genSRLVec():

	def __init__(self,train_dirname,test_dirname,predict_dirname,w2v_model_path,size):
		
		self.train_dirname = train_dirname
		self.test_dirname = test_dirname
		self.predict_dirname = predict_dirname
		self.w2v_model_path = w2v_model_path
		self.size = size
		self.x_wt = []
		self.Ylabels = []
		self.xTest_wt = []
		self.xPred_wt = []		
		self.Y_test = []
		self.Y_pred = []
		self.result = []
		self.fn_train = []
		self.fn_test = []


	def start(self):

		td = TrainData()

		self.x_wt, self.Ylabels,self.fn_train = td.train_model(self.train_dirname, self.w2v_model_path,self.size)

		self.xTest_wt, self.Y_test,self.fn_test = td.train_model(self.test_dirname, self.w2v_model_path,self.size)	


		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xTest_wt)

		self.result = self.getAnalysis(self.Y_test,self.Y_pred)



	def train_predict(self):
		td = TrainData()

		f = []
		self.x_wt, self.Ylabels,self.fn_train  = td.train_model( self.train_dirname, self.w2v_model_path,self.size)

		pd = PredictData()
		self.xPred_wt,self.fn_test = pd.train_model(self.predict_dirname, self.w2v_model_path,self.size)


		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xPred_wt)

		self.savePredictResult2File(self.fn_test,self.Y_pred,"pred_output_srl.tsv")


	def savePredictResult2File(self,fn_test,Y_pred,fname):
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tpredicted\n"))

		for i in range(len(Y_pred)):
			input_filename.write((fn_test[i]+"\t"+str(self.Y_pred[i])+"\n"))
		input_filename.close()


## Print results 
	def printAnalysis(self,true_pred,y_pred1):

		print "########## Analysing the Model result ##########################"


		math_corr = matthews_corrcoef( true_pred,y_pred1)
		roc_auc = roc_auc_score( true_pred,y_pred1)

		print(classification_report( true_pred,y_pred1))
		print("Matthews correlation :" + str(matthews_corrcoef( true_pred,y_pred1)))
		print("ROC AUC score :" + str(roc_auc_score( true_pred,y_pred1)))

	def getAnalysis(self,true_pred,y_pred1):
		precision, recall, fscore, support = score(true_pred,y_pred1)
		return [matthews_corrcoef( true_pred,y_pred1),roc_auc_score( true_pred,y_pred1),precision[0],precision[1],recall[0],recall[1],fscore[0],fscore[1],support[0],support[1]]

	def printResult(self):
		
		self.printAnalysis(self.Y_test,self.Y_pred)

	def getResult(self):
		return self.result

	def saveResult2file(self,fname):
		
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tlabelled\tpredicted\n"))

		for i in range(len(self.Y_test)):
			input_filename.write((self.fn_test[i]+"\t"+str(self.Y_test[i])+"\t"+str(self.Y_pred[i])+"\n"))
		input_filename.close()
		
	def saveNFoldResult2file(self,y_test,y_pred,fn_test,fname):
		
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tlabelled\tpredicted\n"))

		for i in range(len(y_test)):
			input_filename.write((fn_test[i]+"\t"+str(y_test[i])+"\t"+str(y_pred[i])+"\n"))
		input_filename.close()


	def NFoldTest(self, iter_N=5,split =0.15,random_state=0):

		x_total = self.x_wt #+ self.xTest_wt
		y_total = self.Ylabels #+ self.Y_test
		fn_total = self.fn_train #+ self.fn_test
		self.saveNFoldResult2file(y_total,y_total,fn_total,"total_srl_data.tsv")
		kf_total = cross_validation.ShuffleSplit(len(x_total), n_iter=iter_N, test_size=split,   random_state=random_state)
		x_tot_np = Math.array(x_total)
		y_tot_np = Math.array(y_total)
		j =0
		fn_test = []
		for train, test in kf_total:
			j += 1
			lgr = LogisticRegression(C=1.0).fit(x_tot_np[train],y_tot_np[train])
			y_pred = lgr.predict(x_tot_np[test])
			y_test = y_tot_np[test]
			fn_test = []
			for i in test:			
				fn_test.append(fn_total[i])
			fn = "Output_srl_" + str(j) + ".tsv"
			self.saveNFoldResult2file(y_test,y_pred,fn_test,fn)


		


if __name__ == '__main__': 
	train_dirname = '/home/viswanath/workspace/test_resume/train'
#	test_dirname = '/home/viswanath/workspace/resume_data/res_dir/test'
#	total_dirname = '/home/viswanath/workspace/resume_data/res_dir/total'
	predict_dirname = '/home/viswanath/workspace/test_resume/predict'
	w2v_model_path = '/home/viswanath/workspace/test_resume/model/w2v_model_100.mod'
	size = 100

	gsl = genSRLVec(train_dirname,"",predict_dirname,w2v_model_path,size)
	gsl.train_predict()

