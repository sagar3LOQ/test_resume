
import sys
import gensim
from gensim.models import Word2Vec
from sklearn.externals import joblib
from gensim import utils, matutils
import random
import scipy
from nltk.corpus import stopwords
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
from sklearn import datasets, linear_model, cross_validation
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from word2vec_utils import DocumentFeatures
from text_extract_doc_flist import Sentences
import numpy as Math
import pylab as Plot
import operator


def cleanse_data(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

#	print "cleansing"
	text = re.sub(r'[^\x00-\x7F]+',' ', text)
	text = re.sub(r'(\d+(\s)?(yrs|year|years|Yrs|Years|Year|yr))'," TIME ",text)
	text = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",text)
	text = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',text)
	text = re.sub(r"[\r\n]+[\s\t]+",'\n',text)	
	wF = set(string.punctuation) - set(["+"])
	for c in wF:
        	text =text.replace(c," ")	

	return text


## Training data class

class TrainData():

	def __init__(self):
		pass


## Load Gensim framework
	def load_w2vmodel(self,model):
		return gensim.models.Word2Vec.load(model)


## get tfidf model trained from given directory 'dirname' 
	def get_tfidf_model(self, dirname):
		dirname = "/home/viswanath/Downloads/total_resume/temp_files"
		data = Sentences(dirname)
		tfidf_vectorizer = TfidfVectorizer(stop_words='english')
		tfidf_matrix = tfidf_vectorizer.fit_transform(data)
		mat_array = tfidf_matrix.toarray()
		fn = tfidf_vectorizer.get_feature_names()
	#	print fn
		return tfidf_vectorizer


#train model based on top N wwords from TFIDF model	
	def train_model(self, dirname, w2v_model_path,topN,ndim):
		tfidf_model = self.get_tfidf_model(dirname)
		X = []
		label_data = []
		fn = []
		for fname in os.listdir(dirname):
			print "Processing = " + fname
			f = open(os.path.join(dirname, fname),"r")
			raw_text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(raw_text)
			nword = self.top_n_words_doc(text,tfidf_model,topN)
		#	print nword

	#		X_coeff = self.gen_LR_weight( nword,text, w2v_model_path,tfidf_model, ndim)
			X_coeff = self.get_docvec( w2v_model_path,tfidf_model, nword, text)


			if fname[-10:-4] == 'accept':
				label = 1
			else:
				label = 0
			fn.append(fname)
			X.append(X_coeff[0])
			label_data.append(label)
		return X, label_data, fn


# find top N words from TFIDF model
	def top_n_words_doc(self,text,tfidf_model,topn=20):
		
		token = text.lower().split()
		words = {}
	
		for w in token:
			wt = tfidf_model.idf_[tfidf_model.vocabulary_[w]] if w in tfidf_model.vocabulary_ else 0
			words[w] = wt

		lenw = len(words)
	#	print lenw, len(token)
		if (lenw < topn): topn = lenw

		sorted_x = sorted(words.items(), key=operator.itemgetter(1),reverse=True)
		listd = sorted_x[0:topn]
	#	print listd
		word= []
		for i in range((topn)):
			word.append(listd[i][0])

		return word


	def get_docvec(self,w2v_model,tfidf_model, pos_words, text,neg_words=[]):
		model = self.load_w2vmodel(w2v_model)
		model_vocab =  tfidf_model.vocabulary_

		X1 = [tfidf_model.idf_[tfidf_model.vocabulary_[i]]* model[i] for i in pos_words if i in model_vocab if i in model.vocab]
		if len(neg_words) == 0:
			n_neg = 300
			neg_words = set(random.sample(model_vocab,n_neg)) - set(pos_words)
		X2 = [tfidf_model.idf_[tfidf_model.vocabulary_[i]]* model[i] for i in neg_words if i in model.vocab]
		X = X1 + X2
		
		y = [1] * len(X1) + [0] * len(X2)
	

		regr = LogisticRegression().fit(X, y)
		docvector = matutils.unitvec(regr.coef_)
		return docvector
#		pprint(get_closest_words_to_vec(docvector,model,50))

# Generate wieghts from Logistic Regression Model
	def gen_LR_weight(self, words,text, w2v_model,tfidf_model, ndim):
		model = self.load_w2vmodel(w2v_model)
		text = text.lower().split()

		notfound = 0
		denom = 0
		nvecs = 0
		label = []
		doc_vec = []
		wt =0
	#	print len(text), len(words)
		for w in text:
			if w in model:
				wt = tfidf_model.idf_[tfidf_model.vocabulary_[w]] if w in tfidf_model.vocabulary_ else 0
				ve = np.multiply(model[w], wt)

				doc_vec.append(ve)				
				if w in words: 
					label.append(1)
				else: 
					label.append(0)
				nvecs += 1
				denom += wt
			else:
				notfound += 1
		if denom == 0:
			return []
		
		logit = LogisticRegression(C=1.0).fit(doc_vec, label)

		return logit.coef_



class PredictData():

	def __init__(self):
		pass


## Load Gensim framework
	def load_w2vmodel(self,model):
		return gensim.models.Word2Vec.load(model)


## get tfidf model trained from given directory 'dirname' 
	def get_tfidf_model(self, dirname):
		data = Sentences(dirname)
		tfidf_vectorizer = TfidfVectorizer(stop_words='english')
		tfidf_matrix = tfidf_vectorizer.fit_transform(data)
		mat_array = tfidf_matrix.toarray()
		fn = tfidf_vectorizer.get_feature_names()
		return tfidf_vectorizer


#train model based on top N wwords from TFIDF model	
	def predict_model(self, dirname, w2v_model_path,topN,ndim):
		tfidf_model = self.get_tfidf_model(dirname)
		X = []
	#	label_data = []
		fn = []
		for fname in os.listdir(dirname):
			print "Procession = " + fname
			f = open(os.path.join(dirname, fname),"r")
			raw_text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(raw_text)
			nword = self.top_n_words_doc(text,tfidf_model,topN)
			X_coeff = self.gen_LR_weight( nword,text, w2v_model_path,tfidf_model, ndim)
	#		if fname[-10:-4] == 'accept':
	#			label = 1
	#		else:
	#			label = 0
			fn.append(fname)
			X.append(X_coeff[0])
	#		label_data.append(label)
		return X, fn


# find top N words from TFIDF model
	def top_n_words_doc(self,text,tfidf_model,topn=20):
		
		token = text.split()
		words = {}

		for w in token:
			wt = tfidf_model.idf_[tfidf_model.vocabulary_[w]] if w in tfidf_model.vocabulary_ else 0
			words[w] = wt

		lenw = len(words)
		if (lenw < topn): topn = lenw

		sorted_x = sorted(words.items(), key=operator.itemgetter(1),reverse=True)
		listd = sorted_x[0:topn]

		word= []
		for i in range((topn)):
			word.append(listd[i][0])

		return word


# Generate wieghts from Logistic Regression Model
	def gen_LR_weight(self, words,text, w2v_model,tfidf_model, ndim):
		model = self.load_w2vmodel(w2v_model)
		text = text.lower().split()

		notfound = 0
		denom = 0
		nvecs = 0
		label = []
		avg_vec = []
		wt =0
		for w in text:

			if w in model:

				wt = tfidf_model.idf_[tfidf_model.vocabulary_[w]] if w in tfidf_model.vocabulary_ else 0
				ve = np.multiply(model[w], wt)

				avg_vec.append(ve)				
				if w in words: 
					label.append(1)
				else: 
					label.append(0)
				nvecs += 1
				denom += wt
			else:
				notfound += 1
		if denom == 0:
			return []

		logit = LogisticRegression(C=1.0).fit(avg_vec, label)

		return logit.coef_

		
		
class genTopNVec:

	def __init__(self,train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN):
		self.train_dirname = train_dirname
		self.test_dirname = test_dirname
		self.predict_dirname = predict_dirname
		self.w2v_model_path = w2v_model_path
		self.size = size
		self.topN = topN
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

		f = []
		self.x_wt, self.Ylabels,self.fn_train  = td.train_model( self.train_dirname, self.w2v_model_path,self.topN,self.size)

		
		self.xTest_wt,self.Y_test,self.fn_test = td.train_model(self.test_dirname, self.w2v_model_path,self.topN,self.size)


		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xTest_wt)

		self.result = self.getAnalysis(self.Y_test,self.Y_pred)


	def train_predict(self):
		td = TrainData()

		f = []
		self.x_wt, self.Ylabels,self.fn_train  = td.train_model( self.train_dirname, self.w2v_model_path,self.topN,self.size)

		pd = PredictData()
		self.xPred_wt,self.fn_test = pd.predict_model(self.predict_dirname, self.w2v_model_path,self.topN,self.size)


		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xPred_wt)

		fp = "pred_output_topN_" + str(self.topN) + ".tsv"

		self.savePredictResult2File(self.fn_test,self.Y_pred,fp)

	
	

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
		return matthews_corrcoef( true_pred,y_pred1),roc_auc_score( true_pred,y_pred1),precision[0],precision[1],recall[0],recall[1],fscore[0],fscore[1],support[0],support[1]

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

	def savePredictResult2File(self,fn_test,Y_pred,fname):
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tpredicted\n"))
		print Y_pred
		for i in range(len(Y_pred)):
			input_filename.write((fn_test[i]+"\t"+str(self.Y_pred[i])+"\n"))
		input_filename.close()
		
		
	def saveNFoldResult2file(self,y_test,y_pred,fn_test,fname):
		
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tlabelled\tpredicted\n"))

		for i in range(len(y_test)):
			input_filename.write((fn_test[i]+"\t"+str(y_test[i])+"\t"+str(y_pred[i])+"\n"))
		input_filename.close()


	def NFoldTest(self, iter_N=5,split =0.30,random_state=0):

		total_dirname = '/home/viswanath/workspace/test_resume/train'
		td = TrainData()
		print "extracting data"
		x_total, y_total, fn_total  = td.train_model( total_dirname, self.w2v_model_path,self.topN,self.size)
		print "random split"
		kf_total = cross_validation.ShuffleSplit(len(x_total), n_iter=iter_N, test_size=split,   random_state=random_state)
		x_tot_np = Math.array(x_total)
		y_tot_np = Math.array(y_total)
		
		j =0
		fn_test = []
		fnRes = "Output_topN_" + str(self.topN)+"_split_"+str(split*100) + ".tsv"
		fp = open(fnRes,"wb")
		fp.write("matthews_corrcoef\troc_auc_score\tprecision[0]\tprecision[1]\trecall[0]\trecall[1]\tfscore[0]\tfscore[1]\tsupport_0\tsupport_1\n")
		for train, test in kf_total:
			j += 1
			print "Case:: " + str(j) +"\n\n"
			lgr = LogisticRegression(C=1.0).fit(x_tot_np[train],y_tot_np[train])
			y_pred = lgr.predict(x_tot_np[test])
			y_test = y_tot_np[test]
			k = 0
			fn_test = []
			for i in test:
				fn_test.append(fn_total[i])
				k += 1

		#	fn = "Output_topN_" + str(j) + ".tsv"
			print "writing data to file"
			#self.saveNFoldResult2file(y_test,y_pred,fn_test,fn)
			#self.printAnalysis(y_test,y_pred)
			matthews_corrcoef, roc_auc_score, precision_0 , precision_1 , recall_0 , recall_1 , fscore_0 , fscore_1 , support_0, support_1 = self.getAnalysis(y_test,y_pred)
			Resultstr = str(matthews_corrcoef)+"\t"+str(roc_auc_score)+"\t"+str(precision_0)+"\t"+str(precision_1)+"\t"+str(recall_0)+"\t"+str(recall_1)+"\t"+str(fscore_0)+"\t"+str(fscore_1)+"\t"+str(support_0)+"\t"+str(support_1)+"\n"
			fp.write(Resultstr)

if __name__ == '__main__': 


	train_dirname = '/home/viswanath/workspace/test_resume/train'
#	test_dirname = '/home/viswanath/workspace/resume_data/res_dir/test'
#	total_dirname = '/home/viswanath/workspace/resume_data/res_dir/total'
	test_dirname = ''
	predict_dirname = '/home/viswanath/workspace/test_resume/predict'
	w2v_model_path = '/home/viswanath/workspace/test_resume/model/w2v_model_100.mod'
	size = 100
	topNA = [300]  
	for topN in topNA:
		print "\nFor TopN N=" + str(topN) + "\n"
		gt = genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.NFoldTest(iter_N=10,split =0.40)

#	gt.printResult()
#	gt.saveResult2file("topN_result.tsv")
#	gt.NFoldTest()
#	print gt.getResult()
