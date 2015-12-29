
import sys
import gensim
from gensim.models import Word2Vec
from sklearn.externals import joblib
from gensim import utils, matutils
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
	temp = re.sub(r'[^\x00-\x7F]+',' ', text)
	temp = re.sub(r'(\d+(\s)?(yrs|year|years|Yrs|Years|Year|yr))'," TIME ",temp)
	temp = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",temp)
	temp = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',temp)
	temp = re.sub(r"[\r\n]+[\s\t]+",'\n',temp)	
	wF = set(string.punctuation) - set(["+"])
	for c in wF:
        	temp =temp.replace(c," ")	

	return temp


## Training data class

class TrainData():

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
	def train_model(self, dirname, w2v_model_path,topN,ndim):
		tfidf_model = self.get_tfidf_model(dirname)
		X = []
		label_data = []
		for fname in os.listdir(dirname):
			f = open(os.path.join(dirname, fname),"r")
			raw_text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(raw_text)
			nword = self.top_n_words_doc(text,tfidf_model,topN)
			X_coeff = self.gen_LR_weight( nword,text, w2v_model_path,tfidf_model, ndim)
			if fname[-10:-4] == 'accept':
				label = 1
			else:
				label = 0
			X.append(X_coeff[0])
			label_data.append(label)
		return X, label_data


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
		temp = text.lower().split()

		notfound = 0
		denom = 0
		nvecs = 0
		label = []
		avg_vec = []
		wt =0
		for w in temp:

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

	def __init__(self,train_dirname,test_dirname,w2v_model_path,size,topN):
		self.train_dirname = train_dirname
		self.test_dirname = test_dirname
		self.w2v_model_path = w2v_model_path
		self.size = size
		self.topN = topN
		self.x_wt = []
		self.Ylabels = []
		self.xTest_wt = []
		self.Y_test = []
		self.Y_pred = []
		self.result = []

	def start(self):
		td = TrainData()

		self.x_wt, self.Ylabels = td.train_model( self.train_dirname, self.w2v_model_path,self.topN,self.size)


		self.xTest_wt,self.Y_test = td.train_model(self.test_dirname, self.w2v_model_path,self.topN,self.size)

		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xTest_wt)

		self.result = self.getAnalysis(self.Y_test,self.Y_pred)
	

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


if __name__ == '__main__': 
	train_dirname = '/home/viswanath/workspace/resume_data/res_dir/train'
	test_dirname = '/home/viswanath/workspace/resume_data/res_dir/test'
	w2v_model_path = '/home/viswanath/workspace/resume_data/res_dir/model/w2v_model.mod'
	size = 100
	topN = 200

	gt = genTopNVec(train_dirname,test_dirname,w2v_model_path,size,topN)
	gt.start()
	gt.printResult()
	print gt.getResult()
