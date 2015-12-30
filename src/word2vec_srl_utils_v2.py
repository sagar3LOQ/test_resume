
import sys
from gensim.models import Word2Vec
from sklearn.externals import joblib
from gensim import utils, matutils
import scipy
import numpy as np
import fastcluster
import scipy.cluster.hierarchy
import scipy.cluster.hierarchy as sch 
from scipy.ndimage import convolve
from pprint import pprint
from configobj import ConfigObj
import traceback

__author__ = 'sagar sahu'
__meta_data__ = 'based on http://www.ofai.at/~marcin.skowron/papers/Wu_Skowron_Petta-Reading_Between_the_Lines.pdf'

class FeatureUtils():

    def __init__(self):
        pass

    def get_tfidf_weighted_vec(self, words, w2v_model, tfidf_model, ndim):
        try:
            nvecs = 0 
            notfound = 0 
            denom = 0 
            avg_vec = np.zeros((ndim), dtype='float32')
            for w in words:
                if w in w2v_model:
                    wt = tfidf_model.idf_[tfidf_model.vocabulary_[w]] if w in tfidf_model.vocabulary_ else 1
                    avg_vec = np.add(np.multiply(w2v_model[w], wt), avg_vec)
                    nvecs += 1
                    denom += wt
                else:
                    notfound += 1
            if denom == 0:
                return []
            else:
                return matutils.unitvec(np.divide(avg_vec, denom))
        except:
            raise


    def get_tfidf_srl_circonv_weighted_vec(self, words, w2v_model, tfidf_model, ndim,VA0,VA1):
        try:
            nvecs = 0 
            notfound = 0 
            denom = 0 
          #  print ndim
            avg_vec = np.zeros((ndim), dtype='float32')
            Wa0_vec = np.zeros((ndim), dtype='float32')
            Wa1_vec = np.zeros((ndim), dtype='float32')
            Wa0_vec_ = np.zeros((ndim), dtype='float32')
            Wa1_vec_ = np.zeros((ndim), dtype='float32')

            WaRem_vec = np.zeros((ndim), dtype='float32')
            explore = []

      #      print "part 1"

            for Wv in VA0:
                Wa0_vec = np.zeros((ndim), dtype='float32')
                for Wa0 in VA0[Wv]:
                    if Wa0 in w2v_model:
                        wt = tfidf_model.idf_[tfidf_model.vocabulary_[Wa0]] if Wa0 in tfidf_model.vocabulary_ else 1

                        Wa0_vec = np.add(np.multiply(w2v_model[Wa0], wt), Wa0_vec)
                        nvecs += 1
#                        denom += wt
                    else:
                        notfound += 1
                    explore.append(Wa0)
       #         print "\n\n" + Wa0
        #        print Wa0_vec
                if Wv in w2v_model: Wa0_vec_ += convolve(w2v_model[Wv],Wa0_vec, mode='wrap')	#np.multiply(w2v_model[Wv],Wa0_vec)
         #       print Wa0_vec_
                explore.append(Wv)

      #      print "part 2"

            for Wv in VA1:
                Wa1_vec = np.zeros((ndim), dtype='float32')
                for Wa1 in VA1[Wv]:
                    if Wa1 in w2v_model:
                        wt = tfidf_model.idf_[tfidf_model.vocabulary_[Wa1]] if Wa1 in tfidf_model.vocabulary_ else 1
                        Wa1_vec = np.add(np.multiply(w2v_model[Wa1], wt), Wa1_vec)
                        nvecs += 1
#                        denom += wt
                    else:
                        notfound += 1
                    explore.append(Wa1)
        #        print "\n\n" + Wa1
         #       print Wa1_vec
                if Wv in w2v_model: Wa1_vec_ += convolve(w2v_model[Wv],Wa1_vec, mode='wrap')     #np.multiply(w2v_model[Wv],Wa1_vec)
          #      print Wa1_vec_
                explore.append(Wv)

       #     print "part 3"

            for w in words:
                if w in explore: continue
                if w in w2v_model:
                    wt = tfidf_model.idf_[tfidf_model.vocabulary_[w]] if w in tfidf_model.vocabulary_ else 1
                    avg_vec = np.add(np.multiply(w2v_model[w], wt), avg_vec)
                    nvecs += 1
                 #   denom += wt
                else:
                    notfound += 1

            avg_vec += (Wa1_vec_+ Wa0_vec_)
           # print "\n\n\nAverage Vector\n"
           # print avg_vec
            return avg_vec
#            if denom == 0:
#                return []
#            else:
#                return matutils.unitvec(np.divide(avg_vec, denom))
        except:
            raise


    def get_avg_vec(self, words, w2v_model, ndim):
        try:
            nvecs = 0 
            avg_vec = np.zeros((ndim), dtype='float32')
            if type(words[0]) == type('str'):
                for w in words:
                    if w in w2v_model:
                        avg_vec = np.add(w2v_model[w], avg_vec)
                        nvecs += 1
            elif type(words[0]) == np.ndarray:
                for w in words:
                    avg_vec = np.add(w, avg_vec)
                    nvecs += 1
            else:
                pass
            if nvecs == 0:
                return []
            else:
                return matutils.unitvec(np.divide(avg_vec, nvecs))
        except:
            raise

    def find_clusters(self, X, cluster_max_dist=0.8):
        try:
            if len(X) == 0:
                return []
            if len(X) == 1:
                return [[0]]
            distance = scipy.spatial.distance.pdist(X, 'cosine')
            linkage = fastcluster.linkage(distance, method="complete")
            linkage = [[ele if ele > 0 else 0 for ele in row] for row in linkage]
            labels = sch.fcluster(linkage, cluster_max_dist * distance.max(), 'distance')
            clusters = [[] for i in set(labels)]
            for i, cid in enumerate(labels):
                clusters[cid - 1].append(i)
            return clusters
        except:
            raise
        

class DocumentFeatures():

    def __init__(self):
        self.fuo = FeatureUtils()

    def get_ngram_vecs(self, sent, w2v_model, ndim, N, wtmethod='avg', tfidf_model=None):
        try:
            x = sent.split()
            ngrams = [x[i:i + N] for i in xrange(len(x) - N + 1)]
            if wtmethod == 'tfidf':
                avg_ngrams = [self.fuo.get_tfidf_weighted_vec(ng, w2v_model, tfidf_model, ndim) for ng in ngrams]
            elif wtmethod == 'avg':
                avg_ngrams = [self.fuo.get_avg_vec(ng, w2v_model, ndim) for ng in ngrams]
            else:
                raise Exception("function parameter for wtmethod '%s' doesn't exist" % (wtmethod))
            avg_ngrams = [x for x in avg_ngrams if len(x) != 0]
            return avg_ngrams
        except:
            raise

    def get_sent_vec(self, sent, w2v_model, ndim, wtmethod='avg', tfidf_model=None):
        try:
            x = sent.split()
            if wtmethod == 'tfidf':
                avg_ngrams = [self.fuo.get_tfidf_weighted_vec(x, w2v_model, tfidf_model, ndim)]
            elif wtmethod == 'avg':
                avg_ngrams = [self.fuo.get_avg_vec(x, w2v_model, ndim)]
            else:
                raise Exception("function parameter for wtmethod '%s' doesn't exist" % (wtmethod))
            avg_ngrams = [x for x in avg_ngrams if len(x) != 0]
            return avg_ngrams
        except:
            raise


    def get_sent_circconv_vec(self, sent, w2v_model, ndim, wtmethod='avg', tfidf_model=None,VA0={},VA1={}):
        try:
            x = sent.split()
            if wtmethod == 'tfidf':
                avg_ngrams = [self.fuo.get_tfidf_srl_circonv_weighted_vec(x, w2v_model, tfidf_model, ndim,VA0,VA1)]
#            elif wtmethod == 'avg':
 #               avg_ngrams = [self.fuo.get_avg_vec(x, w2v_model, ndim)]
            else:
                raise Exception("function parameter for wtmethod '%s' doesn't exist" % (wtmethod))
            avg_ngrams = [x for x in avg_ngrams if len(x) != 0]
            return avg_ngrams
        except:
            raise


    def get_cluster_vecs(self, sent, w2v_model, ndim, wtmethod='avg', tfidf_model=None):
        try:
            x = [w for w in sent.split() if w in w2v_model]
            vecs = [w2v_model[i] for i in x if i in w2v_model]
            cluster_vecs = self.fuo.find_clusters(vecs)
            if wtmethod == 'tfidf':
                avg_cluster_vecs = [self.fuo.get_tfidf_weighted_vec([x[j] for j in i], w2v_model, tfidf_model, ndim) for i in cluster_vecs]
                pass
            elif wtmethod == 'avg':
                avg_cluster_vecs = [self.fuo.get_avg_vec([vecs[j] for j in i], w2v_model, ndim) for i in cluster_vecs]
            else:
                raise Exception("function parameter for wtmethod '%s' doesn't exist" % (wtmethod))
            return avg_cluster_vecs
        except:
            raise

