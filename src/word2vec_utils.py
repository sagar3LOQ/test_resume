
import sys
from gensim.models import Word2Vec
from sklearn.externals import joblib
from gensim import utils, matutils
import scipy
import numpy as np
import fastcluster
import scipy.cluster.hierarchy
import scipy.cluster.hierarchy as sch 

from pprint import pprint
from configobj import ConfigObj
import traceback

__author__ = 'sudheerkovela'

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
                return np.divide(avg_vec, denom)
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
                return np.divide(avg_vec, nvecs)
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

