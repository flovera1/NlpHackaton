import pandas as pd
import re
import string
import numpy as np
import os
import nltk

#nltk.download('stopwords')



#from nltk.corpus import gutenberg
from string import punctuation

import gensim
from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage


from sklearn.cluster import KMeans
from sklearn import cluster


from sklearn.decomposition import PCA



#https://ai.intelligentonlinetools.com/ml/text-clustering-doc2vec-word-embedding-machine-learning/
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("my_word2vec_model")

#load model from saved file
print("Save Word2Vec trained model")
#model.save(fname)
model = gensim.models.Word2Vec.load(fname)

#model.delete_temporary_training_data(keep_wordtags_vectors=True, keep_inference=True)

#Building depth
w1 = "bouwdiepte"
print("top 6 similar words to bouwdiepte", model.wv.most_similar(positive=w1,topn=6))

#Decision
w2 = "beslissing"
print("top 6 similar words to beslissing", model.wv.most_similar(positive=w2,topn=6))

#Permit decision
w3 = "vergunningsbeslissing"
print("top 6 similar words to vergunningsbeslissing", model.wv.most_similar(positive=w3,topn=6))


w4 = "winkelruimte"
print("top 6 similar words to winkelruimte", model.wv.most_similar(positive=w4,topn=6))
