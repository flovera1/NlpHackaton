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


directory = "../data/tauExptData/threeSent/"
#directory = "../data/tauExptData/hundSent/"

filesno=0
textSents = []
filelist = os.listdir(directory)

for docname in filelist:
    fullpath = directory+docname
    if fullpath.endswith("txt"):
        print(fullpath)
        f = open(fullpath, 'r')
        filesno+=1
        textSents += f.readlines()
		#print(textSents)
        
print("Total number of files",filesno)




'''
These function below are for supporting'text_to_phrases' function
'''
# identify all possible phrases
def key_words_phrases(raw):
    ngramlist=[]
    x=minlen
    ngramlimit = maxlen
    tokens=nltk.word_tokenize(raw)

    while x <= ngramlimit:
        ngramlist.extend(nltk.ngrams(tokens, x))
        x+=1
    return ngramlist

# join words into a new list
def concat_words(wordlist):
    new_list = []
    for words in wordlist:
        new_list.append(' '.join(words))   
    return new_list



# define maximum and minimum number of words in one phrase
maxlen = 10 
minlen = 4 

def text_to_phrases(text):
    phrases = []
    for sentence in text:
        if len(str(sentence).split(' ')) <= maxlen:
            phrases.append(sentence)
        else:
            wordlist = key_words_phrases(sentence)
            phrases += concat_words(wordlist)
    
    print(len(phrases))
    print("Phrase length obtained")
    return phrases


phrases = text_to_phrases(textSents)


#phrases = ['gemeente TESSENDERLO , vertegenwoordigd', 'TESSENDERLO , vertegenwoordigd door', ', vertegenwoordigd door het', 'vertegenwoordigd door het college', 'door het college van', 'de gemeente TESSENDERLO , vertegenwoordigd', 'gemeente TESSENDERLO , vertegenwoordigd door', 'TESSENDERLO , vertegenwoordigd door het', ', vertegenwoordigd door het college', 'vertegenwoordigd door het college van', 'de gemeente TESSENDERLO , vertegenwoordigd door', 'gemeente TESSENDERLO , vertegenwoordigd door het', 'TESSENDERLO , vertegenwoordigd door het college', ', vertegenwoordigd door het college van', 'de gemeente TESSENDERLO , vertegenwoordigd door het', 'gemeente TESSENDERLO , vertegenwoordigd door het college', 'TESSENDERLO , vertegenwoordigd door het college van', 'de gemeente TESSENDERLO , vertegenwoordigd door het college', 'gemeente TESSENDERLO , vertegenwoordigd door het college van', 'de gemeente TESSENDERLO , vertegenwoordigd door het college van', 'burgemeester en schepenen \n', ' \n', 'verwerende partij \n', ' \n', '\n', ' \n', '\n', ' \n', '\n', ' \n', ' \n', 'In zake: \n']



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')




def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
	#text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
	#text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace


newphrases=[]
for items in phrases:
	newphrases.append(clean_text(items))

#print("newphrases", newphrases)

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('dutch')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
	#doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


norm_corpus = normalize_corpus(newphrases)

#https://kite.com/python/answers/how-to-remove-empty-strings-from-a-list-of-strings-in-python
norm_corpus_sans_empty_strings = [string for string in norm_corpus if string != ""]
norm_corpus = norm_corpus_sans_empty_strings
print("Normalised corpus")
print(norm_corpus)
print("lenght of norm corpus", len(norm_corpus))


#https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa

#https://stackoverflow.com/questions/41462711/python-calculate-hierarchical-clustering-of-word2vec-vectors-and-plot-the-resu
sentences_split = [s.lower().split(' ') for s in norm_corpus]
print(sentences_split)







from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#output of sentences_split expected as -->
#[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]

#now treat sentences obtained as documents on their own
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences_split)]

print (documents)
"""
output
[TaggedDocument(words=['human', 'interface', 'computer'], tags=[0]), TaggedDocument(words=['survey', 'user', 'computer', 'system', 'response', 'time'], tags=[1]), TaggedDocument(words=['eps', 'user', 'interface', 'system'], tags=[2]), TaggedDocument(words=['system', 'human', 'system', 'eps'], tags=[3]), TaggedDocument(words=['user', 'response', 'time'], tags=[4]), TaggedDocument(words=['trees'], tags=[5]), TaggedDocument(words=['graph', 'trees'], tags=[6]), TaggedDocument(words=['graph', 'minors', 'trees'], tags=[7]), TaggedDocument(words=['graph', 'minors', 'survey'], tags=[8])]
 
"""
 
model = Doc2Vec(documents, size=10, window=2, min_count=1, workers=4)
#Persist a model to disk:
 
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("my_doc2vec_model")
 
print (fname)
#output: C:\Users\userABC\AppData\Local\Temp\my_doc2vec_model
 
#load model from saved file
model.save(fname)
model = Doc2Vec.load(fname)  
# you can continue training with the loaded model!
#If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
 
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
 
#Infer vector for a new document:
#Here our text paragraph just 2 words
vector = model.infer_vector(["Vergunningsbetwistingen", "samengesteld"])
print (vector)



