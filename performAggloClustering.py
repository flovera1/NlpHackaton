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


directory = "/home/tauseef/dbrc/hund/txtFiles/oldFiles/"
#directory = "/home/tauseef/dbrc/hund/"

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



#model = gensim.models.Word2Vec(sentences_split, min_count=2)
model = gensim.models.Word2Vec(sentences_split, size=50, min_count=3, sg=1)

def vectorizer(sent, m):
    vec =[]
    numw = 0
    for w in sent:
        try:
            if numw==0:
                vec = model[w]
            else:
                vec = np.add(vec, model[w])
            numw = numw+1
        except:
            pass

    return np.asarray(vec)  / numw

l=[]

for i in sentences_split:
    l.append(vectorizer(i, model))

X = np.array(l)

print(l)
#print(X)

n_clusters = 8
clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
labels = clf.fit_predict(X)
print("labels", labels)

for index, sent in enumerate(sentences_split):
    print(str(labels[index]) + ":" + str(sent))



