#import docx
#from docx import Document
import re
import string
import numpy as np
#import pandas as pd
import nltk
#nltk.download('punkt')


#Define a function that read docx file and save as text. Remove special punctuations and characters.

#wordcloud imports
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt


import os

directory = "../../data/tauExptData/hundSent/"

filesno=0
text = []
filelist = os.listdir(directory)

for docname in filelist:
    fullpath = directory+docname
    if fullpath.endswith("txt"):
        print(fullpath)
        f = open(fullpath, 'r')
        filesno+=1
        text += f.readlines()
		#print(text)
        
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

phrases = text_to_phrases(text)
print(phrases)
#print(phrases.values)
#print(type(phrases))

#######################################################################################################
#https://www.datacamp.com/community/tutorials/wordcloud-python
# Create and generate a word cloud image:
#wordcloud = WordCloud().generate(phrases)

# Display the generated image:
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()


#######################################################################################################
#https://stackoverflow.com/questions/45588724/generating-word-cloud-for-items-in-a-list-in-python
#convert list to string and generate
unique_string=(" ").join(phrases)
wordcloud = WordCloud(width = 1800, height = 1000).generate(unique_string)
plt.figure(figsize=(25,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("dbrc_100files_wc"+".png", bbox_inches='tight')
plt.show()
plt.close()


