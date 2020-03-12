import json
import os

#import nltk
from nltk.corpus import stopwords,wordnet
from nltk import pos_tag,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


import re
import string

import nltk
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


directory = "/home/tauseef/dbrc/hund/txtFiles/oldFiles/"
#directory = "/home/tauseef/dbrc/hund/"




def wordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

documentDict=dict()
for filename in os.listdir(directory):
    if filename[-3:] == 'txt':
        with open(os.path.join(directory,filename),'r') as infile:
			#documentDict[filename]=infile.read()
            dataFile = infile.read()
            whole_text = ''.join(dataFile)
			#docText.append(whole_text)
            documentDict[filename]=whole_text

print(documentDict)


print("Cleaning....")

#https://www.science-emergence.com/Articles/How-to-remove-string-control-characters-n-t-r-in-python/
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
NEWLINE_SYMBOLS_RE = re.compile(r'[\n]')
#STOPWORDS = nltk.corpus.stopwords.words('dutch')
STOPWORDS = set(stopwords.words('dutch'))


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
	#text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = text.strip() # remove newline \n
    text = text.rstrip() # remove newline \n characters anywhere in string
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = NEWLINE_SYMBOLS_RE.sub('', text) # delete newline symbols
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation]) 
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace


#adapted from http://brandonrose.org/clustering_mobile
def string_tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


documents=[]
for filename,docutext in documentDict.items():
    cleansedText = clean_text(docutext)
	#tokens=tokenize(cleansedText)
    stringtokens=string_tokenize_only(cleansedText)
    tagged_tokens=pos_tag(cleansedText)
    lemma=WordNetLemmatizer()
    stemmedTokens = [lemma.lemmatize(word, wordnetPos(tag)).lower()
                     for word, tag in tagged_tokens]
    documents.append({
        'filename': filename,
        'text': docutext,
        'words': stemmedTokens,
    })
with open('all_stories.json', 'w') as outfile:
    outfile.write(json.dumps(documents))
print('Cleaning is done!')

