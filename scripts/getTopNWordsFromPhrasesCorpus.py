#https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
#Usage python getTopNWordsFromPhrasesCorpus.py cleanedPhrasesCorpus.txt
from sklearn.feature_extraction.text import CountVectorizer
import sys
import codecs

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


input_file = sys.argv[1]
fp = codecs.open(input_file, 'r', 'utf-8')

#cars_for_sell = [line.replace("\n", "") for line in open("cars_for_sell.txt")]
phrasesCorpusList = [line.replace("\n", "") for line in fp]
common_words = get_top_n_words(phrasesCorpusList, 300)
for word, freq in common_words:
    print(word, freq)
