Used the idea and GitHub code from https://github.com/sethuiyer/Document-Clusterer
to cluster data on arrests from DBRC Hackathon

First run the script makeBigBagUniqWordsFrmAllDocs.py which in turn calls a shell script. Goal is to get a list of unique unigrams from the corpus
of 100 arrests. Inspect the file allUnigrams.txt for correctness. Keep in mind some intermediary txt files are produced.


Then run the script cleaningOn100DBRCFiles.py to do further text processing using the unigrams and the arrests and make a dictionary
with keys=names of arrests and values = cleaned content of the arrests

Finally do the clustering using model.py

Tested on 100 arrests and clustered documents are kept in clusteredDocuments_3groups
