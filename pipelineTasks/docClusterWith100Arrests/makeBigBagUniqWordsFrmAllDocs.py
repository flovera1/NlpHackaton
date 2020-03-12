import os
import subprocess

directory = "../../data/tauExptData/hundSent"

if __name__ == '__main__':
    	# Get all the PDF filenames.
    txtFiles = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
             txtFiles.append(os.path.join(directory, filename))

    for filename in txtFiles:
        print(filename)
		#txtFileObj = open(filename, 'rb')
		#subprocess.call(r"./aggregateAllWords.sh %s"%txtFileObj, shell=True)
        result = subprocess.run(['./aggregateAllWords.sh', filename, './aggregatedBagOfWords.txt'], stdout=subprocess.PIPE)
		#txtFileObj.close()
