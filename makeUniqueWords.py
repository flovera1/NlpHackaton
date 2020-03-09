import os
import subprocess

if __name__ == '__main__':
    	# Get all the PDF filenames.
    txtFiles = []
    for filename in os.listdir('.'):
        if filename.endswith('.txt'):
             txtFiles.append(filename)

    for filename in txtFiles:
        txtFileObj = open(filename, 'rb')
        subprocess.call(r"./wordForm.sh %s"%txtFileObj, shell=True)
        txtFileObj.close()
