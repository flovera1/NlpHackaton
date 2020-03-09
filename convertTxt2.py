import os
from os.path import isfile, join
import fileinput

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


class PdfConverter:

   def __init__(self, file_path):
       self.file_path = file_path
# convert pdf file to a string which has space among words 
   def convert_pdf_to_txt(self):
       rsrcmgr = PDFResourceManager()
       retstr = StringIO()
       codec = 'utf-8'  # 'utf16','utf-8'
       laparams = LAParams()
       device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
       fp = open(self.file_path, 'rb')
       interpreter = PDFPageInterpreter(rsrcmgr, device)
       password = ""
       maxpages = 0
       caching = True
       pagenos = set()
       for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
           interpreter.process_page(page)
       fp.close()
       device.close()
       str = retstr.getvalue()
       retstr.close()
       return str
# convert pdf file text to string and save as a text_pdf.txt file
   def save_convert_pdf_to_txt(self,filename):
       content = self.convert_pdf_to_txt()
       newSavedFile = str(filename) + '.txt'
       txt_pdf = open(newSavedFile, 'wb')
       txt_pdf.write(content.encode('utf-8'))
       txt_pdf.close()





#print(pdfFiles)


if __name__ == '__main__':
    	# Get all the PDF filenames.
    pdfFiles = []
    for filename in os.listdir('.'):
        if filename.endswith('.pdf'):
             pdfFiles.append(filename)

    for filename in pdfFiles:
		#pdfFileObj = open(filename, 'rb')
        pdfConverter = PdfConverter(file_path=filename)
		#print(pdfConverter.convert_pdf_to_txt())
        pdfConverter.save_convert_pdf_to_txt(filename)
