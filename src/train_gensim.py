#!/usr/bin/python


import os,re,gensim, string
from nltk.corpus import stopwords


def cleanse_data(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

#	print "cleansing"
	temp = text
#	temp =  re.sub(r"(c|C)\+\+",'cpp',temp)
	temp = re.sub(r'(\d+(\s)?(yrs|year|years|Yrs|Years|Year|yr))'," TIME ",temp)
	temp = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",temp)
	temp = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',temp)
	temp = re.sub(r"[\r\n]+[\s\t]+",'\n',temp)	
	wF = set(string.punctuation) - set(["+"])
	for c in wF:
        	temp =temp.replace(c," ")	

	return temp


def scan_file(dir_name):

##
	##  scan every file in a directory 
	##  and extract text from it
##
	for fname in os.listdir(dir_name):
		fp = open(os.path.join(dir_name, fname),"r")
		text = fp.read()
		yield cleanse_data(text)

def save_file(text,fname):

##
	##  save the files 
	##  text is text and fname is the path and name of the file to be save 
##
	fp = open(fname,"a")
#	fp.write("\n\n ********************** NEW FILE **********************\n\n")
	fp.write(text)
	fp.write("\n")
	fp.close()

def words(stringIterable):
    #upcast the argument to an iterator, if it's an iterator already, it stays the same
    lineStream = iter(stringIterable)
    for line in lineStream: #enumerate the lines
        for word in line.split(" \n\t"): #further break them down
            yield word

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname
	
	def __iter__(self):
		count = 0
		for fname in os.listdir(self.dirname):
			count = count +1
			print fname, count
			for line in open(os.path.join(self.dirname, fname)):
				
				line = cleanse_data(line)
				yield line.lower().split()



def main():
	dir_name = '/home/viswanath/workspace/resume_data/rejected_profiles'
#	text_seq = scan_file(dir_name)
#	wordlist = []
#	model = gensim.models.Word2Vec()
#	text_ = ''
#	for text in text_seq:
#		if 'java' in text:
#			print 'JAVA'
	#	print text
#		text_ += text
#		save_file(text,os.path.join(dir_name, "cleanse_data.txt"))
#	text_s = text_.split()
#	for word in words(text_):
#		wordlist.append(word)

#	wordlist = re.sub("[^\w]", " ",  text_).split()


	sentences = MySentences(dir_name) # a memory-friendly iterator
#	print "model training starting"
	model = gensim.models.Word2Vec(sentences,min_count =1,workers=4)

#	for w in wordlist[:-10]:
#		print w
#	model = gensim.models.Word2Vec(wordlist,workers=4)
#	model.save('/home/viswanath/workspace/myModel')
	outf = '/home/viswanath/workspace/myModel_reject'
#	print outf
#	print "saving model "
#	model.save_word2vec_format(outf, fvocab=None, binary=False)
	print model
#	model.init_sims(replace=True)
#	model.similarity('java','scala')
	print model.most_similar_cosmul(positive=['java'])
#	if "java" in model:
#		print "Result mil gya \n"


if __name__ == "__main__": main()
