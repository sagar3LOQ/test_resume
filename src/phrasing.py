

import sys
import gensim
from streamer import Sentences
from configobj import ConfigObj

def phrase_detection(raw_in_file):
	sents = Sentences(raw_in_file, False)

	bigram_transformer = gensim.models.Phrases(sents, min_count=5, threshold = 2, delimiter='_')
	print bigram_transformer

	config = ConfigObj('./config.ini')

	input_data_file = config['preprocessing']['raw_in_file']
	out_file = config['preprocessing']['phrased_out_file']

	print '################ Writing phrased input to: [' + out_file + ']'
	with open(out_file, 'w') as f:
		for each in  bigram_transformer[sents]:
			f.write("%s\n"%each)
	print '################## Done'


if __name__ == '__main__':
	config = ConfigObj('./config.ini')
	raw_in_file = config['preprocessing']['raw_in_file']
	
	print '################### Phrase detection in: ' + raw_in_file
	phrase_detection(raw_in_file)

