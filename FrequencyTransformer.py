#!/usr/bin/env python

import numpy as np
import pickle
from scipy import sparse


class FrequencyTransformer(object):
	def __init__(self, count_vect):
		self.word_freq = pickle.load(open("word_freq.db", "rb"))
		self.count_vect = count_vect

	def fit(self, x = None):
		return self

	def transform(self, arr):
		array = []
		for doc in arr:
			tmp = []
			word_tfidf = dict(zip(self.count_vect.get_feature_names(), doc))
			for word in word_tfidf:
				if word_tfidf[word] != 0 and word in self.word_freq:
					tmp.append(self.word_freq[word])
				else:
					tmp.append(0)
			array.append(tmp)
		B = np.array(array)
		return sparse.csr_matrix(B)
