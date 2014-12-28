#!/usr/bin/env python
__author__ = 'ilkin safarli'

import numpy as np
import pickle
from scipy import sparse


class FrequencyTransformer(object):
	"""
	This is a custom transformer for scikit-learn: creates word frequency array for each document.
	"""
	def __init__(self, count_vect):
		"""
		:param count_vect: vector counter object
		"""
		self.word_freq = pickle.load(open("word_freq.db", "rb"))
		self.count_vect = count_vect

	def fit(self, x = None, y = None):
		"""
		This transformer does not need to be fitted, hence return self
		"""
		return self

	def transform(self, array):
		"""
		This function creates frequency array.
		:param array: an array where rows are documents and columns are word counts
		:return: frequency matrix
		"""
		freq_array = []
		for doc in array:
			tmp = []
			word_tfidf = dict(zip(self.count_vect.get_feature_names(), doc))
			for word in word_tfidf:
				if word_tfidf[word] != 0 and word in self.word_freq:
					tmp.append(self.word_freq[word])
				else:
					tmp.append(0)
			freq_array.append(tmp)
		B = np.array(freq_array)
		return sparse.csr_matrix(B)