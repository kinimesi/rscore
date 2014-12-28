#!/usr/bin/env python
__author__ = 'ilkin safarli'

import unittest
from FrequencyTransformer import *
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
counts = count_vect.fit_transform(["hello world"])


class TestFrequencyTransformer(unittest.TestCase):

	def test_fit(self):
		x = FrequencyTransformer(count_vect)
		self.assertEqual(x, x.fit())

	def test_transform(self):
		global count_vect
		x = FrequencyTransformer(count_vect)
		c = sparse.csc_matrix([1, 1])
		boolean = (c-x.transform(counts.toarray())).nnz == 0
		self.assertEqual(boolean, True)

if __name__ == '__main__':
	unittest.main()

