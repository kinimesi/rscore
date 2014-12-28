#!/usr/bin/env python
__author__ = 'ilkin safarli'

import unittest
from Classifier import Classifier


class TestClassifier(unittest.TestCase):

	def test_predict(self):
		x = Classifier()
		x.train()
		predicted = x.predict("train", "directory")
		actual = [(u'intermediate test', 2), (u'elementary test', 1), (u'advanced test', 0)]
		self.assertEqual(predicted, actual)


if __name__ == '__main__':
	unittest.main()
