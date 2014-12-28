#!/usr/bin/env python
__author__ = 'ilkin safarli'

import unittest
from RScore import *


class TestRScore(unittest.TestCase):

	def test_tokenize(self):
		x=RScore()
		self.assertEqual(["i", "love", "computer", "science"], x.tokenize("i love computer science"))

	def test_clear_text(self):
		x=RScore()
		self.assertEqual("hello world", x.clear_text("test_rscore.txt"))

	def test_tf_idf(self):
		x=RScore()
		self.assertEqual({"world" : 0.5, "hello" : 0.5}, x.tf_idf("test_rscore.txt"))

	def test_rscore(self):
		x=RScore()
		self.assertEqual(5000, x.rscore("test_rscore.txt"))

if __name__ == '__main__':
	unittest.main()
