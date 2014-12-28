#!/usr/bin/env python
__author__ = 'ilkin safarli'

import string, pickle
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class RScore(object):
	"""
	This class calculates readability score - based on English word frequency.
	"""
	def __init__(self):
		self.score = 0
		self.freq = pickle.load(open("word_freq.db","rb"))

	def clear_text(self, file_name):
		"""
		Clear all punctuations
		:param file_name: name of file
		:return sting without punctuations
		"""
		text = open(file_name, 'r').read()
		text = text.lower()
		no_punctuation = text.translate(None, string.punctuation)
		return no_punctuation

	def tokenize(self, text):
		"""
		Tokenize text into words.
		:param text: string
		:return: tokenized text
		"""
		tokens = word_tokenize(text)
		return tokens

	def tf_idf(self, file_name):
		"""
		Calculate tf-idf score.
		:param file_name: name of file
		:return: dictionary where keys are words and values are tf scores.
		"""
		tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english', norm="l1")
		tfs = tfidf.fit_transform([self.clear_text(file_name)]).toarray()
		word_tfidf = dict(zip(tfidf.get_feature_names(), tfs[0]))
		return word_tfidf

	def rscore(self, file_name):
		"""
		Calculates readability score.
		:param file_name: name of file
		:return: readability score
		"""
		self.score = 0
		word_tfidf = self.tf_idf(file_name)
		for word in word_tfidf:
			try:
				self.score += self.freq[word]*word_tfidf[word]
			except:
				continue
		return (self.score*10000.0)/len(word_tfidf)  # scaling result
