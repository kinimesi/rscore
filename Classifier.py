#!/usr/bin/env python
__author__ = 'ilkin safarli'

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from FrequencyTransformer import *
from sklearn.datasets import load_files
from RScore import RScore


class Classifier(RScore):
	"""
	This class classifies text into 3 levels by using Naive Bayes classifier.
	Features: english word frequency and tf-idf score
	*** All methods of RScore class can be invoked too.
	"""
	def __init__(self):
		self.categories = ["elementary", "intermediate", "advanced"]
		self.count_vect = CountVectorizer()
		self.tf_transformer = TfidfTransformer(use_idf=True)
		RScore.__init__(self)

	def train(self):
		"""
		This method trains the classifier by using a given data set.
		:return: trained classifier
		"""
		self.train_data = load_files(container_path='train', description=None, categories=self.categories,shuffle=True)
		train_counts = self.count_vect.fit_transform(self.train_data.data)
		array = train_counts.toarray()
		self.freq_transformer = FrequencyTransformer(self.count_vect)
		self.combined_features = FeatureUnion([("frequency", self.freq_transformer), ("tfidf", self.tf_transformer)]).fit_transform(array)
		self.classifier = MultinomialNB()
		self.classifier.fit(self.combined_features, self.train_data.target)

	def predict(self, file_name, file_type = "text"):
		"""
		Predicts level of given text file or files.
		:param file_name: name of directory or text file.
		:return: predicted level
		"""
		if file_type == "text":
			text = [self.clear_text(file_name)]
		else:
			text = load_files(container_path=file_name, description=None, categories=self.categories,encoding="utf8").data
		new_counts = self.count_vect.transform(text)
		array = new_counts.toarray()
		res = FeatureUnion([("frequency", self.freq_transformer), ("tfidf", self.tf_transformer)]).transform(array)
		predicted = self.classifier.predict(res)
		for doc, category in zip(text, predicted):
			print('%r => %s' % (doc, self.train_data.target_names[category]))
		return zip(text, predicted)