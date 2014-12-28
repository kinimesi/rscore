rscore
======

rscore is a new approach to readability score and readability level classification. It calculates readability score based on English word frequency and predictes readability level by using English word frequency and tf-idf scores as features. Google unigram data is used for word frequency. 

###Required libraries:
```
string
pickle
nltk
scikit-learn
```
###How to use RScore:
```Python
>>> import RScore  # import RScore
>>> demo = RScore()  # create object
>>> demo.rscore('text.txt')  # calculate score of text file
```
###How to use Classifier:
```Python
>>> import Classifier  # import Classifier
>>> demo = Classifier()  # create object
>>> demo.train()  # train the classifier  (might take some time depending on the size of training dataset)
>>> demo.predict('text.txt') # predict the level
```
Installing:
-----
```
sudo python setup.py build
sudo python seup.py install
```
###Future ideas:

- Implement bigram, trigram or four-gram.
- Build a website to make it easier for the use of nontechnical people.
- Increse training data set size for better performance
