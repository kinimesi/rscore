rscore
======

rscore is a new approach to readability score. It calculates readability score based on English word frequency. Google unigram data is used for word frequency. 

###Required libraries:
```
string
pickle
nltk
scikit-learn
```
###How to use:
```Python
>>> import RScore  # import RScore
>>> demo = RScore()  # create object
>>> demo.rscore('text.txt')  # calculate score of text file
```

###Future ideas:

- Implement bigram, trigram or four-gram.
