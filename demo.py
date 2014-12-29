#!/usr/bin/env python
__author__ = 'ilkin safarli'

import RScore
import Classifier

demo = RScore.RScore()
print(demo.rscore("demo_2.txt"))

demo_classify = Classifier.Classifier()
demo_classify.train()
demo_classify.predict("demo_2.txt")