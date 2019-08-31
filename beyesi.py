import scipy as sp
import numpy as np
from beyesi_get_batch import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
batch_xs, batch_ys = get_batch_train()
print('get data success')
#clf = MultinomialNB().fit(batch_xs, batch_ys)
clf = MultinomialNB()
print('set up model')
scores = cross_val_score(clf, batch_xs, batch_ys, cv=5)
print(scores)
print(scores.mean())