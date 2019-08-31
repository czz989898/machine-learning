import scipy as sp
import numpy as np
from beyesi_get_batch import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


clf = MultinomialNB()
print('set up model')

flag=True
while flag:
    batch_xs, batch_ys,flag = get_batch_train(1000)
    print('get data success')
    #scores = cross_val_score(clf, batch_xs, batch_ys, cv=5)
    #print(scores)
    #print(scores.mean())
    clf = MultinomialNB().partial_fit(batch_xs, batch_ys,classes=np.unique(batch_ys))
with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
    config.write(str(0))
