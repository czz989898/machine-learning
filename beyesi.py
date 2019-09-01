import scipy as sp
import numpy as np
from sklearn.externals import joblib
from beyesi_get_batch import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
'''
clf = MultinomialNB()
print('set up model')
flag=True
count = -1
while flag:
    count+=1
    if count>5:break
    batch_xs, batch_ys,flag = get_batch(1000)
    #print('get data success')
    #scores = cross_val_score(clf, batch_xs, batch_ys, cv=5)
    #print(scores)
    #print(scores.mean())
    clf = MultinomialNB().partial_fit(batch_xs, batch_ys,classes=np.unique(batch_ys))
with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
    config.write(str(0))
result = []
flag=True
joblib.dump(clf, "bayes_version1.2.m")
print('train success')
'''
#bayes try#####################################################
classes = ''
with open(r'E:\home work\semester3\machine learning\assignment\data\labels.txt',encoding='UTF-8') as labels:
    for i in labels:
        classes+=i
    classes = classes.split(' ')
clf = MultinomialNB()
print('set up model')
batch_xs, batch_ys,flag = get_batch(1)
with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
    config.write(str(1))
clf.partial_fit(batch_xs, batch_ys,classes=classes)
batch_xs, batch_ys,flag = get_batch(1)
with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
    config.write(str(1))
clf.partial_fit(batch_xs, batch_ys,classes=classes)
#test accuracy#################################################
print('test start')
batch_xs, batch_ys,flag = get_batch(1)
with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
    config.write(str(0))
y_pre = clf.predict(batch_xs)
for i in range(0,len(y_pre)):
    print(y_pre[i],batch_ys[i])
score =accuracy_score(batch_ys, y_pre)
print(score)
#get final result##############################################
'''
clf = joblib.load("train_model.m")
with open(r'E:\home work\semester3\machine learning\assignment\data\result.txt','w', encoding='UTF-8') as result:
    flag = True
    while flag:
        x_batch,flag=get_batch_test(1000)
        y_pre = clf.predict(x_batch)
        for j in y_pre:
            result.write(str(j)+'\n')
'''
