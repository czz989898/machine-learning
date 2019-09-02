from prepare_data import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
print('start')
flag = True
'''
clf = MultinomialNB()
print('set up model')
classes = ''
with open(r'E:\home work\semester3\machine learning\assignment\data\labels.txt',encoding='UTF-8') as labels:
    for i in labels:
        classes+=i
    classes = classes.split(' ')
print('get classes')
count =-1
'''
##########################################train#######################################################
'''
print('training start')
while True:
    count+=1
    print(count)
    #if count>20:break
    x_batch,y_batch,flag = get_batch(50)
    if not flag:break
    clf.partial_fit(x_batch, y_batch, classes=classes)
joblib.dump(clf, "bayes_version2.0.m")
################################test accracy#####################
x_batch,y_batch = get_batch_accuracy(300,400)
y_pre = clf.predict(x_batch)
score = accuracy_score(y_batch,y_pre)
print(score)
'''
################################output#####################
clf = joblib.load("bayes_version2.0.m")
print('output start')
result = []
count = -1
while True:
    count+=1
    x_batch,flag = get_test_batch(100)
    if not flag: break
    y_pre = clf.predict(x_batch)
    print(count)
    result+=list(y_pre)
    #if count>2:break
df = pd.DataFrame(result)
df.to_csv(r'E:\home work\semester3\machine learning\assignment\data\result.csv',index=True)
