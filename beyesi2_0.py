from prepare_data import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
print('start')
flag = True
clf = MultinomialNB()
print('set up model')
classes = ''
with open(r'E:\home work\semester3\machine learning\assignment\data\labels.txt',encoding='UTF-8') as labels:
    for i in labels:
        classes+=i
    classes = classes.split(' ')
print('get classes')
count =-1
print('training start')
while True:
    count+=1
    print(count)
    if count>10:break
    x_batch,y_batch,flag = get_batch(50)
    if not flag:break
    clf.partial_fit(x_batch, y_batch, classes=classes)
#joblib.dump(clf, "bayes_version2.0.m")
################################test accuracy#####################
x_batch,y_batch,flag = get_test_batch(5,100)
y_pre = clf.predict(x_batch)
for i in range(0,len(y_pre)):
    print(y_pre[i],y_batch[i])
score =accuracy_score(y_batch, y_pre)
print(score)