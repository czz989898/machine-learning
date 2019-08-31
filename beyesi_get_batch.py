from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
#nltk.download()
def get_feature():
    with open(r'E:\home work\semester3\machine learning\assignment\data\train_tweets.txt',encoding='UTF-8') as train_data:
        with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_data.txt','w',encoding='UTF-8') as feature_word_data:
            count = 0
            for i in train_data:

                count+=1
                if count%100==0:
                    print(count)
                info = i.split('\t')
                words = info[0] + '\t'
                sentence = info[1].strip()
                j=0
                while j <len(sentence)-1:
                    if (sentence[j].isalpha() or sentence[j].isdigit()) and sentence[j+1] in ['!','@','#','$',',','.','?',':']:
                        sentence = sentence[:j+1]+' '+sentence[j+1]
                    j+=1
                sentence.replace(',','')
                sentence.replace('.', '')
                sentence_l = sentence.split(' ')
                for j in sentence_l:
                    j = j.lower()
                    if j not in stopwords.words('english'):
                        words += j + ' '
                words+='\n'
                feature_word_data.write(words)


def get_batch_train(num):
    features = []
    vectorizer = CountVectorizer()
    doc = []
    flag = False
    start = 0
    with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt',encoding='UTF-8') as config:
        for i in config:
            start = int(i)
    with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_data.txt',encoding='UTF-8') as doc_source:
        count = -1
        y_data = []

        for i in doc_source:
            count += 1
            if count>=start:
                if count % 200 == 0:
                    print(count)
                info = i.split('\t')
                doc.append(info[1])
                y_data.append(info[0])
            if count>start+num:
                flag = True
                break
    a = vectorizer.fit_transform(doc).toarray()
    with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
        config.write(str(start+num))
    return a,y_data,flag




#get_batch_train()
