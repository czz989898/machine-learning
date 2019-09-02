from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
sentences = []
authors = []
with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_data.txt',encoding='UTF-8') as doc_source:
    for i in doc_source:
        info= i.split('\t')
        if len(info)==2:
            sentences.append(info[1])
            authors.append(info[0])
tfidf_vec = TfidfVectorizer(stop_words ="english",max_features = 15000)
tfidf_matrix = tfidf_vec.fit_transform(sentences)
#print(tfidf_vec.get_feature_names())
#print(tfidf_vec.vocabulary_)
global start
start = 0
def get_batch(num):
    global start
    a = start
    start+=num
    x_batch = tfidf_matrix[a:a+num].toarray()
    if len(x_batch)==0:
        return 0,0,False
    else:
        return x_batch,authors[a:a+num],True
def get_test_batch(start,end):
    x_batch = tfidf_matrix[start:end].toarray()
    if len(x_batch)==0:
        return 0,0,False
    else:
        return x_batch,authors[start:end],True