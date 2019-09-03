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
print(tfidf_matrix.shape)
#print(tfidf_vec.get_feature_names())
#print(tfidf_vec.vocabulary_)
global start
start = 0
test_sentences=[]
with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_unlabeled.txt',encoding='UTF-8') as feature_word_unlabeled:
    for i in feature_word_unlabeled:
        test_sentences.append(i)
    test_matrix = tfidf_vec.transform(test_sentences)
global test_start
test_start = 0
def get_batch(num):
    global start
    a = start
    start+=num
    x_batch = tfidf_matrix[a:a+num].toarray()
    if len(x_batch)==0:
        return 0,0,False
    else:
        return x_batch,authors[a:a+num],True
def get_test_batch(num):
    global test_start
    a = test_start
    test_start += num
    x_batch = test_matrix[a:a + num].toarray()
    if len(x_batch) == 0:
        return 0, False
    else:
        return x_batch, True
def get_batch_accuracy(start,end):
    return tfidf_matrix[start:end].toarray(),authors[start:end]
