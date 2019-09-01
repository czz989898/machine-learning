from nltk.corpus import stopwords
import nltk
#nltk.download()
def normalize(type = 'test'):
    if type!='test':
        #print('train')
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
                            sentence = sentence[:j+1]+' '+sentence[j+1:]
                        j+=1
                    sentence.replace(',','')
                    sentence.replace('.','')
                    sentence_l = sentence.split(' ')
                    for j in sentence_l:
                        j = j.lower()
                        if j not in stopwords.words('english'):
                            words += j + ' '
                    words+='\n'
                    feature_word_data.write(words)
    else:
        #print('test')
        with open(r'E:\home work\semester3\machine learning\assignment\data\test_tweets_unlabeled.txt',
                  encoding='UTF-8') as train_data:
            with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_unlabeled.txt', 'w',
                      encoding='UTF-8') as feature_word_data:
                count = 0
                for i in train_data:
                    count += 1
                    #if count % 100 == 0:
                        #print(count)
                    sentence = i.strip()
                    words = ''
                    j = 0
                    while j < len(sentence) - 1:
                        if (sentence[j].isalpha() or sentence[j].isdigit()) and sentence[j + 1] in ['!', '@', '#', '$',
                                                                                                    ',', '.', '?', ':']:
                            sentence = sentence[:j + 1] + ' ' + sentence[j + 1:]
                        j += 1
                    sentence.replace(',', '')
                    sentence.replace('.', '')
                    sentence_l = sentence.split(' ')
                    for j in sentence_l:
                        j = j.lower()
                        if j not in stopwords.words('english'):
                            words += j + ' '
                    words += '\n'
                    feature_word_data.write(words)
def get_feature():
    words = {}
    count = 0
    with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_data.txt', 'r',
              encoding='UTF-8') as feature_word_data:
        for i in feature_word_data:
            count+=1
            if count%5000==0:
                print(count,len(words))
            try:
                sentence = i.strip().split('\t')[1]
                sentence_l = sentence.split(' ')
                for j in sentence_l:
                    if j not in words and '\\' not in j and '/' not in j:
                        words[j]=1
            except:
                pass
    content = ' '.join(words)
    with open(r'E:\home work\semester3\machine learning\assignment\data\features.txt', 'w',
              encoding='UTF-8') as feature:
        feature.write(content)
def get_batch(num):
    flag = False
    start = 0
    features = []
    fd = {}
    x_batch = []
    y_data = []
    with open(r'E:\home work\semester3\machine learning\assignment\data\features.txt', 'r',
              encoding='UTF-8') as feature:
        for i in feature:
            features = i.split()
    with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt',encoding='UTF-8') as config:
        for i in config:
            start = int(i)
    with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_data.txt',encoding='UTF-8') as doc_source:
        count = -1
        for i in doc_source:
            count += 1
            if count>=start+num:
                flag = True
                break
            if count>=start:
                ###############################
                fd = {}
                for j in features[0:100000]:
                    if j not in fd:
                        fd[j] = 0
                ################################
                info = i.split('\t')
                sentence = ''
                if len(info)==2:
                    sentence = info[1]
                sentence = sentence.split(' ')
                for j in sentence:
                    if j in fd:
                        fd[j]+=1
                x_batch.append(list(fd.values()))
                y_data.append(info[0])
    with open(r'E:\home work\semester3\machine learning\assignment\data\config.txt','w', encoding='UTF-8') as config:
        config.write(str(start+num))
    return x_batch,y_data,flag
def get_batch_test(num):
    flag = False
    start = 0
    features = []
    fd = {}
    x_batch = []
    with open(r'E:\home work\semester3\machine learning\assignment\data\features.txt', 'r',
              encoding='UTF-8') as feature:
        for i in feature:
            features = i.split()
    with open(r'E:\home work\semester3\machine learning\assignment\data\test_config.txt', encoding='UTF-8') as config:
        for i in config:
            start = int(i)
    with open(r'E:\home work\semester3\machine learning\assignment\data\feature_word_unlabeled.txt',
              encoding='UTF-8') as doc_source:
        count = -1
        for i in doc_source:
            count += 1
            if count >= start + num:
                flag = True
                break
            if count >= start:
                ###############################
                fd = {}
                for j in features[0:100000]:
                    if j not in fd:
                        fd[j] = 0
                ################################
                if count % 200 == 0:
                    print(count)
                info = i.split('\t')
                sentence = ''
                if len(info) == 1:
                    sentence = info[0]
                sentence = sentence.split(' ')
                for j in sentence:
                    if j in fd:
                        fd[j] += 1
                x_batch.append(list(fd.values()))

    with open(r'E:\home work\semester3\machine learning\assignment\data\test_config.txt', 'w', encoding='UTF-8') as config:
        config.write(str(start + num))
    return x_batch, flag
def get_all_author():
    with open(r'E:\home work\semester3\machine learning\assignment\data\train_tweets.txt',
              encoding='UTF-8') as train_data:
        authors = set([])
        for i in train_data:
            authors.add(i.split('\t')[0])
        content = ' '.join(authors)
        with open(r'E:\home work\semester3\machine learning\assignment\data\labels.txt','w',encoding='UTF-8') as labels:
            labels.write(content)
#get_batch_train()
#get_feature()
#get_batch(100)
#get_all_author()
