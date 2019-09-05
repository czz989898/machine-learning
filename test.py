import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import adam

################################## read dataset ###################################

# training set
train_df = pd.read_csv("tidy_train_tweets.csv")
train_df.tweet = train_df.tweet.astype(str)
train_y = train_df.user.astype(str)

# test set
test_df = pd.read_csv("tidy_test_tweets_unlabeled.csv")
test_df.tweet = test_df.tweet.astype(str)

################################## encode dataset ###################################
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
labels = le.classes_
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()

################################## Tokenizer #########################################
max_words = 50000
max_len = 250

# tokenize train set
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为50000
tok.fit_on_texts(train_df.tweet)

# tokenize test set
tok.fit_on_texts(test_df.tweet)

################################## Sequential ########################################
train_seq = tok.texts_to_sequences(train_df.tweet)
test_seq = tok.texts_to_sequences(test_df.tweet)

################################## Padding ###########################################
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)


################################## LSTM ##############################################

model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(train_df['user'].unique().tolist()), activation='softmax'))
opt = adam(lr=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
batch_size = 256
history = model.fit(train_seq_mat, train_y, epochs=60, batch_size=batch_size, validation_split=0.1)

################################# test ##############################################
print(test_seq_mat)
predictions = [0]
for i in range(len(test_seq_mat)):

    input_pred = np.array(test_seq_mat[i]).reshape(-1, max_len)
    prediction = model.predict(input_pred)
    rnn_label = labels[np.argmax(prediction)]
    predictions.append(rnn_label)


################################# export predictions ################################
df_nb = pd.DataFrame(predictions)
df_nb.to_csv("predictionsWithRNN.csv")
