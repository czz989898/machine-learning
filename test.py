import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import adam

################################## read dataset ###################################
train_df = pd.read_csv("tidy_train_tweets.csv")
train_df = train_df[:10000]
train_df.tweet = train_df.tweet.astype(str)
train_y = train_df.user

################################## encode dataset ###################################
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1,1)

ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()

################################## Tokenizer #########################################
max_words = 50000
max_len = 250
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
tok.fit_on_texts(train_df.tweet)

################################## Sequential ########################################
train_seq = tok.texts_to_sequences(train_df.tweet)

################################## Padding ###########################################
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)

################################## LSTM ##############################################

model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(236, activation='softmax'))
opt = adam(lr=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
batch_size = 128
history = model.fit(train_seq_mat, train_y, epochs=50, batch_size=batch_size,validation_split=0.1,)

