import datetime
import os
import time

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from chirpreport.preprocess_tweets import PreprocessTweets

MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 200
dataFrame = pd.read_csv('data/single_classification_tweets.csv', encoding='utf-8')

tweet_text = dataFrame.values[:, 3]
tweet_emotion = dataFrame.values[:, 2]

print(tweet_text.shape)
print(tweet_emotion.shape)

# ----------------------------Preprocessing Text Data---------------------------------------

preprocess = PreprocessTweets(tweet_text, MAX_SEQUENCE_LENGTH)
x_data, word_index = preprocess.process()

# ----------------------------Building Embedding Layer---------------------------------------

# using pretrained glove vector
print("Loading Glove Vectors ...")
embeddings_index = {}
f = open(os.path.join('', 'gloves/glove.twitter.27B.200d.txt'), 'r', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded GloVe Vectors Successfully')

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Embedding Matrix Generated : ", embedding_matrix.shape)

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)

# -----------------------------------------------------------------------------------------------------

# ---------------------------One Hot encoding Labels ------------------------------------------------------------


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(tweet_emotion)
le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print("Label Encoding Classes as ")
print(le_name_mapping)

y_data = np_utils.to_categorical(integer_encoded)
print("One Hot Encoded class shape ")
print(y_data.shape)
# ---------------------------Building Model ------------------------------------------------------------
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(30, 1, activation="relu"))
model.add(MaxPooling1D(4))
model.add(LSTM(100, return_sequences=True))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(y_data.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
print(model.summary())

print("Finished Preprocessing data ...")
print("x_data shape : ", x_data.shape)
print("y_data shape : ", y_data.shape)

# spliting data into training, testing set
print("spliting data into training, testing set")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

print(f"x_test: {x_data[0]}")
print(f"x_test: {x_test[0]}")
print(f"y_test: {y_test[0]}")

batch_size = 64
num_epochs = 100
x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
# define the checkpoint
filepath = "200d-twitter-model_weights-improvement-{epoch:02d}-{val_accuracy:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs,
                    callbacks=callbacks_list)

scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

pyplot.plot(history.history['accuracy'], label='Training Accuracy')
pyplot.plot(history.history['val_accuracy'], label='Validation Accuracy')

pyplot.legend()
pyplot.show()