import datetime
import os
import pickle

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, SpatialDropout1D
from keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from chirpreport.text_preprocessor import process
import plotly.graph_objects as go

MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 200
DATE = datetime.datetime.now().strftime("%m-%d-%H.%M")
TOKENIZER_NAME = f"{EMBEDDING_DIM}d-{MAX_SEQUENCE_LENGTH}l-tokenizer-{DATE}"
SENTIMENT_MODEL_NAME = f"{EMBEDDING_DIM}d-{MAX_SEQUENCE_LENGTH}l-sentiment-{DATE}"
EMOTIONS_MODEL_NAME = f"{EMBEDDING_DIM}d-{MAX_SEQUENCE_LENGTH}l-emotions-{DATE}"


def process_tweets(tweet_text):
    word_sequences = []
    for tweet in tweet_text:
        sequence = process(tweet)
        word_sequences.append(sequence)
    return word_sequences


def tokenize(word_sequences):
    # Tokenizing words to word indices
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_sequences)
    word_indices = tokenizer.texts_to_sequences(word_sequences)
    word_index = tokenizer.word_index
    print("Tokenized to Word indices as ")
    print(np.array(word_indices).shape)
    # padding word_indices
    x_data = pad_sequences(word_indices, maxlen=MAX_SEQUENCE_LENGTH)
    print("After padding data")
    print(x_data.shape)
    with open(f'models/{TOKENIZER_NAME}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return x_data, word_index


def build_embedding_layer(word_index):
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
    return Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)


def encode_labels(classifications):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classifications)
    le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    print("Label Encoding Classes as ")
    print(le_name_mapping)
    y_data = np_utils.to_categorical(integer_encoded)
    print("One Hot Encoded class shape ")
    print(y_data.shape)
    return y_data


def build_model(embedding_layer, x_data, y_data, model_name, title, epoch):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(30, 1, activation="relu"))
    model.add(SpatialDropout1D(0.4))
    model.add(MaxPooling1D(4))
    model.add(LSTM(100, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(y_data.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    print(model.summary())
    train_model(model, x_data, y_data, model_name, title, epoch)


def train_model(model, x_data, y_data, model_name, title, epoch):
    print("Finished Preprocessing data ...")
    print("x_data shape : ", x_data.shape)
    print("y_data shape : ", y_data.shape)
    # spliting data into training, testing set
    print("spliting data into training, testing set")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    batch = 64
    x_valid, y_valid = x_train[:batch], y_train[:batch]
    x_train2, y_train2 = x_train[batch:], y_train[batch:]
    filepath = "models/" + model_name + "-{epoch:02d}-{val_accuracy:.6f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch,
                        epochs=epoch, callbacks=callbacks_list)
    scores = model.evaluate(x_test, y_test, verbose=0)
    graph_results(history, scores, title)


def graph_results(history, scores, title):
    x = list(range(1, len(history.history['accuracy']) + 1))
    accuracy = go.Figure()
    accuracy.add_trace(go.Scatter(x=x, y=history.history['accuracy'], mode='lines', name='Training Accuracy'))
    accuracy.add_trace(go.Scatter(x=x, y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
    accuracy.update_layout(title=f'{title} Model Accuracy With Final Test Accuracy of {(scores[1] * 100):.{2}f}%',
                           xaxis_title='Iteration', yaxis_title='Accuracy')
    accuracy.show()

    loss = go.Figure()
    loss.add_trace(go.Scatter(x=x, y=history.history['loss'], mode='lines', name='Training Loss'))
    loss.add_trace(go.Scatter(x=x, y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    loss.update_layout(title=f'{title} Model Loss With Final Test Accuracy of {(scores[1] * 100):.{2}f}%',
                       xaxis_title='Iteration', yaxis_title='Loss')
    loss.show()


def create_prediction_model():
    # Get training data and split text from classifications
    dataFrame = pd.read_csv('data/tweets_classified.csv', encoding='utf-8')
    tweet_emotions = dataFrame.values[:, 1]
    tweet_sentiments = dataFrame.values[:, 2]
    tweet_text = dataFrame.values[:, 3]

    # Preprocess tweets and creating the vocabulary for both models
    word_sequences = process_tweets(tweet_text)
    tweets, word_index = tokenize(word_sequences)
    layer = build_embedding_layer(word_index)

    # Build Sentiment Model
    sentiment_classifications = encode_labels(tweet_sentiments)
    build_model(layer, tweets, sentiment_classifications, SENTIMENT_MODEL_NAME, 'Sentiment', 200)

    # Build Emotions Model
    emotion_classifications = encode_labels(tweet_emotions)
    build_model(layer, tweets, emotion_classifications, EMOTIONS_MODEL_NAME, 'Emotion', 250)


create_prediction_model()
