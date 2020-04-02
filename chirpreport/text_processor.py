import operator

import numpy as np
import pickle
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from chirpreport.emotions import Emotions
from chirpreport.sentiments import Sentiments
from chirpreport.text_preprocessor import process

S_MODEL = '200d-twitter-model_weights-improvement-74-0.718750.hdf5'
E_MODEL = '200d-20l-emotions-92-0.578125.hdf5'
S_T_MODEL = 'tokenizer.pickle'
E_T_MODEL = 'tokenizer.pickle'
MAX_SEQUENCE_LENGTH = 20


def load_tokenizer(model):
    with open(f'models/{model}', 'rb') as handle:
        tokenizer_pickle = pickle.load(handle)
    return tokenizer_pickle


sentiment_model = load_model(f'models/{S_MODEL}')
sentiment_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
emotion_model = load_model(f'models/{E_MODEL}')
emotion_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
sentiment_tokenizer = load_tokenizer(S_T_MODEL)
emotion_tokenizer = load_tokenizer(E_T_MODEL)


def process_tweet(tweet):
    predictions = {}
    sentiments, emotions = get_predictions(tweet)
    predictions.update(sentiments)
    predictions.update(emotions)
    for keys in predictions:
        predictions[keys] = str(predictions[keys])
    return predictions


def get_predictions(tweet):
    sentiments = {}
    emotions = {}
    tweet_sequence = process(tweet)
    sentiments_values = analyze_text(tweet_sequence, sentiment_tokenizer, sentiment_model)
    emotions_values = analyze_text(tweet_sequence, emotion_tokenizer, emotion_model)
    for i in sentiments_values:
        sentiments[Sentiments(np.argwhere(sentiments_values == i)).name] = i
    for i in emotions_values:
        emotions[Emotions(np.argwhere(emotions_values == i)).name] = i
    return sentiments, emotions


def process_tweets(tweet_json):
    top_emotions = {
        "Angry": 0,
        "Bored": 0,
        "Excited": 0,
        "Fear": 0,
        "Happy": 0,
        "Sad": 0,
    }
    counts = {
        "Negative": 0,
        "Neutral": 0,
        "Positive": 0,
        "Angry": 0,
        "Bored": 0,
        "Excited": 0,
        "Fear": 0,
        "Happy": 0,
        "Sad": 0,
        "Top_Angry": "",
        "Top_Bored": "",
        "Top_Excited": "",
        "Top_Fear": "",
        "Top_Happy": "",
        "Top_Sad": "",
    }
    for tweet in tweet_json:
        sentiment, emotion = get_predictions(tweet['text'])
        classified_sentiment = max(sentiment.items(), key=operator.itemgetter(1))[0]
        counts[classified_sentiment] = counts[classified_sentiment] + 1
        classified_emotion = max(emotion.items(), key=operator.itemgetter(1))[0]
        counts[classified_emotion] = counts[classified_emotion] + 1
        if emotion[classified_emotion] > top_emotions[classified_emotion]:
            top_emotions[classified_emotion] = emotion[classified_emotion]
            counts[f"Top_{classified_emotion}"] = tweet
    return counts


def analyze_text(sequence, tokenizer, model):
    instance = tokenizer.texts_to_sequences(sequence)
    flat_list = []
    [[flat_list.append(item) for item in sublist] for sublist in instance]
    instance = pad_sequences([flat_list], maxlen=MAX_SEQUENCE_LENGTH)
    return model.predict(instance)[0]
