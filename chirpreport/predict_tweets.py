from keras.models import load_model
from chirpreport.preprocess_tweets import PreprocessTweets
from chirpreport.emotions import Emotions
from chirpreport.sentiments import Sentiments
import pandas as pd

model = load_model('models/sentiment-200d-twitter-model_weights-improvement-50-0.718750.hdf5')
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
print(model.summary())

tweets = ["@MojoInTheMorn these Kevin whatever ads are probably the worst thing to happen to the podcasts. I have to listen to the same ad every 5-10 minutes and the guys voice is annoying"]

dataFrame = pd.read_csv('text_emotion_twitter.csv', encoding='utf-8')

tweet_text = dataFrame.values[:, 3]


MAX_SEQUENCE_LENGTH = 20

preprocess = PreprocessTweets(tweets, MAX_SEQUENCE_LENGTH)
x_data, word_index = preprocess.process()
# print(f"TWEETS: {tweet_text[0]}")
# print(f"XDATA {x_data[0]}")
# print(f"WORD INDEX{word_index}")

predicted_classes = model.predict(x_data)
predicted_class = model.predict_classes(x_data)

print(f"Predicted Classes: {predicted_classes}")
print(f"Predicted Class: {predicted_class}")
print(f"Predicted: {Sentiments(predicted_class).name}")