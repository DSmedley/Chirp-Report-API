import pickle

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from chirpreport.preprocess_tweets import PreprocessTweets
from chirpreport.emotions import Emotions
from chirpreport.sentiments import Sentiments
import pandas as pd

model = load_model('models/200d-twitter-model_weights-improvement-74-0.718750.hdf5')
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
print(model.summary())

tweets = ["@MojoInTheMorn these Kevin whatever ads are probably the worst thing to happen to the podcasts. I have to listen to the same ad every 5-10 minutes and the guys voice is annoying"]

dataFrame = pd.read_csv('data/text_emotion_twitter.csv', encoding='utf-8')

tweet_text = dataFrame.values[:, 3]


MAX_SEQUENCE_LENGTH = 20

preprocess = PreprocessTweets(MAX_SEQUENCE_LENGTH)
sequence = preprocess.process(tweets)

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

instance = tokenizer.texts_to_sequences(sequence)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=20)
print(f"XDATA {instance}")

predicted_classes = model.predict(instance)
predicted_class = model.predict_classes(instance)

print(f"Predicted Classes: {predicted_classes}")
print(f"Predicted Class: {predicted_class}")
print(f"Predicted: {Sentiments(predicted_class).name}")