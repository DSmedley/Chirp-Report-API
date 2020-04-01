import pickle
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from chirpreport.text_preprocessor import TextPreProcessor

S_MODEL = '200d-twitter-model_weights-improvement-74-0.718750.hdf5'
E_MODEL = '200d-twitter-model_weights-improvement-74-0.718750.hdf5'
T_MODEL = 'tokenizer.pickle'
MAX_SEQUENCE_LENGTH = 20


def load_tokenizer(model):
    with open(f'models/{model}', 'rb') as handle:
        tokenizer_pickle = pickle.load(handle)
    return tokenizer_pickle


sentiment_model = load_model(f'models/{S_MODEL}')
sentiment_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# emotion_model = emotion_model
tokenizer = load_tokenizer(T_MODEL)
preprocessor = TextPreProcessor(MAX_SEQUENCE_LENGTH)


def process_tweet(tweet):
    tweet_sequence = preprocessor.process([tweet])
    sentiments = analyze_sentiment(tweet_sequence[0])
    emotions = analyze_emotion(tweet_sequence[0])
    return sentiments


def process_tweets(tweets):
    tweet_sequences = preprocessor.process(tweets)
    for sequence in tweet_sequences:
        analyze_sentiment(sequence)
        analyze_emotion(sequence)


def analyze_sentiment(sequence):
    instance = tokenizer.texts_to_sequences(sequence)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=20)
    return sentiment_model.predict(instance)


def analyze_emotion(tweet):
    return ""
