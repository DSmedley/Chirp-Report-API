import sys

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import text_to_word_sequence, Tokenizer
from nltk.corpus import stopwords
import numpy as np


def get_stop_words():
    stop_words = set(stopwords.words('english'))
    all_stop_words = set(stop_words)
    [all_stop_words.add(word.replace('\'', '')) for word in stop_words]
    return all_stop_words


class PreprocessTweets:
    def __init__(self, length):
        self.sequence_length = length

    def process(self, tweets):
        sequence_length = self.sequence_length
        stop_words = get_stop_words()

        # removing @ from default base filter, to remove that whole word, which might be considered as user or page name
        base_filters = '\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

        word_sequences = []

        for tweet in tweets:
            tweet = str(tweet)

            tweet = tweet.replace('\'', '')
            newlist = [x for x in text_to_word_sequence(tweet, filters=base_filters, lower=True) if
                       not x.startswith("@")]
            filtered_sentence = [w for w in newlist if not w in stop_words]
            word_sequences.append(filtered_sentence)

        return word_sequences

    def tokenize(self, word_sequences):
        sequence_length = self.sequence_length
        # Tokenizing words to word indices
        print(f"Word Sequences: {word_sequences}")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(word_sequences)
        word_indices = tokenizer.texts_to_sequences(word_sequences)
        print(f"Word Indices: {word_indices}")
        word_index = tokenizer.word_index
        print("Tokenized to Word indices as ")
        print(np.array(word_indices).shape)
        # padding word_indices
        x_data = pad_sequences(word_indices, maxlen=sequence_length)
        print("After padding data")
        print(x_data.shape)
        return x_data, word_index, tokenizer
