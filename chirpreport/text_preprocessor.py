from keras_preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords


def get_stop_words():
    stop_words = set(stopwords.words('english'))
    all_stop_words = set(stop_words)
    [all_stop_words.add(word.replace('\'', '')) for word in stop_words]
    return all_stop_words


class TextPreProcessor:
    def __init__(self, length):
        self.sequence_length = length

    def process(self, tweets):
        stop_words = get_stop_words()
        base_filters = '\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '
        word_sequences = []

        for tweet in tweets:
            tweet = str(tweet)
            tweet = tweet.replace('\'', '')
            new_list = [x for x in text_to_word_sequence(tweet, filters=base_filters, lower=True) if
                        not x.startswith("@")]
            filtered_sentence = [w for w in new_list if not w in stop_words]
            word_sequences.append(filtered_sentence)

        return word_sequences
