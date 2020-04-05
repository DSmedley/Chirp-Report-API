import re

from keras_preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords


def get_stop_words():
    stop_words = set(stopwords.words('english'))
    all_stop_words = set(stop_words)
    [all_stop_words.add(word.replace('\'', '')) for word in stop_words]
    return all_stop_words


def process(tweet):
    print(f"Tweet: {tweet}")
    stop_words = get_stop_words()
    base_filters = '\n\t!"#$%&()*+,-–./:;<=>?[\]^_`{|}~ 0123456789'

    tweet = str(tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
    tweet = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '',tweet)  # remove hyperlinks
    tweet = tweet.replace('\'', '')
    new_list = [x for x in text_to_word_sequence(tweet, filters=base_filters, lower=True) if
                not x.startswith("@")]
    print(f"Processed Tweet: {new_list}")
    final = [w for w in new_list if not w in stop_words]
    print(f"Removed stop words: {final}")
    return final
