from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
import re
import string
import spacy


nlp = spacy.load("en_vectors_web_lg")
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stopwords_english = stopwords.words('english')

oneTweet = "@MojoInTheMorn these Kevin whatever ads are probably the worst thing to happen to the podcasts. I have to listen to the same ad every 5-10 minutes and the guys voice is annoying"


def remove_duplicates(x):
    return list(dict.fromkeys(x))


def clean_tweets(tweet):
    tweet = re.sub(r'$\w', '', tweet)  # remove stock market tickers like $GE
    tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
    tweet = re.sub(r'https?://.[\r\n]*', '', tweet)  # remove hyperlinks
    tweet = re.sub(r'#', '', tweet)  # remove hashtags only removing the hash # sign from the word
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            tweets_clean.append(word)

    return tweet_tokens, tweets_clean


# TODO: Implement this later
def get_similarity(original, synonym):
    w1 = wordnet.synset(original)
    w2 = wordnet.synset(synonym)
    return w1.wup_similarity(w2)


def convert(lst):
    print(lst)
    ret = ' '.join(lst)
    print(ret)
    return ret.strip()


def get_synonyms(words_list):
    for word in words_list:
        print(word)
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms) > 0:
            first = nlp(word.strip())
            tokens = nlp(convert(remove_duplicates(synonyms)))
            if first and first.vector_norm:
                for token in tokens:
                    if token and token.vector_norm:
                        print(first[0].text, token.text, first[0].similarity(token))
            print(f"SYNONYMS: {remove_duplicates(synonyms)}")


def analyze_tweet(tweet):
    for sentence in sent_tokenize(tweet):
        tokens, clean_tokens = clean_tweets(sentence)
        get_synonyms(clean_tokens)


analyze_tweet(oneTweet)
