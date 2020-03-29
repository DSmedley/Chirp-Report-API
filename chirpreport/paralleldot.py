import paralleldots
import pandas as pd
import re


# Cleans the text in a array of ['id', 'text']
def clean_tweets(tweet):
    tweet[1] = re.sub(r'$\w', '', tweet[1])  # remove stock market tickers like $GE
    tweet[1] = re.sub(r'\@', '', tweet[1])  # remove @ symbol
    tweet[1] = re.sub(r'…', '', tweet[1])  # remove … from file
    tweet[1] = re.sub(r'^RT[\s]+', '', tweet[1])  # remove old style retweet text "RT"
    tweet[1] = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '',
                      tweet[1])  # remove hyperlinks
    tweet[1] = re.sub(r'#', '', tweet[1])  # remove hashtags only removing the hash # sign from the word
    return tweet


# Cleans the initial tweet csv and removes all the unused columns
def create_clean_tweets():
    tweets = pd.read_csv("tweets.20150430-223406.tweet.csv", usecols=['id', 'text']).values
    cleaned = [clean_tweets(tweet) for tweet in tweets]
    df = pd.DataFrame(cleaned, columns=['id', 'text'])
    df = remove_duplicate_tweets(df, 'id')
    df = remove_duplicate_tweets(df, 'text')
    df.to_csv("tweets_cleaned.csv", index=False)


def append_list_as_row(output_file, new_row, column_names):
    file = pd.read_csv(output_file, header=0)
    df2 = pd.DataFrame(new_row, columns=column_names)
    new_file = file.append(df2)
    new_file.to_csv(output_file, index=False)


def classify_emotion(input_file, output_file):
    df = pd.read_csv(input_file, header=0)
    drop_index = []
    for i in range(50):
        drop_index.append(i)
        tweet_id = df.loc[i, 'id']
        tweet = df.loc[i, 'text']
        res = paralleldots.emotion(tweet)
        print(res)
        emotion = res['emotion']
        insert_row = [(tweet_id, tweet, emotion['Bored'], emotion['Angry'], emotion['Sad'], emotion['Fear'], emotion['Happy'], emotion['Excited'])]
        print(insert_row)
        append_list_as_row(output_file, insert_row, ['id', 'text', 'Bored', 'Angry', 'Sad', 'Fear', 'Happy', 'Excited'])
    df = df.drop(drop_index)
    df.to_csv(input_file, index=False)


def classify_sentiment(input_file, output_file):
    df = pd.read_csv(input_file, header=0)
    drop_index = []
    for i in range(50):
        drop_index.append(i)
        tweet_id = df.loc[i, 'id']
        tweet = df.loc[i, 'text']
        res = paralleldots.sentiment(tweet)
        print(res)
        sentiment = res['sentiment']
        insert_row = [(tweet_id, tweet, df.loc[i, 'Bored'], df.loc[i, 'Angry'], df.loc[i, 'Sad'], df.loc[i, 'Fear'], df.loc[i, 'Happy'], df.loc[i, 'Excited'], sentiment['negative'], sentiment['neutral'], sentiment['positive'])]
        print(insert_row)
        append_list_as_row(output_file, insert_row, ['id', 'text', 'Bored', 'Angry', 'Sad', 'Fear', 'Happy', 'Excited', 'negative', 'neutral', 'positive'])
    df = df.drop(drop_index)
    df.to_csv(input_file, index=False)


def remove_duplicate_tweets(tweets, category):
    tweets.drop_duplicates(subset=category, keep='first', inplace=True)
    return tweets


def create_classified_tweets():

    for i in range(19):
        for key in keys:
            print(key)
            paralleldots.set_api_key(key)
            #classify_emotion('tweets_cleaned.csv', 'classified_tweets.csv')
            classify_sentiment('classified_tweets.csv', 'final_classified_tweets.csv')


#create_clean_tweets()
create_classified_tweets()
