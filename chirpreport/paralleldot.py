from csv import writer
import paralleldots
import pandas as pd
import re


def clean_tweets(tweet):
    tweet = re.sub(r'$\w', '', tweet)  # remove stock market tickers like $GE
    tweet = re.sub(r'@\w', '', tweet)  # remove stock market tickers like $GE
    tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
    tweet = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '',
                   tweet)  # remove hyperlinks
    tweet = re.sub(r'#', '', tweet)  # remove hashtags only removing the hash # sign from the word
    return tweet


def create_clean_tweets():
    tweets = pd.read_csv("tweets.csv").values
    cleaned = [clean_tweets(tweet[0]) for tweet in tweets]
    df = pd.DataFrame(cleaned, columns=['text'])
    df.to_csv("tweets_cleaned.csv", index=False)


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='', encoding="utf-8") as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def analyze(input_file, output_file):
    df = pd.read_csv(input_file, header=0)
    drop_index = []
    for i in range(50):
        drop_index.append(i)
        tweet = df.loc[i, 'text']
        print(tweet)
        res = paralleldots.emotion(tweet)
        print(res)
        emotion = res['emotion']
        insert_row = [tweet, emotion['Bored'], emotion['Angry'], emotion['Sad'], emotion['Fear'], emotion['Happy'],
                      emotion['Excited']]
        print(insert_row)
        append_list_as_row(output_file, insert_row)
    df = df.drop(drop_index)
    df.to_csv(input_file, index=False)


# create_clean_tweets()



for i in range(20):
    for key in keys:
        print(key)
        paralleldots.set_api_key(key)
        # paralleldots.sentiment( text )
        analyze('tweets_cleaned.csv', 'classified_tweets.csv')

response = "{'emotion': [{'Excited': 0.0947050187, 'Happy': 0.09413058, 'Bored': 0.0944737127, 'Sad': 0.1304103751, 'Fear': 0.1717756152, 'Angry': 0.4145046983}, {'Excited': 0.1811228963, 'Happy': 0.102223794, 'Bored': 0.1367001269, 'Sad': 0.1669378585, 'Fear': 0.3188618448, 'Angry': 0.0941534795}]}"
