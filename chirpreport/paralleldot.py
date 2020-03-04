from csv import writer
import paralleldots
import pandas as pd
import re

# Cleans the text in a array of ['id', 'text']
def clean_tweets(tweet):
    tweet[1] = re.sub(r'$\w', '', tweet[1])  # remove stock market tickers like $GE
    tweet[1] = re.sub(r'@\w', '', tweet[1])  # remove stock market tickers like $GE
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


create_clean_tweets()


def add_id_to_already_classified_tweets():
    # This function is not finished
    tweetsWithID = pd.read_csv('tweets_cleaned.csv').values
    classifiedTweetsWithoutID = pd.read_csv('classified_tweets.csv', encoding='iso-8859-1').values

    return 1


add_id_to_already_classified_tweets();

# for i in range(20):
#     for key in keys:
#         print(key)
#         paralleldots.set_api_key(key)
#         # paralleldots.sentiment( text )
#         analyze('tweets_cleaned.csv', 'classified_tweets.csv')

# response = "{'emotion': [{'Excited': 0.0947050187, 'Happy': 0.09413058, 'Bored': 0.0944737127, 'Sad': 0.1304103751, 'Fear': 0.1717756152, 'Angry': 0.4145046983}, {'Excited': 0.1811228963, 'Happy': 0.102223794, 'Bored': 0.1367001269, 'Sad': 0.1669378585, 'Fear': 0.3188618448, 'Angry': 0.0941534795}]}"
