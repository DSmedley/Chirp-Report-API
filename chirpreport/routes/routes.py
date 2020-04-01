from chirpreport import app
from flask import request, jsonify

from chirpreport.text_processor import process_tweet


def analyzedJson():
    return jsonify(none=0,
                   anger=1,
                   anticipation=2,
                   disgust=3,
                   fear=4,
                   joy=5,
                   sadness=6,
                   surprise=7,
                   trust=8,
                   neutral=0,
                   positive=1,
                   negative=2,
                   top_anger="Top anger",
                   top_anticipation="Top anticipation",
                   top_disgust="Top digust",
                   top_fear="Top fear",
                   top_joy="Top joy",
                   top_sadness="Top sadness",
                   top_surprise="Top surprise",
                   top_trust="Top trust")


def iterate(object):
    for tweets in object:
        for value in tweets.values():
            print(value)
    return analyzedJson()


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Chirp Report API</h1>
    <p>This is a flask api for sentiment analysis
    <br/>
    Powered by Chirp Report</p>'''


@app.route('/v1/analyze/tweet', methods=['POST'])
def analyze_tweet():
    tweet = request.form.get('tweet')
    sentiments = process_tweet(tweet)[0]
    return jsonify(Negative=str(sentiments[0]),
                   Neutral=str(sentiments[1]),
                   Positive=str(sentiments[2]))

@app.route('/v1/analyze/tweets', methods=['POST'])
def analyze_tweets():
    request_json = request.get_json()
    return iterate(request_json)
