from chirpreport import app
from flask import request, jsonify
from chirpreport.text_processor import process_tweet, process_tweets


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Chirp Report API</h1>
    <p>This is a flask api for sentiment analysis
    <br/>
    Powered by Chirp Report</p>'''


@app.route('/v1/analyze/tweet', methods=['POST'])
def analyze_tweet():
    analysis = process_tweet(request.form.get('tweet'))
    return jsonify(analysis)


@app.route('/v1/analyze/tweets', methods=['POST'])
def analyze_tweets():
    counts = process_tweets(request.get_json())
    return jsonify(counts)
