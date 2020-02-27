from chirpreport import app


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Chirp Report API</h1>
    <p>This is a flask api for sentiment analysis
    <br/>
    Powered by Chirp Report</p>'''