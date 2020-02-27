from flask import Flask, request, jsonify
app = Flask(__name__)
from chirpreport.routes.Routes import *


app.config["DEBUG"] = True

if __name__ == "__main__":
    app.run()
