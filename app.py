from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return "🎯 Coupon Redemption Prediction API is running!"


if __name__ == '__main__':
    app.run(debug=True)
