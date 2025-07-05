from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
lr_model = joblib.load('coupon_model_lr.pkl')
dt_model = joblib.load('coupon_model_dt.pkl')
xg_model = joblib.load('coupon_model_xgboost.pkl')

# Load feature lists
features = joblib.load('model_features.pkl')           # For LR & DT
features_pca = joblib.load('model_features_pca.pkl')   # For XGBoost (PCA)

@app.route('/')
def home():
    return "Coupon Redemption Prediction API"

@app.route('/predict', methods=['POST'])
def predict_lr():
    try:
        input_data = pd.DataFrame([request.json])[features]
        pred = lr_model.predict(input_data)[0]
        proba = lr_model.predict_proba(input_data)[0][1]
        return jsonify({'model': 'Logistic Regression', 'prediction': int(pred), 'probability': float(proba)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    try:
        input_data = pd.DataFrame([request.json])[features]
        pred = dt_model.predict(input_data)[0]
        proba = dt_model.predict_proba(input_data)[0][1]
        return jsonify({'model': 'Decision Tree', 'prediction': int(pred), 'probability': float(proba)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_xg', methods=['POST'])
def predict_xg():
    input_data = pd.DataFrame([request.json])
    for col in features_pca:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[features_pca]
    pred = xg_model.predict(input_data)[0]
    proba = xg_model.predict_proba(input_data)[0][1]
    return jsonify({'model': 'XGBoost', 'prediction': int(pred), 'probability': float(proba)})

if __name__ == '__main__':
    app.run(debug=True)
