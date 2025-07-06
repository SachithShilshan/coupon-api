from flask import Flask, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load models
lr_model = joblib.load('coupon_model_lr.pkl')
dt_model = joblib.load('coupon_model_dt.pkl')

# Load XGBoost Booster model saved as .json
xg_model = xgb.Booster()
xg_model.load_model('coupon_model_xgboost.json')

# Load features
features = joblib.load('model_features.pkl')           # for LR & DT
features_pca = joblib.load('model_features_pca.pkl')   # for XGBoost with PCA

# (Optional) Load threshold for XGBoost
try:
    best_thresh = joblib.load('xgboost_best_threshold.pkl')
except:
    best_thresh = 0.5  # fallback if not saved


@app.route('/')
def home():
    return "🎯 Coupon Redemption Prediction API is running!"


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict_lr():
    try:
        input_data = pd.DataFrame([request.json])

        # Fill missing features
        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[features]

        pred = lr_model.predict(input_data)[0]
        proba = lr_model.predict_proba(input_data)[0][1]

        return jsonify({
            'model': 'Logistic Regression',
            'prediction': int(pred),
            'probability': float(proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    try:
        input_data = pd.DataFrame([request.json])

        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[features]

        pred = dt_model.predict(input_data)[0]
        proba = dt_model.predict_proba(input_data)[0][1]

        return jsonify({
            'model': 'Decision Tree',
            'prediction': int(pred),
            'probability': float(proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_xg', methods=['POST'])
def predict_xg():
    try:
        input_data = pd.DataFrame([request.json])

        for col in features_pca:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[features_pca]

        dmatrix = xgb.DMatrix(input_data)
        proba = xg_model.predict(dmatrix)[0]
        prediction = int(proba >= best_thresh)

        return jsonify({
            'model': 'XGBoost (PCA)',
            'prediction': prediction,
            'probability': float(proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
