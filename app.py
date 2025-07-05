from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
lr_model = joblib.load('coupon_model_lr.pkl')
dt_model = joblib.load('coupon_model_dt.pkl')
xg_model = joblib.load('coupon_model_xgboost.pkl')

# Load feature lists
features = joblib.load('model_features.pkl')           # For LR and DT
features_pca = joblib.load('model_features_pca.pkl')   # For XGBoost

@app.route('/')
def home():
    return "🎯 Coupon Redemption Prediction API is running!"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    try:
        df = pd.DataFrame([request.json])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

        pred = lr_model.predict(df)[0]
        proba = lr_model.predict_proba(df)[0][1]

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
        df = pd.DataFrame([request.json])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

        pred = dt_model.predict(df)[0]
        proba = dt_model.predict_proba(df)[0][1]

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
        df = pd.DataFrame([request.json])
        for col in features_pca:
            if col not in df.columns:
                df[col] = 0
        df = df[features_pca]

        pred = xg_model.predict(df)[0]
        proba = xg_model.predict_proba(df)[0][1]

        return jsonify({
            'model': 'XGBoost (PCA)',
            'prediction': int(pred),
            'probability': float(proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
