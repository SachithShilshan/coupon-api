from flask import Flask, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load scikit-learn models
lr_model = joblib.load('coupon_model_lr.pkl')
dt_model = joblib.load('coupon_model_dt.pkl')

# Load XGBoost model (trained using xgb.train())
xg_model = xgb.Booster()
#xg_model.load_model('coupon_model_xgboost.json')

# Load best threshold for XGBoost
#best_thresh = joblib.load('xgboost_best_threshold.pkl')

# Load feature lists
features = joblib.load('model_features.pkl')           # For Logistic & DT
#features_pca = joblib.load('model_features_pca.pkl')   # For XGBoost with PCA

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

        # Fill missing features with 0
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



if __name__ == '__main__':
    app.run(debug=True)
