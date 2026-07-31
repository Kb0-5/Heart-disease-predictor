import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("Warning: model.pkl not found. Please run train_model.py first.")

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 
    'fbs', 'restecg', 'thalach', 'exang', 
    'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/', methods=['GET'])
def home():
    """Renders basic frontend interface or API status."""
    if os.path.exists('templates/index.html'):
        return render_template('index.html')
    return jsonify({
        "status": "API is online",
        "endpoint": "/predict",
        "method": "POST"
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input with patient clinical features.
    Returns JSON prediction.
    """
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    try:
        data = request.get_json(force=True)
        
        if "features" in data:
            input_features = np.array(data["features"]).reshape(1, -1)
        else:
            input_features = [float(data[feat]) for feat in FEATURE_NAMES]
            input_features = np.array(input_features).reshape(1, -1)

        raw_pred = model.predict(input_features)[0]
        
        prediction_text = "Heart Disease Detected" if raw_pred == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": prediction_text,
            "raw_output": int(raw_pred)
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Invalid input payload",
            "details": str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)