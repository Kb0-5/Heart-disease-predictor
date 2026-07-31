import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 
    'fbs', 'restecg', 'thalach', 'exang', 
    'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/', methods=['GET'])
def home():
    """Render home form webpage."""
    return render_template('index.html')

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    """Handles HTML Form submission and renders result.html."""
    try:
        input_data = [float(request.form[col]) for col in FEATURE_NAMES]
        features_array = np.array(input_data).reshape(1, -1)

        raw_pred = int(model.predict(features_array)[0])
        prediction_text = "Heart Disease Detected" if raw_pred == 1 else "No Heart Disease Detected"

        return render_template('result.html', prediction=prediction_text)
    except Exception as e:
        return f"Error processing form data: {str(e)}", 400

@app.route('/predict', methods=['POST'])
def predict():
    """JSON REST API Endpoint (Task 3 requirement)."""
    try:
        data = request.get_json(force=True)
        if "features" in data:
            input_data = np.array(data["features"]).reshape(1, -1)
        else:
            input_data = np.array([[float(data[col]) for col in FEATURE_NAMES]])

        raw_pred = int(model.predict(input_data)[0])
        result = "Heart Disease Detected" if raw_pred == 1 else "No Heart Disease Detected"

        return jsonify({"prediction": result}), 200
    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)