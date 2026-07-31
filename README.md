# Heart Disease Prediction REST API Deployment

This repository contains an end-to-end Machine Learning model for predicting heart disease risk based on patient clinical attributes, deployed as a REST API on Render.

## Live Deployment URL
- **Render Service Web App:** `https://heart-disease-predictor-ia1t.onrender.com/`
- **Predict API Endpoint:** `https://heart-disease-predictor-ia1t.onrender.com/predict`

## Repository Structure
HeartDiseaseDeployment/
│
├── app.py               # Flask REST API server
├── train_model.py       # Preprocessing & Random Forest model training script
├── model.pkl            # Serialized trained model
├── requirements.txt     # Python runtime dependencies
├── README.md            # Project documentation & Render URL
├── heart.csv            # Dataset file
└── templates/
    └── index.html       # Basic landing page interface

## 🚀 API Usage Instructions

### **Endpoint:** `POST /predict`

#### **Request Header:**
`Content-Type: application/json`

#### **Sample JSON Request:**
```json
{
  "age": 57,
  "sex": 1,
  "cp": 2,
  "trestbps": 128,
  "chol": 229,
  "fbs": 0,
  "restecg": 0,
  "thalach": 156,
  "exang": 1,
  "oldpeak": 1.0,
  "slope": 1,
  "ca": 1,
  "thal": 3
}
```

#### **Sample Response:**
```json
{
  "prediction": "Heart Disease Detected",
  "raw_output": 1
}
```

---

## 📝 Conclusion

### Model Performance
The Random Forest Classifier trained on the Heart Disease dataset achieved strong predictive accuracy across evaluation metrics, effectively modeling clinical features like chest pain type (`cp`), maximum heart rate achieved (`thalach`), and ST depression (`oldpeak`).

### Challenges Faced
During deployment, key operational challenges included matching scikit-learn dependency versions across local and production environments, managing Gunicorn port configuration dynamically on Render's ephemeral web instances, and structuring the API payload to validate numerical fields robustly.

### Importance of MLOps
MLOps (Machine Learning Operations) ensures seamless transition from static offline notebooks to robust, continuously deployed production systems. By establishing automated continuous integration, dependency tracking, model artifact serialization, and containerized serving, MLOps guarantees model reproducibility, monitoring, and zero-downtime integration into enterprise healthcare platforms.