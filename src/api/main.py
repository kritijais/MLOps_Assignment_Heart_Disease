import logging
from fastapi import FastAPI, HTTPException
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

# Import the modular model loader
from src.inference.model_loader import load_model

# Import the input schema
from src.api.schema import HeartDiseaseFeatures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- LOAD MODEL ----------------
try:
    model = load_model()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk using a pre-trained ML pipeline",
    version="1.0"
)

# ---------------- Prometheus Metrics ----------------
Instrumentator().instrument(app).expose(app)

# ---------------- HEALTH CHECK ----------------
@app.get("/")
def root():
    return {"status": "Heart Disease Prediction API is running"}

# ---------------- ENDPOINT ----------------
# Sample request - {
#     "age": 63.0,
#     "sex": 1,
#     "cp": 3,
#     "trestbps": 145.0,
#     "chol": 233.0,
#     "fbs": 1,
#     "restecg": 0,
#     "thalach": 150.0,
#     "exang": 0,
#     "oldpeak": 2.3,
#     "slope": 0,
#     "ca": 0,
#     "thal": 1
# }
@app.post("/predict")
def predict(patient: HeartDiseaseFeatures):
    try:
        # Convert incoming JSON dict to DataFrame
        df = pd.DataFrame([patient.dict()])

        # Ensure categorical columns are of type 'object'
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            df[col] = df[col].astype('object')

        # Predict
        pred_prob = model.predict_proba(df)[:, 1]
        pred_class = model.predict(df)

        return {
            "prediction": int(pred_class[0]),
            "risk_probability": float(pred_prob[0])
        }

    except Exception as e:
        return {"detail": f"Inference error: {str(e)}"}