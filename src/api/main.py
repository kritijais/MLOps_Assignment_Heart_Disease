import logging
from fastapi import FastAPI
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time

# Import the modular model loader
from src.inference.model_loader import load_model

# Import the input schema
from src.api.schema import HeartDiseaseFeatures, PredictionResult

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- ML METRICS ----------------

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of predictions made by the model"
)

PREDICTION_BY_CLASS = Counter(
    "model_predictions_by_class_total",
    "Predictions grouped by output class",
    ["prediction"]
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time taken for model inference",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)

INFERENCE_ERRORS = Counter(
    "model_inference_errors_total",
    "Total inference errors"
)


# ---------------- LOAD MODEL ----------------
try:
    model = load_model()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk using a pre-trained ML pipeline",
    version="1.0",
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
    start_time = time.time()
    try:
        # Convert incoming JSON dict to DataFrame
        df = pd.DataFrame([patient.dict()])

        # Ensure categorical columns are of type 'object'
        categorical_cols = [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ]
        for col in categorical_cols:
            df[col] = df[col].astype("object")

        # Predict
        pred_prob = model.predict_proba(df)[:, 1]
        pred_class = model.predict(df)
        PREDICTION_COUNT.inc()
        PREDICTION_BY_CLASS.labels(
            prediction=str(int(pred_class[0]))
        ).inc()

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        return PredictionResult(
            prediction=int(pred_class[0]),
            probability=float(pred_prob[0]),
            message="High risk" if pred_class[0] == 1 else "Low risk",
        )
        # return {
        #     "prediction": int(pred_class[0]),
        #     "risk_probability": float(pred_prob[0])
        # }

    except Exception as e:
        INFERENCE_ERRORS.inc()
        logger.error(f"Inference error: {e}")
        return {"detail": f"Inference error: {str(e)}"}
