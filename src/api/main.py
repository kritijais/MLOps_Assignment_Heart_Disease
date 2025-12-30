import logging
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from prometheus_fastapi_instrumentator import Instrumentator

# Assuming these files are in the same directory structure (src/api)
from .schemas import HeartDiseaseFeatures, PredictionResult
from .model_handler import predict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting the presence of heart disease using UCI Cleveland dataset features.",
    version="1.0.0"
)

# --- Prometheus Metrics ---
Instrumentator().instrument(app).expose(app)

# --- Endpoints ---

@app.get("/", tags=["Health Check"])
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "message": "Heart Disease Prediction API is running."}

@app.post(
    "/predict", 
    response_model=PredictionResult, 
    status_code=200, 
    tags=["Prediction"]
)
async def get_prediction(data: HeartDiseaseFeatures) -> PredictionResult:
    """
    Accepts patient feature data and returns the heart disease prediction (0 or 1) 
    and the associated probability.
    """
    logger.info("Received prediction request")
    # Convert Pydantic model data to a standard dictionary for the prediction function
    input_data_dict = data.model_dump()
    
    # Get the prediction results from the model handler
    result = predict(input_data_dict)
    
    # Check if the prediction failed (e.g., if model handler returned -1)
    if result["prediction"] == -1:
         logger.error("Prediction failed: %s", result["message"])
         raise HTTPException(
            status_code=500, 
            detail=result["message"]
        )
    
    logger.info("Prediction successful")
    # Return the result which matches the PredictionResult schema
    return PredictionResult(**result)
