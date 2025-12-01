import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Union

# Define the expected artifact paths relative to the project root
MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

# CRITICAL: Define the exact order and names of the features 
# that the preprocessor expects, based on your original data pipeline.
# This ensures consistency between training and serving.
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Global variables to hold the loaded model and preprocessor
MODEL = None
PREPROCESSOR = None

def load_artifacts():
    """
    Loads the trained model and preprocessor pipeline from disk.
    This function should be called once when the FastAPI server starts.
    """
    global MODEL, PREPROCESSOR
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            print(f"ERROR: Model or Preprocessor files not found at {MODEL_PATH} or {PREPROCESSOR_PATH}.")
            print("Please ensure your training script has saved these artifacts.")
            return
            
        MODEL = joblib.load(MODEL_PATH)
        PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
        print("Model and Preprocessor artifacts loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading artifacts: {e}")
        MODEL = None
        PREPROCESSOR = None


def predict(data: Dict[str, Union[float, int]]) -> Dict[str, Any]:
    """
    Performs preprocessing and inference on the input data.

    Args:
        data: A dictionary containing the input features for one patient.

    Returns:
        A dictionary containing the prediction (0 or 1) and probability.
    """
    if MODEL is None or PREPROCESSOR is None:
        return {
            "prediction": -1,
            "probability": 0.0,
            "message": "Model not loaded. Check server logs."
        }

    try:
        # 1. Convert input dictionary to a pandas DataFrame
        # CRITICAL FIX: Ensure correct column order, and use explicit numpy array creation
        # to guarantee float64 dtype before passing to the pipeline.
        
        # Create a list of values in the correct order
        input_values = [[data[col] for col in FEATURE_COLUMNS]]
        
        # Convert to NumPy array with explicit float64 dtype
        input_array = np.array(input_values, dtype=np.float64)
        
        # Create the final DataFrame
        # This DataFrame is now guaranteed to have float64 numeric data for the pipeline.
        input_df = pd.DataFrame(input_array, columns=FEATURE_COLUMNS)
        
        # Ensure that the data doesn't contain any unexpected types
        assert input_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all(), "Non-numeric values found"
        
        # --- DEBUGGING START ---
        print("\n--- DEBUG INFO BEFORE PREDICTION ---")
        print("Input DataFrame Head:")
        print(input_df.head())
        print("\nInput DataFrame dtypes:")
        print(input_df.dtypes)
        print("-----------------------------------\n")
        # --- DEBUGGING END ---
        
        # Since the provided train.py saves the *full pipeline* as model.pkl,
        # we use the full pipeline (MODEL) which handles preprocessing internally.
        
        # 3. Predict probability (The full pipeline handles preprocessing internally)
        # Predict_proba returns probabilities for [class 0, class 1]
        proba = MODEL.predict_proba(input_df)[0]
        positive_class_proba = round(proba[1], 4)
        
        # 4. Predict the class (0 or 1)
        prediction = int(MODEL.predict(input_df)[0])

        # 5. Generate message
        message = "High likelihood of heart disease." if prediction == 1 else "Low likelihood of heart disease."

        return {
            "prediction": prediction,
            "probability": positive_class_proba,
            "message": message
        }

    except Exception as e:
        # We need to print the actual exception to the server console
        print(f"Prediction error: {e}")
        return {
            "prediction": -1,
            "probability": 0.0,
            "message": f"Prediction failed due to internal error: {e}"
        }

# Load artifacts immediately when the module is imported (i.e., when the server starts)
load_artifacts()
