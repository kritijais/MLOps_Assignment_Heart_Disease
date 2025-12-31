import joblib
import os

# Path to the saved best pipeline
MODEL_PATH = os.path.join("artifacts", "best_model_pipeline.pkl")


def load_model():
    """
    Loads the pre-trained ML pipeline from disk.

    Returns:
        model (Pipeline): scikit-learn pipeline with preprocessing + classifier
    Raises:
        FileNotFoundError: if the model file does not exist
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
    return model
