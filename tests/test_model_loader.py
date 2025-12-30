import pytest
from src.inference.model_loader import load_model

def test_load_model_file_exists():
    model = load_model()
    # The loaded object should have predict and predict_proba methods
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
