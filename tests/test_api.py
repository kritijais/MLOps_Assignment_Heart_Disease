from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"].startswith("Heart Disease Prediction API")


def test_predict_endpoint():
    sample_payload = {
        "age": 63.0,
        "sex": 1,
        "cp": 3,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
