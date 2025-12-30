## Heart Disease Prediction – MLOps Assignment

### Project Overview
This project is an MLOps pipeline for predicting heart disease using Machine Learning.
The ML lifecycle includes:
-Data preprocessing & EDA.
-Model training and evaluation.
-Experiment tracking with MLflow.
-CI & CI/CD pipelines using GitHub Actions.
-Containerization using Docker.
-Deployment on Kubernetes (Minikube).
-Observability with logging and Prometheus metrics.
The final result is a production-ready FastAPI inference service deployed on Kubernetes.

### Project Structure

```
MLOps_Assignment_Heart_Disease/
│
├── .github/workflows/
│   ├── ci_pipeline.yml
│   └── ci-cd-pipeline.yml
│
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   └── schema.py            # Request/response schemas
│   │
│   ├── features/
│   │   ├── data_pipeline.py     # Data loading & preprocessing
│   │   └── eda_plots.py         # EDA visualizations
│   │
│   ├── inference/
│   │   └── model_loader.py      # Loading pre-trained ML pipeline
│   │
│   ├── models/
│   │   └── train.py             # Model training & MLflow logging
│
├── tests/
│   ├── test_data_pipeline.py    # Unit tests for data pipeline
│   ├── test_model_loader.py     # Unit tests for model loader
│   └── test_api.py              # Unit tests for API
│
├── Kubernetes/
│   ├── deployment.yaml          # K8s Deployment
│   └── service.yaml             # K8s Service
│
├── Dockerfile
├── requirements.txt
├── artifacts/
│   └── best_model_pipeline.pkl  # Trained model pipeline (includes preprocessor)
├── eda_artifacts/               # EDA plots/images
└── README.md
```


### Setup & Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/kritijais/MLOps_Assignment_Heart_Disease.git
cd MLOps_Assignment_Heart_Disease
```

#### 2. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Run the API Locally

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

* **FastAPI docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health check:** [http://localhost:8000/](http://localhost:8000/)

**Sample Predict Request:**

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

#### 4. Run Docker Container

```bash
docker build -t heart-disease-api .
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

* **API endpoint:** [http://localhost:8000](http://localhost:8000)
* **Metrics endpoint:** [http://localhost:8000/metrics](http://localhost:8000/metrics)


### MLOps Workflow

#### 1. Data & Feature Engineering
-Data loaded and cleaned by using data_pipeline.py
-Categorical and numerical features processed by using ColumnTransformer.
-EDA plots are generated and logged as eda-artifacts.

#### 2. Model Training & Experiment Tracking
-Training for Logistic Regression and Random Forest models.
-Metrics logged:
    --Accuracy.
    --ROC-AUC.
    --Precision, Recall, F1 (via cross-validation).
-Final trained artifacts are saved as model.pkl, preprocessor.pkl.

#### 3. CI & CI/CD Pipelines
-CI Pipeline is triggered on every push.
    Steps:
    -Dependency installation
    -Linting (Flake8)
    -Unit tests (Pytest)
    -Model training with MLflow logging
-CI/CD Pipeline is triggered on push to "main",It ensures Code quality, Successful training and Deployment readiness.

#### 4. Containerization (Docker)
-FastAPI inference service packaged as a Docker image
-Model and preprocessor included as artifacts
-Image built using:
    docker build -t heart-disease-api:latest .

#### 5. Kubernetes Deployment (Minikube)
-Deployed on a local Kubernetes cluster using Minikube
-Resources:
    -Deployment: Manages FastAPI pods
    -Service (NodePort): Exposes the application

#### 6. Accessing the Application
-Access application on browser by starting the service using:
     minikube service heart-disease-service
-Access Metrics using: link/metrics
-Access docs using: link/docs