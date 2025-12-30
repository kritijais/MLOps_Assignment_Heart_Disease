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
MLOps_Assignment_Heart_Disease/
│
├── .github/workflows/
│   ├── ci_pipeline.yml
│   └── ci-cd-pipeline.yml
│
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── model_handler.py     # Model loading & inference
│   │   └── schemas.py           # Request/response schemas
│   │
│   ├── features/
│   │   ├── data_pipeline.py     # Data loading & preprocessing
│   │   └── eda_plots.py         # EDA visualizations
│   │
│   ├── training/
│   │   └── train.py             # Model training & MLflow logging
│   │
│   └── train.py                 # Entry-point training script
│
├── tests/
│   └── test_data_pipeline.py    # Unit tests
│
├── Kubernetes/
│   ├── deployment.yaml          # K8s Deployment
│   └── service.yaml             # K8s Service
│
├── Dockerfile
├── requirements.txt
├── model.pkl
├── preprocessor.pkl
└── README.md

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