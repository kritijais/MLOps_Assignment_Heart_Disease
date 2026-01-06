## Heart Disease Prediction – MLOps Assignment

### Project Overview

(Link to the flow explaination video: https://github.com/kritijais/MLOps_Assignment_Heart_Disease/blob/main/Group_30_MLOPS_video.mp4)

This project is an MLOps pipeline for predicting heart disease using Machine Learning.
The ML lifecycle includes:

- Data preprocessing & EDA.
- Model training and evaluation.
- Experiment tracking with MLflow.
- CI & CI/CD pipelines using GitHub Actions.
- Containerization using Docker.
- Deployment on Kubernetes (Minikube).
- Observability with logging and Prometheus metrics.
  The final result is a production-ready FastAPI inference service deployed on Kubernetes.

### Project Structure

```
MLOps_Assignment_Heart_Disease/
│
├── .github/workflows/
│   ├── ci-cd-pipeline.yml
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
├── artifacts/                   # Trained model artifacts (ignored by Git)
│   └── best_model_pipeline.pkl  # Trained model pipeline (includes preprocessor)
├── data/                        # Raw and processed data (ignored by Git)
│   ├── raw/
│   └── processed/
├── eda_artifacts/               # EDA plots/images (ignored by Git)
├── mlruns/                      # MLflow tracking files (ignored by Git)
├── venv/                        # Python virtual environment (ignored by Git)
├── Dockerfile
├── Dockerfile.prometheus
├── prometheus.yml
├── requirements.txt
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

#### 3. Running MLflow UI & Tracking Experiments

##### 1. Start the MLflow Tracking UI

From the project root directory, run:

```bash
mlflow ui --port 5000
```

- MLflow UI will be available at:
  **[http://localhost:5000](http://localhost:5000)**
- Keep this terminal running while training the model.

> By default, MLflow stores runs locally in the `mlruns/` directory.

##### 2. Train the Model and Log Experiments

Open a **new terminal**, activate the virtual environment, and run:

```bash
python -m src.models.train
```

During training, MLflow logs:

- Model parameters
- Metrics (Accuracy, ROC-AUC, Precision, Recall, F1-score)
- Artifacts (trained model pipeline `.pkl`)

##### 3. View Experiments in MLflow UI

1. Open a browser and go to:
   **[http://localhost:5000](http://localhost:5000)**
2. Select the experiment name (e.g., `Heart_Disease_Models`)
3. Click on a run to view:

   - **Metrics** plotted over time
   - **Parameters** used for training
   - **Artifacts** (saved model and pipeline)

#### 4. Run the API Locally

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- **FastAPI docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** [http://localhost:8000/](http://localhost:8000/)

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

#### 5. Run Docker Container

```bash
docker build -t heart-disease-api .
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

#### 6. Production Deployment & Validation

```bash
# Build the Docker image for the FastAPI application
# -t assigns a name (tag) to the image
# . indicates the current directory containing the Dockerfile
docker build -t heart-disease-api .

# List all locally available Docker images
# Used to verify that the image was built successfully
docker images
```

- Then Deploy on render using the public image.
- **API endpoint:** https://heart-disease-api-latest-ib1p.onrender.com/docs
- **Metrics endpoint:** https://heart-disease-api-latest-ib1p.onrender.com/metrics
- **Prometheus endpoint:** https://heart-disease-prometheus.onrender.com/targets

### EDA and Modelling Choices

- Exploratory Data Analysis (EDA) was performed to understand the distribution of features and their relationship with the target variable.
- The dataset was preprocessed using `data_pipeline.py`, which includes handling missing values, encoding categorical variables, and scaling numerical features.
- Two models were trained: Logistic Regression and Random Forest. The best-performing model was selected based on metrics like Accuracy, ROC-AUC, Precision, Recall, and F1-score.

### Experiment Tracking Summary

- Experiment tracking was done using MLflow, which logged model parameters, metrics, and artifacts.
- The MLflow UI provides a visual representation of the experiments, allowing for easy comparison of different runs.

### Link to Code Repository

[https://github.com/kritijais/MLOps_Assignment_Heart_Disease](https://github.com/kritijais/MLOps_Assignment_Heart_Disease)


### MLOps Workflow

#### 1. Data & Feature Engineering

- Data loaded and cleaned by using `data_pipeline.py`.
- Categorical and numerical features processed by using `ColumnTransformer`.
- EDA plots are generated and logged as `eda-artifacts`.

#### 2. Model Training & Experiment Tracking

- Training for Logistic Regression and Random Forest models.
- Metrics logged:
  - Accuracy.
  - ROC-AUC.
  - Precision, Recall, F1 (via cross-validation).
- Final trained artifacts are saved as `best_model_pipeline.pkl` which includes the preprocessor.

#### 3. CI & CI/CD Pipelines

- CI Pipeline is triggered on every push.
  Steps:
  - Dependency installation
  - Linting (Flake8)
  - Unit tests (Pytest)
  - Model training with MLflow logging
- CI/CD Pipeline is triggered on push to "main", It ensures Code quality, Successful training and Deployment readiness.

#### 4. Containerization (Docker)

- FastAPI inference service packaged as a Docker image
- Model and preprocessor included as artifacts
- Image built using:
  ```bash
  docker build -t heart-disease-api:latest .
  ```

#### 5. Production Deployment (Render)

- Deployed on public server and the application will be up.

#### 6. Accessing the Application
- Access Metrics using: https://heart-disease-api-latest-ib1p.onrender.com/metrics
- Access docs using: https://heart-disease-api-latest-ib1p.onrender.com/docs
