import mlflow
import os
import shutil
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Import custom modules ---
from src.features.data_pipeline import load_and_clean_data, create_preprocessor
from src.features.eda_plots import generate_and_log_eda
from src.training.train import train_and_log_model

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Heart_Disease_Prediction_Full_Cycle")

def run_ml_pipeline():
    """
    Orchestrates the entire ML development cycle: Data -> EDA -> Training -> Logging.
    """
    print("--- Starting MLOps Project Pipeline ---")
    
    # --- 1. Data Acquisition, Cleaning, and Preparation (Task 1 & 4) ---
    try:
        data = load_and_clean_data()
        X = data.drop('heart_disease', axis=1)
        y = data['heart_disease']
        
        # Split data for training/validation (20% reserved for later testing if needed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # Create the preprocessor pipeline object (Task 4)
        preprocessor = create_preprocessor()

    except Exception as e:
        print(f"Fatal Error during data preparation: {e}")
        return

    # --- 2. EDA and Visualization (Task 1 & 3) ---
    run_name_eda = "initial_eda_and_data_prep"
    with mlflow.start_run(run_name=run_name_eda) as run:
        print(f"MLflow EDA Run ID: {run.info.run_id}")
        artifact_dir = "temp_eda_plots"
        generate_and_log_eda(data, artifact_dir)
        
        # Clean up
        if os.path.exists(artifact_dir):
            shutil.rmtree(artifact_dir)

    # --- 3. Model Training and Tracking (Task 2, 3, 4) ---
    
    # 3.1. Logistic Regression Model
    lr_params = {'penalty': 'l2', 'C': 0.1, 'solver': 'liblinear', 'random_state': 42}
    with mlflow.start_run(run_name="Logistic_Regression_v1", nested=True) as run_lr:
        train_and_log_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name="LogisticRegression",
            model=LogisticRegression(**lr_params),
            params=lr_params
        )

    # 3.2. Random Forest Model
    rf_params = {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 2, 'random_state': 42}
    with mlflow.start_run(run_name="Random_Forest_v1", nested=True) as run_rf:
        train_and_log_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name="RandomForest",
            model=RandomForestClassifier(**rf_params),
            params=rf_params
        )
        
    print("\n--- Pipeline Complete ---")
    print(f"MLflow experiments saved locally in the '{MLFLOW_TRACKING_URI}' directory.")
    print("Run 'mlflow ui' to compare the results of the two models.")

if __name__ == '__main__':
    run_ml_pipeline()