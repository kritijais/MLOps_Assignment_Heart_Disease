import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
import os
import mlflow
import mlflow.sklearn

# Import the functions we fixed and created earlier
from features.data_pipeline import load_and_clean_data, create_preprocessor 

# --- CONFIGURATION ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_ARTIFACT_PATH = "model.pkl"
PREPROCESSOR_ARTIFACT_PATH = "preprocessor.pkl"
MODEL_NAME = "LogisticRegression"
MLFLOW_EXPERIMENT_NAME = "Heart_Disease_Prediction"


def evaluate_model(model: ClassifierMixin, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """
    Evaluates a model using k-fold cross-validation and standard metrics.
    
    Args:
        model: The scikit-learn classifier model.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of cross-validation folds.
        
    Returns:
        A dictionary of averaged cross-validation metrics.
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'f1': 'f1'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Calculate average metrics
    metrics = {
        'mean_accuracy': cv_results['test_accuracy'].mean(),
        'mean_precision': cv_results['test_precision'].mean(),
        'mean_recall': cv_results['test_recall'].mean(),
        'mean_roc_auc': cv_results['test_roc_auc'].mean(),
        'mean_f1': cv_results['test_f1'].mean()
    }
    return metrics

def train_and_save_model():
    """
    1. Loads and processes the data.
    2. Trains a Logistic Regression model within a scikit-learn Pipeline.
    3. Logs the model to MLflow.
    4. Saves the final trained Pipeline (model) and the fitted preprocessor locally.
    """
    print("--- Starting Model Training Pipeline ---")
    
    # 1. Initialize MLflow Run
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # 2. Load Data
        try:
            df = load_and_clean_data()
            print(f"Data loaded successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"FATAL ERROR: Could not load data. Check data_pipeline.py and the URL. Error: {e}")
            return

        # 3. Define Features (X) and Target (y)
        X = df.drop('heart_disease', axis=1)
        y = df['heart_disease']

        # 4. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Data split: Training size={len(X_train)}, Testing size={len(X_test)}")

        # 5. Create Preprocessor and Model Pipeline
        preprocessor = create_preprocessor()
        model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
        
        # Log hyperparameters
        mlflow.log_params({
            "model_type": MODEL_NAME,
            "solver": model.solver,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE
        })

        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        print("Full ML Pipeline created.")

        # 6. Train the Pipeline
        print("Training model...")
        full_pipeline.fit(X_train, y_train)
        print("Training complete.")

        # 7. Evaluation & Metric Logging (Using the held-out test set)
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_roc_auc": roc_auc
        })
        print(f"\nModel Test Accuracy: {accuracy:.4f}")
        print(f"Model Test ROC AUC: {roc_auc:.4f}")
        
        # 8. Log the Model Artifact to MLflow
        mlflow.sklearn.log_model(
            sk_model=full_pipeline, 
            artifact_path="model_pipeline",
            registered_model_name=f"HeartDisease_{MODEL_NAME}_Model"
        )
        print(f"3. Full pipeline logged to MLflow under run: {run_id}")

        # --- LOCAL ARTIFACT SAVING (FOR FASTAPI DEPLOYMENT) ---
        # 9. Save the Full Pipeline locally (for FastAPI to load as model.pkl)
        joblib.dump(full_pipeline, MODEL_ARTIFACT_PATH)
        print(f"1. Full trained model (Pipeline) saved locally to: {MODEL_ARTIFACT_PATH}")
        
        # 10. Save the FITTED Preprocessor separately (CRITICAL for FastAPI)
        # We extract the fitted preprocessor from the full pipeline and save it.
        fitted_preprocessor = full_pipeline['preprocessor']
        joblib.dump(fitted_preprocessor, PREPROCESSOR_ARTIFACT_PATH)
        print(f"2. Fitted preprocessor saved locally to: {PREPROCESSOR_ARTIFACT_PATH}")
        
    print("\nTraining, evaluation, and artifact saving finished successfully.")


if __name__ == "__main__":
    train_and_save_model()