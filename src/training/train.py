import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin

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

def train_and_log_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    preprocessor: Pipeline,
    model_name: str, 
    model: ClassifierMixin, 
    params: dict
):
    """
    Builds the full pipeline, trains the model, logs parameters, metrics, 
    and the final model artifact to MLflow.
    """
    print(f"\n--- Training {model_name} ---")
    
    # Create the full ML pipeline: Preprocessing + Model (Task 4: Reproducibility)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Log hyperparameters (Task 3)
    mlflow.log_params(params)
    
    # 1. Train the full pipeline
    full_pipeline.fit(X_train, y_train)
    
    # 2. Evaluate using Cross-Validation
    # Note: We use X_train and y_train here for cross-validation evaluation
    # to avoid data leakage from a separate test set, aligning with best practice.
    metrics = evaluate_model(full_pipeline, X_train, y_train, cv=5)
    
    # Log metrics (Task 3)
    mlflow.log_metrics(metrics)
    print(f"Logged cross-validation metrics for {model_name}: {metrics}")

    # 3. Log the entire pipeline (model packaging) (Task 4)
    # This logs the preprocessor and classifier together for production use.
    mlflow.sklearn.log_model(
        sk_model=full_pipeline, 
        artifact_path="model",
        registered_model_name=f"HeartDisease_{model_name}_Model"
    )
    print(f"Logged full pipeline for {model_name} to MLflow.")

    return full_pipeline, metrics