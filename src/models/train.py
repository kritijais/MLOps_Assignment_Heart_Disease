import os
import joblib
import platform
import sklearn
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
                             roc_auc_score,
                             accuracy_score,
                             precision_score,
                             recall_score, f1_score
                            )
from src.features.data_pipeline import load_and_clean_data, create_preprocessor
from src.features.eda_plots import generate_and_log_eda

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "Heart_Disease_Prediction"
ARTIFACT_DIR = "artifacts"
BEST_MODEL_PATH = f"{ARTIFACT_DIR}/best_model_pipeline.pkl"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------------------
def train_and_select_best_model():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---- Load data ----
    df = load_and_clean_data()

    # ---- Generate and log EDA plots ----
    generate_and_log_eda(df)

    # ---- Split features and target ----
    X = df.drop("heart_disease", axis=1)
    y = df["heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Candidate models ----
    models = {
        "LogisticRegression": LogisticRegression(
            solver="liblinear", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    best_auc = 0.0
    best_pipeline = None
    best_model_name = None

    # ---- Train & evaluate each model ----
    for model_name, classifier in models.items():

        with mlflow.start_run(run_name=model_name):

            # ---- Reproducibility metadata ----
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "python_version": platform.python_version(),
                    "sklearn_version": sklearn.__version__,
                    "test_size": TEST_SIZE,
                }
            )

            # ---- Build full pipeline ----
            pipeline = Pipeline(
                [("preprocessor", create_preprocessor()), ("classifier", classifier)]
            )

            # ---- Cross-validation ----
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
            )

            cv_auc = cv_results["test_score"].mean()
            mlflow.log_metric("cv_roc_auc", cv_auc)

            # ---- Train on full training set ----
            pipeline.fit(X_train, y_train)

            # ---- Test evaluation ----
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred = pipeline.predict(X_test)

            test_auc = roc_auc_score(y_test, y_prob)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)

            mlflow.log_metrics(
                {
                    "test_roc_auc": test_auc,
                    "test_accuracy": test_accuracy,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                }
            )

            # ---- Log model to MLflow ----
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(
                f"{model_name} | "
                f"CV AUC: {cv_auc:.4f} | "
                f"Test AUC: {test_auc:.4f} | "
                f"Acc: {test_accuracy:.4f} | "
                f"Prec: {test_precision:.4f} | "
                f"Recall: {test_recall:.4f}"
            )

            # ---- Track best model ----
            if test_auc > best_auc:
                best_auc = test_auc
                best_pipeline = pipeline
                best_model_name = model_name

    # ---- Save & register BEST model only ----
    print(f"\nBest model selected: {best_model_name} (ROC-AUC = {best_auc:.4f})")

    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    with mlflow.start_run(run_name="Best_Model_Registration"):
        mlflow.log_params(
            {
                "selected_model": best_model_name,
                "selection_metric": "roc_auc",
                "best_roc_auc": best_auc,
            }
        )

        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            registered_model_name="HeartDiseaseClassifier",
        )

    print(f"Best model saved to: {BEST_MODEL_PATH}")


# ---------------------------------------
if __name__ == "__main__":
    train_and_select_best_model()
