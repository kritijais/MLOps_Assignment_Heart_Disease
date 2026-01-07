import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from ucimlrepo import fetch_ucirepo

# The UCI dataset lacks headers, so we define them.
# The 'num' (index 13) is the target variable (0-4).
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# --- Feature Groups for Preprocessing ---

# Features that benefit from scaling (numerical)
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Features that are categorical and will be One-Hot Encoded
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

# Features with missing values that require imputation and encoding
IMPUTE_ENCODE_FEATURES = ["ca", "thal"]


# Data Acquisition & Data Cleaning
# No major missing values; verified with .isna().sum()
# Encoded categorical features: sex, cp, thal, slope
def load_and_clean_data() -> pd.DataFrame:
    """
    Loads the Heart Disease UCI dataset, handles missing values,
    and binarizes the target.
    """
    raw_data_path = "data/raw/heart_disease.csv"
    processed_data_path = "data/processed/heart_disease_processed.csv"

    # Ensure data/raw directory exists
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Check if processed data exists
    if os.path.exists(processed_data_path):
        return pd.read_csv(processed_data_path)

    # Check if raw data exists, if not download it
    if not os.path.exists(raw_data_path):
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = COLUMNS
        df.to_csv(raw_data_path, index=False)
    else:
        df = pd.read_csv(raw_data_path)

    # 5. Replace '?' with NaN
    df = df.replace("?", pd.NA)

    # 6. Convert numeric columns properly
    for col in NUMERICAL_FEATURES + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. Target Binarization
    # 0 = no disease, 1 = disease present
    df["heart_disease"] = (df["target"] > 0).astype(int)
    df = df.drop("target", axis=1)  # Drop original target column

    # 8. Convert categorical features to object type
    for col in CATEGORICAL_FEATURES + IMPUTE_ENCODE_FEATURES:
        df[col] = df[col].astype("object")

    # Save processed data
    df.to_csv(processed_data_path, index=False)

    return df


# Feature Engineering & Model Development
# Scaling → StandardScaler for continuous features
# One-hot encoding → Categorical features
# Pipeline created using sklearn.pipeline.Pipeline
def create_preprocessor() -> ColumnTransformer:
    """
    Creates and returns a scikit-learn ColumnTransformer for
    reproducible preprocessing. This includes imputation, scaling,
    and encoding.
    """

    # 1. Pipeline for Features Needing Imputation (ca, thal)
    # They are ordinal/categorical, so we impute with the most frequent value
    # (mode).
    imputer_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 2. Pipeline for Categorical Features (cp, restecg, etc.)
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    # 3. Pipeline for Numerical Features (age, chol, etc.)
    numerical_pipeline = Pipeline(
        [("scaler", StandardScaler())]  # Scaling to mean=0, std=1
    )

    # 4. Combine all pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("imp_enc", imputer_pipeline, IMPUTE_ENCODE_FEATURES),
        ],
        remainder="passthrough",  # Drop any columns not specified
        n_jobs=-1,
    )

    return preprocessor


# --- Example Usage (Not run during actual training/serving, just
# for demonstration) ---
if __name__ == "__main__":
    print("--- Running Data Pipeline Demo ---")

    # Load and clean
    data = load_and_clean_data()
    print(f"Loaded data shape: {data.shape}")
    print(f"Data columns: {data.columns}")
    print(f"Target balance:\n{data['heart_disease'].value_counts()}")

    # Separate features (X) and target (y)
    X = data.drop("heart_disease", axis=1)
    y = data["heart_disease"]

    # Split for training/testing (will be used in train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training data shape: {X_train.shape}")

    # Create the preprocessor
    preprocessor = create_preprocessor()

    # Fit the preprocessor on the training data
    print("Fitting preprocessor...")
    preprocessor.fit(X_train)

    # Transform a sample of the training data
    X_train_processed = preprocessor.transform(X_train)

    # The output is a NumPy array, ready for ML model training
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(
        f"Number of resulting features after encoding: "
        f"{X_train_processed.shape[1]}"
    )
    print("--- Data Pipeline Demo Complete ---")
