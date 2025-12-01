import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# The UCI dataset lacks headers, so we define them.
# The 'num' (index 13) is the target variable (0-4).
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# --- Feature Groups for Preprocessing ---

# Features that benefit from scaling (numerical)
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Features that are categorical and will be One-Hot Encoded
CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'sex', 'fbs', 'exang']

# Features with missing values that require imputation and encoding
IMPUTE_ENCODE_FEATURES = ['ca', 'thal']

def load_and_clean_data(url: str = None) -> pd.DataFrame:
    """
    Loads the Heart Disease UCI dataset, handles missing values, and binarizes the target.
    """
    if url is None:
        # Default URL for the Cleveland dataset
        #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        #url = "https://archive.ics.uci.edu/dataset/45/heart+disease"

    # 1. Data Acquisition
    # Use '?' as the marker for missing values. 
    # Use 'encoding' and 'delim_whitespace=True' to handle the file format quirks.
    df = pd.read_csv(
        url, 
        names=COLUMNS, 
        na_values='?',
        #delim_whitespace=True, 
        sep=',',              
        encoding='iso-8859-1' 
    )

    # 2. Target Binarization
    # The original target is 0-4. We binarize: 0=No disease, >0=Disease present.
    df['heart_disease'] = (df['target'] > 0).astype(int)
    df = df.drop('target', axis=1) # Drop original target column

    # 3. Type Conversion
    # Convert features known to be categorical/ordinal to object type
    for col in CATEGORICAL_FEATURES + IMPUTE_ENCODE_FEATURES:
        df[col] = df[col].astype('object')

    return df

def create_preprocessor() -> ColumnTransformer:
    """
    Creates and returns a scikit-learn ColumnTransformer for reproducible preprocessing.
    This includes imputation, scaling, and encoding.
    """
    
    # 1. Pipeline for Features Needing Imputation (ca, thal)
    # They are ordinal/categorical, so we impute with the most frequent value (mode).
    imputer_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 2. Pipeline for Categorical Features (cp, restecg, etc.)
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Pipeline for Numerical Features (age, chol, etc.)
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler()) # Scaling to mean=0, std=1
    ])

    # 4. Combine all pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
            ('imp_enc', imputer_pipeline, IMPUTE_ENCODE_FEATURES)
        ],
        remainder='passthrough',  # Drop any columns not specified
        n_jobs=-1
    )
    
    return preprocessor

# --- Example Usage (Not run during actual training/serving, just for demonstration) ---
if __name__ == '__main__':
    print("--- Running Data Pipeline Demo ---")
    
    # Load and clean
    data = load_and_clean_data()
    print(f"Loaded data shape: {data.shape}")
    print(f"Target balance:\n{data['heart_disease'].value_counts()}")
    
    # Separate features (X) and target (y)
    X = data.drop('heart_disease', axis=1)
    y = data['heart_disease']
    
    # Split for training/testing (will be used in train.py)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
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
    print(f"Number of resulting features after encoding: {X_train_processed.shape[1]}")
    print("--- Data Pipeline Demo Complete ---")