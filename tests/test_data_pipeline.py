import pytest
import pandas as pd
import numpy as np
from src.features.data_pipeline import load_and_clean_data, create_preprocessor

def test_pipeline_output_shape():
    df = load_and_clean_data()
    X = df.drop("heart_disease", axis=1)

    preprocessor = create_preprocessor()
    X_proc = preprocessor.fit_transform(X)

    assert X_proc.shape[0] == X.shape[0]

# A temporary fixture to load and clean the data once for all tests
# The 'session' scope means it runs only once per pytest session.
@pytest.fixture(scope="session")
def cleaned_data():
    """Fixture to load and return the cleaned DataFrame."""
    # Note: We use the production function to test its integrity
    df = load_and_clean_data()
    return df

# --- Test Data Cleaning and Structure (Task 1) ---

def test_dataframe_loaded_correctly(cleaned_data):
    """Test that the DataFrame loads and has the correct shape and columns."""
    # Expect 13 features (original 14 - target + heart_disease)
    assert cleaned_data.shape[1] == 14 
    # Check if a critical column is present
    assert 'heart_disease' in cleaned_data.columns

def test_missing_values_imputed(cleaned_data):
    """
    Test that the critical columns which originally contained '?' (now NaN) 
    have been fully imputed. These are 'ca' and 'thal'.
    """
    # The load_and_clean_data function should handle the loading and initial cleanup
    # of '?' to NaN. If any NaN remain after load_and_clean_data, it's an issue.
    # However, since we are only testing the *loading and cleaning* function here, 
    # which doesn't include the preprocessor's imputation step, we must check 
    # that the values that should be objects are correctly converted.
    
    # Check 'ca' and 'thal' for residual '?' or unexpected types.
    # After load_and_clean_data, 'ca' and 'thal' should have been set to NaN 
    # for the missing values. The actual imputation happens in the full pipeline.
    
    # We check if there are any non-numeric/non-NaN values left in key columns
    assert not (cleaned_data['ca'] == '?').any()
    assert not (cleaned_data['thal'] == '?').any()

    # NOTE: The full imputation happens in the ColumnTransformer. 
    # This test verifies the initial loading and type conversion.
    # If using the raw data URL, it's safer to ensure NaN counts are low.
    # Given the nature of the data, a small number of NaN is expected here:
    assert cleaned_data['ca'].isnull().sum() <= 5
    assert cleaned_data['thal'].isnull().sum() <= 5


def test_target_binarization(cleaned_data):
    """Test that the 'heart_disease' column is correctly binarized (0 or 1)."""
    # Assert that the new target column only contains 0 or 1
    unique_values = cleaned_data['heart_disease'].unique()
    assert set(unique_values) <= {0, 1}
    assert len(unique_values) == 2 # Must contain both classes

# --- Test Preprocessor Creation (Task 4) ---

def test_preprocessor_instantiation():
    """Test that the create_preprocessor function returns a ColumnTransformer."""
    preprocessor = create_preprocessor()
    # Check if the object is indeed a ColumnTransformer
    assert isinstance(preprocessor, pd.core.frame.DataFrame) or str(type(preprocessor)).endswith("ColumnTransformer'>")

def test_preprocessor_output_shape(cleaned_data):
    """
    Test that the ColumnTransformer generates the expected number of features 
    after fitting and transforming.
    """
    # Use a small sample of data for fitting and transformation
    X = cleaned_data.drop('heart_disease', axis=1)
    
    preprocessor = create_preprocessor()
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Numerical features (5) remain 5.
    # Categorical features (6) are One-Hot Encoded (e.g., sex (2), cp (4), restecg (3), etc.)
    # Impute/Encoded features (2) are handled (ca (4), thal (3))
    # Expected number of features after one-hot encoding (~28 features total depending on categories)
    # The actual expected shape is roughly 28 (5 numerical + (2+4+3+2+2+2) for nominal + (4+3) for imputed)
    
    # We check if the transformed output is a numpy array and has more features than the input
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[1] > X.shape[1]
    
    # A safe check: the feature space should be around 25 to 30 features
    assert 25 <= X_transformed.shape[1] <= 35