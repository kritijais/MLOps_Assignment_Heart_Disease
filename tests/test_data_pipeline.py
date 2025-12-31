import pandas as pd

from src.features.data_pipeline import (
    load_and_clean_data,
    create_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    IMPUTE_ENCODE_FEATURES,
)


def test_load_and_clean_data_shape():
    df = load_and_clean_data()
    # Expect at least 1 row and all required columns
    expected_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + IMPUTE_ENCODE_FEATURES
        + ["heart_disease"]
    )
    assert all(col in df.columns for col in expected_columns)
    assert df.shape[0] > 0


def test_categorical_columns_dtype():
    df = load_and_clean_data()
    for col in CATEGORICAL_FEATURES + IMPUTE_ENCODE_FEATURES:
        assert df[col].dtype == "object"


def test_numerical_columns_dtype():
    df = load_and_clean_data()
    for col in NUMERICAL_FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_preprocessor_transform():
    df = load_and_clean_data()
    X = df.drop("heart_disease", axis=1)
    preprocessor = create_preprocessor()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    assert X_transformed.shape[0] == X.shape[0]  # same number of rows
