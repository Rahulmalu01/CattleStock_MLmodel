import sys
sys.path.insert(0, '/c/Projects/mlmodel')

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import Preprocessor, preprocess_data
from src.data_generation import generate_dataset


@pytest.fixture
def sample_df():
    """Create sample dataset for testing."""
    return generate_dataset()


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = Preprocessor()
    assert preprocessor.scaler is not None
    assert preprocessor.label_encoder is not None


def test_preprocessor_fit(sample_df):
    """Test preprocessor fitting."""
    X = sample_df.drop("disease", axis=1)
    preprocessor = Preprocessor()
    preprocessor.fit(X)

    assert preprocessor.feature_names is not None
    assert preprocessor.numeric_features is not None
    assert len(preprocessor.feature_names) == X.shape[1]


def test_preprocessor_transform(sample_df):
    """Test preprocessor transformation."""
    X = sample_df.drop("disease", axis=1)
    preprocessor = Preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == X.shape[1]
    assert isinstance(X_transformed, np.ndarray)


def test_preprocessor_scaling(sample_df):
    """Test that numeric features are scaled."""
    X = sample_df.drop("disease", axis=1)
    preprocessor = Preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Check that scaled features are in reasonable range (roughly -3 to 3)
    numeric_cols = [i for i, col in enumerate(preprocessor.feature_names)
                   if col != "activity_level"]
    for col_idx in numeric_cols:
        col_data = X_transformed[:, col_idx]
        assert col_data.std() > 0.5  # Should have variance
        assert col_data.mean() < 1  # Mean should be close to 0


def test_preprocessor_categorical_encoding(sample_df):
    """Test categorical variable encoding."""
    X = sample_df.drop("disease", axis=1)
    preprocessor = Preprocessor()
    preprocessor.fit(X)

    # Check that activity_level is encoded
    assert preprocessor.label_encoder.classes_ is not None
    assert len(preprocessor.label_encoder.classes_) >= 1


def test_preprocess_data(sample_df):
    """Test complete preprocessing pipeline."""
    X, y, preprocessor, target_encoder = preprocess_data(sample_df)

    assert X.shape[0] == sample_df.shape[0]
    assert X.shape[1] == sample_df.shape[1] - 1
    assert len(y) == sample_df.shape[0]
    assert preprocessor is not None
    assert target_encoder is not None


def test_preprocessor_consistency(sample_df):
    """Test that preprocessor is consistent across runs."""
    X = sample_df.drop("disease", axis=1)
    preprocessor = Preprocessor()
    preprocessor.fit(X)

    X_trans1 = preprocessor.transform(X.iloc[:10])
    X_trans2 = preprocessor.transform(X.iloc[:10])

    np.testing.assert_array_almost_equal(X_trans1, X_trans2)


def test_preprocessor_preserves_order(sample_df):
    """Test that preprocessor preserves sample order."""
    X = sample_df.drop("disease", axis=1)
    preprocessor = Preprocessor()
    preprocessor.fit(X)

    # Transform two random samples
    sample1 = X.iloc[5:6]
    sample2 = X.iloc[10:11]

    trans1 = preprocessor.transform(sample1)
    trans2 = preprocessor.transform(sample2)

    # Transform both together
    trans_combined = preprocessor.transform(pd.concat([sample1, sample2]))

    np.testing.assert_array_almost_equal(trans1, trans_combined[0:1])
    np.testing.assert_array_almost_equal(trans2, trans_combined[1:2])
