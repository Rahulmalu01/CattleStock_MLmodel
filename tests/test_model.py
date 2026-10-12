import sys
sys.path.insert(0, '/c/Projects/mlmodel')

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.data_generation import generate_dataset
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model, save_metadata, load_metadata
from src.config import RANDOM_STATE, TEST_SIZE, METADATA_PATH
import tempfile


@pytest.fixture
def trained_model():
    """Create a trained model for testing."""
    df = generate_dataset()
    X, y, preprocessor, target_encoder = preprocess_data(df)
    assert target_encoder is not None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=10,  # Small number for testing
        max_depth=5,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test, preprocessor.feature_names


def test_model_training(trained_model):
    """Test model training."""
    model, X_test, y_test, _ = trained_model
    assert model is not None
    assert hasattr(model, 'predict')


def test_model_prediction(trained_model):
    """Test model prediction."""
    model, X_test, y_test, _ = trained_model
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert all(pred in model.classes_ for pred in predictions)


def test_model_prediction_probabilities(trained_model):
    """Test model probability predictions."""
    model, X_test, y_test, _ = trained_model
    probabilities = model.predict_proba(X_test)

    assert probabilities.shape[0] == len(X_test)
    assert probabilities.shape[1] == len(model.classes_)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_evaluate_model(trained_model):
    """Test model evaluation."""
    model, X_test, y_test, _ = trained_model
    metrics = evaluate_model(model, X_test, y_test, "TestModel")

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1


def test_model_accuracy_reasonable(trained_model):
    """Test that model achieves reasonable accuracy."""
    model, X_test, y_test, _ = trained_model
    metrics = evaluate_model(model, X_test, y_test, "TestModel")

    # Should achieve at least 60% accuracy on synthetic data
    assert metrics["accuracy"] > 0.6


def test_save_metadata(trained_model):
    """Test metadata saving."""
    model, X_test, y_test, feature_names = trained_model
    metrics = evaluate_model(model, X_test, y_test, "TestModel")

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = f"{tmpdir}/metadata.json"
        metadata = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        }

        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with open(metadata_path, "r") as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == metrics["accuracy"]


def test_model_feature_importance(trained_model):
    """Test model feature importance."""
    model, X_test, y_test, feature_names = trained_model

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        assert len(importances) == len(feature_names)
        assert sum(importances) > 0


def test_model_consistency(trained_model):
    """Test model prediction consistency."""
    model, X_test, y_test, _ = trained_model

    # Same input should give same prediction
    pred1 = model.predict(X_test[:1])
    pred2 = model.predict(X_test[:1])

    assert pred1[0] == pred2[0]
