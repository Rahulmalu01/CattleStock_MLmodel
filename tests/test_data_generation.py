import sys
sys.path.insert(0, '/c/Projects/mlmodel')

import pytest
import pandas as pd
from src.data_generation import generate_disease_samples, generate_dataset
from src.config import DISEASE_CLASSES, DATASET_SIZE


def test_generate_disease_samples():
    """Test disease sample generation."""
    for disease in DISEASE_CLASSES:
        df = generate_disease_samples(disease, 100)
        assert len(df) == 100
        assert df["disease"].unique()[0] == disease
        assert all(col in df.columns for col in [
            "temperature", "heart_rate", "respiratory_rate",
            "steps", "activity_level", "feeding_efficiency",
            "milk_yield", "milk_conductivity"
        ])


def test_generate_dataset(tmp_path):
    """Test complete dataset generation."""
    output_file = tmp_path / "test_dataset.csv"
    df = generate_dataset(str(output_file))

    assert len(df) >= DATASET_SIZE
    assert output_file.exists()
    assert df["disease"].nunique() == len(DISEASE_CLASSES)

    # Check disease distribution is balanced
    distribution = df["disease"].value_counts()
    min_count = distribution.min()
    max_count = distribution.max()
    assert (max_count - min_count) <= 100  # Allow small imbalance


def test_dataset_feature_ranges():
    """Test that synthetic data falls within expected ranges."""
    df = generate_disease_samples("healthy", 50)

    assert df["temperature"].min() >= 37.5
    assert df["temperature"].max() <= 39.0
    assert df["heart_rate"].min() >= 60
    assert df["heart_rate"].max() <= 80
    assert df["feeding_efficiency"].min() >= 80
    assert df["feeding_efficiency"].max() <= 100


def test_fever_data():
    """Test fever-specific data generation."""
    df = generate_disease_samples("fever", 50)

    assert df["temperature"].min() >= 39.5
    assert df["heart_rate"].min() >= 85
    assert df["respiratory_rate"].min() >= 30
    assert (df["activity_level"] == "low").all()


def test_heat_stress_data():
    """Test heat stress-specific data generation."""
    df = generate_disease_samples("heat_stress", 50)

    assert df["temperature"].min() >= 39.2
    assert df["heart_rate"].min() >= 90
    assert (df["activity_level"] == "low").all()
    assert df["feeding_efficiency"].mean() < 70


def test_mastitis_data():
    """Test mastitis-specific data generation."""
    df = generate_disease_samples("mastitis", 50)

    assert df["milk_conductivity"].min() >= 4.8
    assert df["milk_yield"].max() <= 15


def test_digestive_disorder_data():
    """Test digestive disorder-specific data generation."""
    df = generate_disease_samples("digestive_disorder", 50)

    assert df["rumination_duration"].mean() < 250
    assert df["feeding_efficiency"].mean() < 70
    assert (df["activity_level"] == "low").all()
