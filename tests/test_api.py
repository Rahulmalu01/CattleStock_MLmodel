import sys
sys.path.insert(0, '/c/Projects/mlmodel')

import pytest
from fastapi.testclient import TestClient
from src.data_generation import generate_dataset
from src.training import train_pipeline
from src.config import DATASET_PATH, MODEL_PATH

# Initialize API and training before tests
@pytest.fixture(scope="session", autouse=True)
def setup_models():
    """Set up models for testing."""
    # Generate dataset
    if not DATASET_PATH.exists():
        generate_dataset()

    # Train models
    if not MODEL_PATH.exists():
        train_pipeline()


@pytest.fixture
def client():
    """Create FastAPI test client with startup/shutdown lifecycle."""
    from src.api import app
    with TestClient(app) as client:
        yield client


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200

    data = response.json()
    assert "model_version" in data
    assert "accuracy" in data
    assert "features" in data
    assert isinstance(data["features"], list)


def test_predict_endpoint_healthy_cattle(client):
    """Test prediction endpoint with healthy cattle."""
    payload = {
        "cattle_id": "COW_TEST_001",
        "temperature": 38.5,
        "heart_rate": 70,
        "respiratory_rate": 25,
        "steps": 3500,
        "activity_level": "normal",
        "activity_duration_minutes": 450,
        "rumination_duration": 500,
        "rumination_episodes": 10,
        "feeding_efficiency": 90.0,
        "milk_yield": 25.0,
        "milk_conductivity": 4.0,
        "battery_level": 95,
        "signal_strength": 95
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data

    prediction = data["prediction"]
    assert "disease" in prediction
    assert "confidence" in prediction
    assert "risk_level" in prediction
    assert "recommendation" in prediction
    assert "contributing_factors" in prediction
    assert "model_version" in prediction

    assert 0 <= prediction["confidence"] <= 100
    assert prediction["risk_level"] in ["low", "medium", "high"]


def test_predict_endpoint_sick_cattle(client):
    """Test prediction endpoint with sick cattle."""
    payload = {
        "cattle_id": "COW_TEST_002",
        "temperature": 40.2,
        "heart_rate": 95,
        "respiratory_rate": 38,
        "steps": 800,
        "activity_level": "low",
        "activity_duration_minutes": 120,
        "rumination_duration": 200,
        "rumination_episodes": 3,
        "feeding_efficiency": 55.0,
        "milk_yield": 10.0,
        "milk_conductivity": 5.2,
        "battery_level": 60,
        "signal_strength": 80
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    prediction = data["prediction"]
    assert prediction["disease"] in ["fever", "heat_stress", "mastitis", "digestive_disorder"]


def test_predict_endpoint_missing_fields(client):
    """Test prediction endpoint with missing required fields."""
    payload = {
        "cattle_id": "COW_TEST_003",
        "temperature": 38.5,
        # Missing other required fields
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_activity_level(client):
    """Test prediction with invalid activity level."""
    payload = {
        "cattle_id": "COW_TEST_004",
        "temperature": 38.5,
        "heart_rate": 70,
        "respiratory_rate": 25,
        "steps": 3500,
        "activity_level": "invalid_level",  # Invalid
        "activity_duration_minutes": 450,
        "rumination_duration": 500,
        "rumination_episodes": 10,
        "feeding_efficiency": 90.0,
        "milk_yield": 25.0,
        "milk_conductivity": 4.0,
        "battery_level": 95,
        "signal_strength": 95
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_stats_endpoint(client):
    """Test statistics endpoint."""
    # Make a prediction first
    payload = {
        "cattle_id": "COW_STATS_001",
        "temperature": 38.5,
        "heart_rate": 70,
        "respiratory_rate": 25,
        "steps": 3500,
        "activity_level": "normal",
        "activity_duration_minutes": 450,
        "rumination_duration": 500,
        "rumination_episodes": 10,
        "feeding_efficiency": 90.0,
        "milk_yield": 25.0,
        "milk_conductivity": 4.0,
        "battery_level": 95,
        "signal_strength": 95
    }
    client.post("/predict", json=payload)

    # Get stats
    response = client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert "total_predictions" in data
    assert "disease_distribution" in data


def test_recent_predictions_endpoint(client):
    """Test recent predictions endpoint."""
    response = client.get("/recent?limit=5")
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_prediction_response_schema(client):
    """Test prediction response matches schema."""
    payload = {
        "cattle_id": "COW_SCHEMA_001",
        "temperature": 38.5,
        "heart_rate": 70,
        "respiratory_rate": 25,
        "steps": 3500,
        "activity_level": "normal",
        "activity_duration_minutes": 450,
        "rumination_duration": 500,
        "rumination_episodes": 10,
        "feeding_efficiency": 90.0,
        "milk_yield": 25.0,
        "milk_conductivity": 4.0,
        "battery_level": 95,
        "signal_strength": 95
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    prediction = data["prediction"]
    assert isinstance(prediction["contributing_factors"], list)
    assert len(prediction["contributing_factors"]) > 0
    assert all(isinstance(f, str) for f in prediction["contributing_factors"])
