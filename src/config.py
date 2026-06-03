import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
DB_PATH = LOGS_DIR / "predictions.db"

DATASET_SIZE = 10000
TEST_SIZE = 0.2
RANDOM_STATE = 42

DISEASE_CLASSES = ["healthy", "fever", "mastitis", "heat_stress", "digestive_disorder"]

DISEASE_RECOMMENDATIONS = {
    "healthy": "Cattle appears healthy. Continue regular monitoring.",
    "fever": "High fever detected. Consult veterinarian immediately for diagnosis and treatment.",
    "mastitis": "Mastitis indicators detected. Check udder health and milk quality immediately.",
    "heat_stress": "Move cattle to shaded area and ensure continuous access to water.",
    "digestive_disorder": "Digestive issues detected. Review feed quality and consult with veterinarian."
}

API_TITLE = "CattleCare Disease Prediction API"
API_VERSION = "v1.0"
MODEL_VERSION = "v1.0"

DISEASE_LABELS = {
    0: "healthy",
    1: "fever",
    2: "mastitis",
    3: "heat_stress",
    4: "digestive_disorder"
}