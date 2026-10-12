import sys
sys.path.insert(0, '/c/Projects/mlmodel')

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.config import API_TITLE, API_VERSION, MODEL_PATH, DATASET_PATH
from src.schemas import (
    PredictionRequest, PredictionResponse, HealthCheckResponse, ModelInfoResponse
)
from src.prediction import get_prediction_engine
from src.database import get_database
from src.evaluation import load_metadata
from src.data_generation import generate_dataset
import json

@asynccontextmanager
async def lifespan(app):
    initialize_app()
    yield

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Livestock disease prediction service for CattleCare platform",
    lifespan=lifespan
)

# Initialize components
prediction_engine = None
database = None
metadata = None


def initialize_app():
    """Initialize app components on startup."""
    global prediction_engine, database, metadata

    # Generate dataset if not exists
    if not DATASET_PATH.exists():
        print("Generating dataset on startup...")
        generate_dataset()

    # Initialize database
    database = get_database()

    # Load prediction engine
    try:
        prediction_engine = get_prediction_engine()
        metadata = load_metadata()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Please run training pipeline first.")


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Service status endpoint."""
    return {"status": "healthy"}


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make disease prediction for a cattle."""
    if prediction_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction engine not initialized. Run training pipeline first."
        )

    try:
        # Convert request to dict
        input_data = request.model_dump()
        cattle_id = input_data.pop("cattle_id")

        # Make prediction
        prediction = prediction_engine.predict(input_data)

        # Log to database
        database.log_prediction(
            cattle_id=cattle_id,
            disease=prediction.disease,
            confidence=prediction.confidence,
            risk_level=prediction.risk_level,
            input_features=input_data,
            recommendation=prediction.recommendation
        )

        return PredictionResponse(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information and metrics."""
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not available. Run training pipeline first."
        )

    return ModelInfoResponse(
        model_version=metadata.get("model_version", "v1.0"),
        accuracy=metadata.get("accuracy", 0.0),
        precision=metadata.get("precision", 0.0),
        recall=metadata.get("recall", 0.0),
        f1_score=metadata.get("f1_score", 0.0),
        features=metadata.get("features", [])
    )


@app.get("/stats")
async def get_statistics():
    """Get prediction statistics."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    stats = database.get_statistics()
    return stats


@app.get("/recent")
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions."""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    predictions = database.get_recent_predictions(limit)
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
