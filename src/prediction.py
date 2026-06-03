import sys
sys.path.insert(0, '/c/Projects/mlmodel')

import pandas as pd
import joblib
from src.config import DISEASE_LABELS, MODEL_PATH, PREPROCESSOR_PATH, DISEASE_RECOMMENDATIONS
from src.explainability import get_contributing_factors
from src.schemas import DiseaseInfo

class PredictionEngine:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()

    def load_model(self):
        """Load trained model and preprocessor."""
        try:
            self.model = joblib.load(str(MODEL_PATH))
            self.preprocessor = joblib.load(str(PREPROCESSOR_PATH))
        except FileNotFoundError as e:
            raise RuntimeError(f"Model or preprocessor not found. Train the model first. Error: {e}")

    def predict(self, input_data: dict) -> DiseaseInfo:
        """Make prediction for a asingle cattle."""

        # Create DataFrame for preprocessing
        df = pd.DataFrame([input_data])

        # Transform using preprocessor
        X_transformed = self.preprocessor.transform(df)

        # Get prediction and confidence
        prediction_idx = self.model.predict(X_transformed)[0]
        print("Raw Prediction:", prediction_idx)
        print("Type:", type(prediction_idx))
        prediction = DISEASE_LABELS.get(
            int(prediction_idx),
            "healthy"
        )
        print("Disease:", prediction)
        probabilities = self.model.predict_proba(X_transformed)[0]
        confidence = max(probabilities) * 100

        # Determine risk level
        risk_level = self._get_risk_level(confidence)

        # Get contributing factors
        contributing_factors = get_contributing_factors(input_data, self.model, self.preprocessor)

        # Get recommendation
        recommendation = DISEASE_RECOMMENDATIONS.get(prediction, "Monitor cattle health closely.")

        return DiseaseInfo(
            disease=prediction,
            confidence=round(confidence, 2),
            risk_level=risk_level,
            recommendation=recommendation,
            contributing_factors=contributing_factors,
            model_version="v1.0"
        )

    @staticmethod
    def _get_risk_level(confidence: float) -> str:
        """Determine risk level based on confidence."""
        if confidence < 40:
            return "low"
        elif confidence < 70:
            return "medium"
        else:
            return "high"


# Global prediction engine instance
_engine = None


def get_prediction_engine() -> PredictionEngine:
    """Get or create prediction engine singleton."""
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine


def predict(input_data: dict) -> DiseaseInfo:
    """Convenience function for making predictions."""
    engine = get_prediction_engine()
    return engine.predict(input_data)


if __name__ == "__main__":
    test_input = {
        "cattle_id": "COW_001",
        "temperature": 39.8,
        "heart_rate": 92,
        "respiratory_rate": 32,
        "steps": 1200,
        "activity_level": "low",
        "activity_duration_minutes": 180,
        "rumination_duration": 240,
        "rumination_episodes": 4,
        "feeding_efficiency": 72.5,
        "milk_yield": 18.2,
        "milk_conductivity": 5.1,
        "battery_level": 84,
        "signal_strength": 90
    }

    result = predict(test_input)
    print(f"Prediction: {result}")
