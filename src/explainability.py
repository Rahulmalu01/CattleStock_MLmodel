import joblib
from src.config import MODEL_PATH, PREPROCESSOR_PATH


def get_feature_importance(model, feature_names: list, top_n: int = 5) -> list:
    """Extract top N important features from the model."""

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return feature_names[:top_n]

    # Create feature-importance pairs
    feature_importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top features
    top_features = [feat for feat, _ in feature_importance_pairs[:top_n]]
    return top_features


def get_contributing_factors(prediction_input: dict, model, preprocessor) -> list:
    """Get human-readable contributing factors based on model prediction."""
    feature_names = preprocessor.feature_names
    top_features = get_feature_importance(model, feature_names, top_n=5)

    factor_descriptions = {
        "temperature": "Temperature level",
        "heart_rate": "Heart rate",
        "respiratory_rate": "Respiratory rate",
        "steps": "Activity (steps)",
        "activity_level": "Activity level",
        "activity_duration_minutes": "Activity duration",
        "rumination_duration": "Rumination duration",
        "rumination_episodes": "Rumination episodes",
        "feeding_efficiency": "Feeding efficiency",
        "milk_yield": "Milk yield",
        "milk_conductivity": "Milk conductivity",
        "battery_level": "Sensor battery",
        "signal_strength": "Signal strength"
    }

    factors = []
    for feature in top_features:
        description = factor_descriptions.get(feature, feature)
        value = prediction_input.get(feature, "unknown")

        # Add context about the value
        if feature == "temperature" and value > 39.5:
            factors.append("High temperature")
        elif feature == "heart_rate" and value > 85:
            factors.append("Elevated heart rate")
        elif feature == "respiratory_rate" and value > 30:
            factors.append("Elevated respiratory rate")
        elif feature == "activity_level" and value == "low":
            factors.append("Low activity level")
        elif feature == "feeding_efficiency" and value < 70:
            factors.append("Decreased feeding efficiency")
        elif feature == "milk_conductivity" and value > 4.8:
            factors.append("High milk conductivity")
        else:
            factors.append(description)

    return factors[:5]


if __name__ == "__main__":
    model = joblib.load(str(MODEL_PATH))
    preprocessor = joblib.load(str(PREPROCESSOR_PATH))
    features = get_feature_importance(model, preprocessor.feature_names)
    print(f"Top features: {features}")
