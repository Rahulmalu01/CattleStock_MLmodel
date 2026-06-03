from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from src.config import METADATA_PATH


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print(f"\n{model_name} Evaluation:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    return metrics


def save_metadata(model_name: str, metrics: dict, feature_names: list):
    """Save model metadata to JSON."""
    metadata = {
        "model_version": "v1.0",
        "model_type": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "features": feature_names
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(METADATA_PATH), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved: {METADATA_PATH}")
    return metadata


def load_metadata():
    """Load model metadata from JSON."""
    with open(str(METADATA_PATH), "r") as f:
        return json.load(f)
