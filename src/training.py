import sys
sys.path.insert(0, '/c/Projects/mlmodel')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

from src.config import (
    DATASET_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, DISEASE_CLASSES
)
from src.data_generation import generate_dataset
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model, save_metadata


def train_models(X, y):
    """Train Random Forest, XGBoost, and Gradient Boosting models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(DISEASE_CLASSES),
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = {"model": model, "metrics": metrics}

    return results, X_train, X_test, y_train, y_test


def select_best_model(results) -> tuple:
    """Select best model based on accuracy."""
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["accuracy"])
    print(f"\n🎯 Best Model: {best_name} (Accuracy: {results[best_name]['metrics']['accuracy']:.4f})")
    return best_name, results[best_name]["model"], results[best_name]["metrics"]


def train_pipeline():
    """Complete training pipeline."""
    print("=" * 50)
    print("ML TRAINING PIPELINE")
    print("=" * 50)

    # Generate dataset
    if not DATASET_PATH.exists():
        print("\nGenerating synthetic dataset...")
        df = generate_dataset()
    else:
        print(f"\nLoading existing dataset: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Disease distribution:\n{df['disease'].value_counts()}")

    # Preprocess data
    print("\nPreprocessing data...")
    X, y, preprocessor, target_encoder = preprocess_data(df)
    feature_names = preprocessor.feature_names

    print("\nClass Mapping:")
    for idx, cls in enumerate(target_encoder.classes_):
        print(f"{idx} -> {cls}")

    # Train models
    print("\nTraining models...")
    results, X_train, X_test, y_train, y_test = train_models(X, y)

    # Select best model
    best_name, best_model, best_metrics = select_best_model(results)

    # Save best model
    print("\nSaving model and metadata...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, str(MODEL_PATH))
    print(f"Model saved: {MODEL_PATH}")

    preprocessor.save()
    metadata = save_metadata(best_name, best_metrics, feature_names)
    joblib.dump(target_encoder, MODEL_PATH.parent / "target_encoder.pkl")
    print(f"Target encoder saved: " f"{MODEL_PATH.parent / 'target_encoder.pkl'}")
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE ✓")
    print("=" * 50)

    return best_model, preprocessor, metadata

if __name__ == "__main__":
    train_pipeline()
