import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.config import PREPROCESSOR_PATH


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # activity_level encoder

        self.feature_names = None
        self.categorical_features = ["activity_level"]
        self.numeric_features = None

    def fit(self, X: pd.DataFrame):
        """Fit preprocessing objects."""
        self.feature_names = X.columns.tolist()

        # Fit categorical encoder
        self.label_encoder.fit(X["activity_level"])

        # Identify numeric columns
        self.numeric_features = [
            col for col in self.feature_names
            if col not in self.categorical_features
        ]

        # Fit scaler
        self.scaler.fit(X[self.numeric_features])

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features."""
        X_copy = X.copy()

        # Encode categorical columns
        for feature in self.categorical_features:
            X_copy[feature] = self.label_encoder.transform(
                X_copy[feature]
            )

        # Scale numeric columns
        X_scaled = self.scaler.transform(
            X_copy[self.numeric_features]
        )

        X_transformed = pd.DataFrame(
            X_scaled,
            columns=self.numeric_features
        )

        # Add encoded categorical columns
        for feature in self.categorical_features:
            X_transformed[feature] = X_copy[feature].values

        return X_transformed[self.feature_names].values

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)

    def save(self, path: str = None):
        """Save preprocessor."""
        if path is None:
            path = str(PREPROCESSOR_PATH)

        PREPROCESSOR_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(self, path)
        print(f"Preprocessor saved: {path}")

    @staticmethod
    def load(path: str = None):
        """Load preprocessor."""
        if path is None:
            path = str(PREPROCESSOR_PATH)

        return joblib.load(path)


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess dataset.

    Returns:
        X_processed
        y_encoded
        preprocessor
        target_encoder
    """

    # Features
    X = df.drop("disease", axis=1)

    # Target
    y = df["disease"]

    # Feature preprocessing
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(X)

    # Disease label encoding
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    print("\nDisease Label Mapping:")
    for idx, disease in enumerate(target_encoder.classes_):
        print(f"{idx} -> {disease}")

    return (
        X_processed,
        y_encoded,
        preprocessor,
        target_encoder
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/c/Projects/mlmodel")

    from src.data_generation import generate_dataset

    df = generate_dataset()

    (
        X,
        y,
        preprocessor,
        target_encoder
    ) = preprocess_data(df)

    preprocessor.save()

    joblib.dump(
        target_encoder,
        PREPROCESSOR_PATH.parent / "target_encoder.pkl"
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("Preprocessing complete.")