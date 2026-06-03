import numpy as np
import pandas as pd
from src.config import DATASET_PATH, DATASET_SIZE, DISEASE_CLASSES


def generate_disease_samples(disease: str, count: int) -> pd.DataFrame:
    """Generate synthetic data for a specific disease class."""
    samples = []

    for _ in range(count):
        if disease == "healthy":
            sample = {
                "temperature": np.random.uniform(37.5, 39.0),
                "heart_rate": np.random.uniform(60, 80),
                "respiratory_rate": np.random.uniform(20, 30),
                "steps": np.random.uniform(2500, 5000),
                "activity_level": "normal",
                "activity_duration_minutes": np.random.uniform(360, 600),
                "rumination_duration": np.random.uniform(400, 600),
                "rumination_episodes": np.random.uniform(8, 12),
                "feeding_efficiency": np.random.uniform(80, 100),
                "milk_yield": np.random.uniform(20, 30),
                "milk_conductivity": np.random.uniform(3.5, 4.5),
                "battery_level": np.random.uniform(50, 100),
                "signal_strength": np.random.uniform(70, 100),
            }
        elif disease == "fever":
            sample = {
                "temperature": np.random.uniform(39.5, 41.0),
                "heart_rate": np.random.uniform(85, 110),
                "respiratory_rate": np.random.uniform(30, 45),
                "steps": np.random.uniform(500, 1500),
                "activity_level": "low",
                "activity_duration_minutes": np.random.uniform(60, 200),
                "rumination_duration": np.random.uniform(100, 300),
                "rumination_episodes": np.random.uniform(2, 6),
                "feeding_efficiency": np.random.uniform(40, 70),
                "milk_yield": np.random.uniform(8, 15),
                "milk_conductivity": np.random.uniform(3.8, 4.6),
                "battery_level": np.random.uniform(30, 90),
                "signal_strength": np.random.uniform(60, 95),
            }
        elif disease == "heat_stress":
            sample = {
                "temperature": np.random.uniform(39.2, 40.8),
                "heart_rate": np.random.uniform(90, 120),
                "respiratory_rate": np.random.uniform(35, 50),
                "steps": np.random.uniform(300, 1000),
                "activity_level": "low",
                "activity_duration_minutes": np.random.uniform(30, 150),
                "rumination_duration": np.random.uniform(80, 250),
                "rumination_episodes": np.random.uniform(1, 5),
                "feeding_efficiency": np.random.uniform(30, 60),
                "milk_yield": np.random.uniform(5, 12),
                "milk_conductivity": np.random.uniform(3.6, 4.4),
                "battery_level": np.random.uniform(20, 85),
                "signal_strength": np.random.uniform(50, 90),
            }
        elif disease == "mastitis":
            sample = {
                "temperature": np.random.uniform(38.8, 40.2),
                "heart_rate": np.random.uniform(70, 95),
                "respiratory_rate": np.random.uniform(25, 35),
                "steps": np.random.uniform(1000, 3000),
                "activity_level": "low",
                "activity_duration_minutes": np.random.uniform(200, 400),
                "rumination_duration": np.random.uniform(250, 450),
                "rumination_episodes": np.random.uniform(4, 8),
                "feeding_efficiency": np.random.uniform(60, 80),
                "milk_yield": np.random.uniform(5, 15),
                "milk_conductivity": np.random.uniform(4.8, 6.5),
                "battery_level": np.random.uniform(40, 95),
                "signal_strength": np.random.uniform(65, 100),
            }
        elif disease == "digestive_disorder":
            sample = {
                "temperature": np.random.uniform(37.8, 39.5),
                "heart_rate": np.random.uniform(65, 90),
                "respiratory_rate": np.random.uniform(22, 35),
                "steps": np.random.uniform(800, 2000),
                "activity_level": "low",
                "activity_duration_minutes": np.random.uniform(100, 300),
                "rumination_duration": np.random.uniform(100, 300),
                "rumination_episodes": np.random.uniform(2, 5),
                "feeding_efficiency": np.random.uniform(40, 70),
                "milk_yield": np.random.uniform(12, 22),
                "milk_conductivity": np.random.uniform(3.7, 4.5),
                "battery_level": np.random.uniform(35, 90),
                "signal_strength": np.random.uniform(60, 95),
            }

        sample["disease"] = disease
        samples.append(sample)

    return pd.DataFrame(samples)


def generate_dataset(output_path: str = None) -> pd.DataFrame:
    """Generate synthetic dataset with balanced disease distribution."""
    if output_path is None:
        output_path = str(DATASET_PATH)

    samples_per_disease = DATASET_SIZE // len(DISEASE_CLASSES)
    dfs = [generate_disease_samples(disease, samples_per_disease) for disease in DISEASE_CLASSES]

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path} ({len(df)} records)")

    return df


if __name__ == "__main__":
    generate_dataset()
