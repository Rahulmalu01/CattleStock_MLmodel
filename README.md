# CattleCare ML Service

CattleCare is a Python machine-learning service that classifies cattle health
measurements into one of five labels and exposes predictions through a FastAPI
REST API. The repository includes a synthetic data generator, a training and
evaluation pipeline, saved model artifacts, prediction logging, and API tests.

> **Important:** This project is a software demonstration, not a veterinary
> diagnostic tool. Its built-in training data is synthetic and its disease
> labels are assigned by the generator. Model scores and recommendations have
> not been validated for clinical use. Do not use predictions to replace
> examination or advice from a qualified veterinarian.

## Contents

- [Technology stack](#technology-stack)
- [Machine-learning approach](#machine-learning-approach)
- [Project layout](#project-layout)
- [Requirements and installation](#requirements-and-installation)
- [Train the models](#train-the-models)
- [Run the API](#run-the-api)
- [API reference](#api-reference)
- [Run tests](#run-tests)
- [Artifacts and data](#artifacts-and-data)
- [Known limitations](#known-limitations)

## Technology stack

| Component | Technology | Role in this application |
| --- | --- | --- |
| Language | Python | CLI, data generation, training, inference, and API |
| API | FastAPI, Uvicorn | HTTP endpoints and ASGI server |
| Request validation | Pydantic v2 | Validates prediction payloads and describes API schemas |
| Data handling | pandas, NumPy | Tabular datasets and synthetic measurement generation |
| Machine learning | scikit-learn | Preprocessing, train/test split, classifiers, and evaluation metrics |
| Gradient boosting | XGBoost | One of the candidate multi-class classifiers |
| Persistence | Joblib, JSON | Serialized model/preprocessor/encoder and model metadata |
| Prediction history | SQLite | Stores prediction details and provides recent/statistical views |
| Tests | pytest, HTTPX | Test runner and HTTP client support for API tests |
| Containerization | Docker, Docker Compose | Optional container build and API service configuration |

Pinned Python dependencies are listed in [`requirements.txt`](requirements.txt).
The supplied Dockerfile uses Python 3.12; using Python 3.12 locally is a
reasonable choice for matching that image.

## Machine-learning approach

### Prediction target and input features

The classifier predicts one of these labels:

- `healthy`
- `fever`
- `mastitis`
- `heat_stress`
- `digestive_disorder`

The 13 model input features are:

| Feature | Meaning |
| --- | --- |
| `temperature` | Body temperature in Celsius |
| `heart_rate` | Heart rate in beats per minute |
| `respiratory_rate` | Respiratory rate per minute |
| `steps` | Number of steps |
| `activity_level` | `normal`, `low`, or `high` |
| `activity_duration_minutes` | Duration of activity |
| `rumination_duration` | Rumination duration |
| `rumination_episodes` | Number of rumination episodes |
| `feeding_efficiency` | Feeding efficiency percentage |
| `milk_yield` | Milk yield in liters |
| `milk_conductivity` | Milk conductivity measurement |
| `battery_level` | Sensor battery level |
| `signal_strength` | Sensor signal strength |

`cattle_id` is required by the API for identifying/logging an animal, but is
removed from the feature data before inference; the model does not use it.

### Data generation

When `data/dataset.csv` is absent, the training pipeline generates a balanced
synthetic dataset with 10,000 rows: 2,000 generated samples per disease label.
The generator samples feature values from class-specific ranges and assigns the
corresponding class label. It shuffles the rows before saving the CSV.

If `data/dataset.csv` already exists, training loads that file instead of
regenerating it. The file must contain all 13 feature columns and a `disease`
target column with compatible labels.

### Preprocessing

The preprocessing code applies:

1. `StandardScaler` to the numeric measurements.
2. `LabelEncoder` to the `activity_level` categorical feature.
3. A separate `LabelEncoder` to convert disease labels to model target values.

The fitted preprocessor and target encoder are persisted and reused by the
prediction engine so API requests follow the same transformations as training.

### Candidate classifiers and selection

The training pipeline fits and evaluates three classifiers:

| Classifier | Configuration used in this project |
| --- | --- |
| Random Forest | 100 trees, maximum depth 15, configured random state 42, parallel jobs enabled |
| XGBoost | Multi-class probability objective (`multi:softprob`), 5 classes, 100 estimators, depth 7, learning rate 0.1, configured random state 42, `mlogloss` evaluation metric |
| Gradient Boosting | 100 estimators, maximum depth 7, learning rate 0.1, configured random state 42 |

The dataset is split into 80% training and 20% test data using a stratified
split and random state 42. Each candidate is scored on the test split. The
pipeline selects the model with the highest **accuracy** and saves that model.
It also reports weighted precision, weighted recall, weighted F1 score, and a
confusion matrix for each candidate. Metrics and the selected model name are
written to `models/model_metadata.json`.

### Prediction confidence and contributing factors

Inference gets class probabilities from the selected model. The largest
probability, expressed as a percentage, is returned as `confidence`. The
service maps that confidence to a `risk_level`: below 40 is `low`, below 70 is
`medium`, and 70 or above is `high`. This is a confidence-based software
threshold, not a medically calibrated measure of disease severity.

The response also contains a recommendation selected from a fixed
disease-to-text mapping and up to five contributing factors. The factors are
based on the selected model's global feature importances and a small set of
hard-coded feature thresholds; they are not per-prediction causal explanations.

## Project layout

```text
.
├── main.py                  # CLI entry point: train and api commands
├── src/
│   ├── api.py               # FastAPI application and HTTP endpoints
│   ├── config.py            # Paths, labels, model settings, recommendations
│   ├── data_generation.py   # Synthetic labeled data generator
│   ├── database.py          # SQLite prediction history
│   ├── evaluation.py        # Metrics and model metadata
│   ├── explainability.py    # Feature-importance-based factors
│   ├── prediction.py        # Artifact loading and inference
│   ├── preprocessing.py    # Feature and target encoding/scaling
│   ├── schemas.py           # Pydantic API request/response models
│   └── training.py          # Candidate training and model selection
├── tests/                   # API, model, preprocessing, and data tests
├── data/                    # Dataset CSV (created when needed)
├── models/                  # Serialized model and metadata
├── logs/                    # SQLite prediction database
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Requirements and installation

From the project directory, create and activate a virtual environment in
Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the models

Run the full training pipeline:

```powershell
python main.py train
```

The pipeline generates `data/dataset.csv` if it does not exist. It then
preprocesses the data, fits and evaluates the three classifiers, selects the
highest-accuracy model, and writes these artifacts:

- `models/model.joblib` — selected classifier
- `models/preprocessor.joblib` — fitted feature transformations
- `models/target_encoder.pkl` — disease label encoder
- `models/model_metadata.json` — selected model name and evaluation metrics

The API needs these model artifacts. Starting the API may generate the
synthetic dataset if it is missing, but it does **not** train the models; run
the training command first.

## Run the API

Start the service on the default host and port:

```powershell
python main.py api
```

The default bind address is `0.0.0.0:9000`. For local-only development, use
`127.0.0.1` and optionally enable Uvicorn's auto-reload:

```powershell
python main.py api --host 127.0.0.1 --port 9000 --reload
```

Interactive API documentation is available at
<http://127.0.0.1:9000/docs> when the service is running. The service needs
access to `data/`, `models/`, and `logs/`; those directories are created as
needed in the project configuration.

### Docker Compose

The repository includes a Compose service that publishes port 9000 and mounts
the data, models, and logs directories. Build and start it with:

```powershell
docker compose up --build
```

The model artifacts must be available under `models/` before the API can make
predictions. Stop the foreground Compose service with `Ctrl+C`; remove the
containers with `docker compose down`.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic service status |
| `GET` | `/health` | Health-check response |
| `POST` | `/predict` | Predict a disease label for the supplied measurements |
| `GET` | `/model-info` | Return the saved model version, metrics, and feature names |
| `GET` | `/stats` | Return prediction count and disease distribution |
| `GET` | `/recent?limit=10` | Return recent logged predictions |

Example prediction request:

```json
{
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
```

`battery_level` and `signal_strength` default to 100 if omitted. The API
validates `activity_level` as `normal`, `low`, or `high`. Successful prediction
responses include the predicted disease, confidence percentage, risk level,
recommendation, contributing factors, and model version. Each successful
prediction is recorded in `logs/predictions.db`.

## Run tests

Install the project requirements, then run:

```powershell
pytest
```

To run one test module:

```powershell
pytest tests/test_api.py -q
```

## Artifacts and data

The default locations are configured in `src/config.py`:

- `data/dataset.csv` — generated or supplied training data
- `models/model.joblib` — selected model
- `models/preprocessor.joblib` — fitted feature preprocessor
- `models/target_encoder.pkl` — target label encoder
- `models/model_metadata.json` — model information and test metrics
- `logs/predictions.db` — prediction history

The serialized Joblib/pickle artifacts should only be loaded from trusted
sources. Retraining replaces the saved model artifacts and metadata.

## Known limitations

- The default dataset is synthetic, generated from predefined ranges, and
  should not be interpreted as representative of real cattle or disease.
- The current preprocessing pipeline fits its scaler and encoders before the
  train/test split. This allows preprocessing to see test-set feature
  statistics, so the reported test metrics may be optimistic. For rigorous
  evaluation, fit preprocessing only on training data (ideally in a
  scikit-learn pipeline) and evaluate on an untouched dataset.
- The selected model is chosen using accuracy alone; reported metrics do not
  establish clinical reliability or calibration.
- Returned confidence and risk level are not calibrated clinical probabilities
  or severity scores.
- Recommendations are fixed informational strings, not individualized
  treatment advice.
