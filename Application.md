# MLModel

Brief machine learning project for data generation, preprocessing, training, prediction, and evaluation.

## Requirements
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it and install packages: `pip install -r requirements.txt`

## Usage
- Run the API: `python main.py api`
- Train models: `python main.py train`
- Make predictions: see `src/prediction.py`

## Execution
Follow these steps to run the project locally on Windows (PowerShell):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Prepare data
- Place raw data under `data/raw/` or run `src/data_generation.py` if you use scripted generation:

```powershell
python -m src.data_generation
```

4. Train the model

```powershell
python main.py train
```

5. Run the API / application

```powershell
python main.py api
```

6. Run tests

```powershell
pytest
```

Optional: run with Docker

```powershell
docker-compose up --build
# or for detached mode:
docker-compose up -d --build
```


## Testing
- Run tests with: `pytest`

## Project structure
- `main.py` — entrypoint
- `src/` — core modules (api, training, prediction, preprocessing, evaluation, explainability)
- `data/` — data files (raw/)
- `models/` — trained model artifacts
- `logs/` — runtime logs

## Notes
Keep sensitive keys out of the repo; use environment variables or a `.env` file.
