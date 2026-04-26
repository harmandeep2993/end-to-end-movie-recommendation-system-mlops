# CinePick 🎬

> Your personal AI-powered movie recommendation engine

CinePick is an end-to-end machine learning system that delivers personalized movie recommendations using SVD collaborative filtering, trained on the MovieLens 1M dataset. Built with MLOps best practices from data versioning and experiment tracking to containerized cloud deployment.

---

## Live Demo

API: `http://your-ecs-url:8000/docs`

---

## Features

- Personalized recommendations using SVD matrix factorization
- RMSE of 0.965 on held-out test set (199,708 ratings)
- REST API built with FastAPI and Pydantic validation
- Dark Netflix-style UI with real movie posters via TMDB API
- User watch history with color-coded ratings
- Experiment tracking with MLflow (6 models compared)
- Data versioning with DVC
- Containerized with Docker
- CI/CD via GitHub Actions
- Deployed on AWS ECS Fargate

---

## Architecture

```
MovieLens 1M
      ↓
Data Loading → Preprocessing → Feature Engineering
      ↓
User-Item Matrix (6040 × 3260, 95.94% sparse)
      ↓
Rating Normalization (subtract user mean)
      ↓
SVD Matrix Factorization (50 latent factors)
      ↓
FastAPI REST API
      ↓
CinePick Streamlit UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | SVD Matrix Factorization (SciPy) |
| Baseline Model | ItemKNN (scikit-learn) |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI + Pydantic |
| Frontend | Streamlit + TMDB API |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS ECS Fargate + ECR + IAM |
| Testing | pytest (15 tests) |
| Language | Python 3.12 |
| Package Manager | uv |

---

## Project Structure

```
movie-recommendation-mlops-end-to-end/
│
├── notebooks/
│   ├── 01_EDA.ipynb                  → data exploration
│   ├── 02_preprocessing.ipynb        → data cleaning and splitting
│   └── 03_model_training.ipynb       → model experiments
│
├── src/
│   ├── data/
│   │   ├── loader.py                 → load MovieLens .dat files
│   │   └── preprocessor.py          → filter, clean, split ratings
│   │
│   ├── features/
│   │   └── build_features.py        → sparse matrix + normalization
│   │
│   └── models/
│       ├── train.py                 → ItemKNN + SVD training + MLflow
│       ├── evaluate.py              → RMSE evaluation
│       └── predict.py               → recommendation generation
│
├── api/
│   ├── main.py                      → FastAPI app + endpoints
│   ├── schemas.py                   → request/response validation
│   └── services.py                  → API business logic
│
├── frontend/
│   └── app.py                       → CinePick Streamlit UI
│
├── tests/
│   ├── test_data.py                 → data pipeline tests
│   ├── test_model.py                → model tests
│   └── test_api.py                  → API endpoint tests
│
├── data/
│   ├── raw/ml-1m/                   → MovieLens 1M dataset (DVC tracked)
│   └── processed/                   → matrices + mappings (DVC tracked)
│
├── models/                          → saved model artifacts (DVC tracked)
├── reports/                         → evaluation reports
├── .github/workflows/ci-cd.yml      → CI/CD pipeline
├── Dockerfile
├── main.py                          → training pipeline entry point
└── pyproject.toml
```

---

## Dataset

MovieLens 1M dataset from GroupLens Research:

| Stat | Value |
|---|---|
| Total ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,883 |
| Rating scale | 1 to 5 stars |
| Time period | 2000 — 2003 |
| Sparsity | 95.94% |

Key findings from EDA:
- Positive bias: users rate movies 4-5 stars on average
- Long tail problem: few movies dominate ratings
- No missing values or duplicates

---

## Model Performance

All experiments tracked and compared in MLflow:

| Model | Parameters | RMSE |
|---|---|---|
| ItemKNN | K=10, cosine | 2.72 |
| ItemKNN | K=20, cosine | 2.82 |
| ItemKNN | K=50, cosine | 2.88 |
| SVD | 50 factors | **0.965** ← production |
| SVD | 100 factors | 1.02 |
| SVD | 200 factors | 1.02 |

Final evaluation on full test set (199,708 ratings): **RMSE 0.965**

### Why SVD beats ItemKNN

- SVD finds hidden latent factors (user taste profiles + movie features)
- Rating normalization removes user bias (subtract user mean)
- Handles sparse data better than direct similarity comparison
- Predictions are instant (matrix lookup vs kneighbors loop)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| POST | `/recommendations` | Get personalized recommendations |
| GET | `/user/{id}/history` | Get user watch history |

### Example request

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n": 10}'
```

### Example response

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 3114,
      "title": "Toy Story 2 (1999)",
      "predicted_score": 4.26
    },
    {
      "movie_id": 364,
      "title": "Lion King, The (1994)",
      "predicted_score": 4.25
    }
  ]
}
```

---

## Getting Started

### Prerequisites

- Python 3.12
- uv package manager
- Docker (optional)
- TMDB API key (for movie posters)

### Installation

```bash
# clone the repo
git clone https://github.com/harmandeep2993/movie-recommendation-mlops-end-to-end.git
cd movie-recommendation-mlops-end-to-end

# install dependencies
uv sync

# set up environment variables
cp .env.example .env
# add your TMDB_API_KEY to .env
```

### Download dataset

Download MovieLens 1M from https://grouplens.org/datasets/movielens/1m/ and place in:

```
data/raw/ml-1m/
├── movies.dat
├── ratings.dat
└── users.dat
```

### Train the model

```bash
python main.py
```

This runs the full pipeline:
1. Load and preprocess data
2. Build sparse User-Item Matrix
3. Normalize ratings
4. Train ItemKNN and SVD models
5. Track experiments in MLflow
6. Save best model

### Run the API

```bash
uvicorn api.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

### Run the UI

```bash
streamlit run frontend/app.py
```

UI available at: `http://localhost:8501`

### Run with Docker

```bash
docker build -t cinepick .
docker run -p 8000:8000 cinepick
```

### View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow UI available at: `http://localhost:5000`

---

## MLOps Pipeline

### Data Versioning (DVC)

```bash
dvc add data/raw/ml-1m
dvc add data/processed
dvc push
```

### CI/CD (GitHub Actions)

Every push to main triggers:

```
push to GitHub
      ↓
run pytest tests (15 tests)
      ↓
build Docker image
      ↓
push to AWS ECR
      ↓
deploy to AWS ECS Fargate
      ↓
new version live
```

### AWS Infrastructure

| Service | Purpose |
|---|---|
| ECR | Docker image registry |
| ECS Fargate | Serverless container hosting |
| IAM | Access management for GitHub Actions |

---

## Tests

```bash
pytest tests/ -v
```

| Test File | Coverage |
|---|---|
| test_data.py | data loading, preprocessing pipeline |
| test_model.py | model loading, predictions, artifacts |
| test_api.py | health endpoint, recommendations, history |

Total: 15 tests, all passing.

---

## Roadmap

- Precompute recommendations for sub-100ms API response
- PostgreSQL database for storing user ratings
- Vector database (Qdrant) for semantic movie search
- Neural Collaborative Filtering model
- Model monitoring and automated retraining pipeline
- A/B testing framework for model comparison

---

## What I Learned

This project covers the full ML engineering lifecycle:

- Building sparse matrix representations for recommendation systems
- Implementing and comparing collaborative filtering algorithms
- MLOps practices: experiment tracking, data versioning, CI/CD
- Production API design with FastAPI
- Docker containerization and cloud deployment
- Writing clean modular Python code with proper testing

---

## Author

Harman — Data Science and AI/ML Engineer, Berlin

[GitHub](https://github.com/harmandeep2993) · [LinkedIn](https://linkedin.com/in/harmandeep2993)

---

## License

MIT