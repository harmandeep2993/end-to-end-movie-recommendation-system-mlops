"""
FastAPI application for movie recommendations system.

"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import RecommendationRequest, RecommendationResponse, MovieRecommendation
from api.services import get_recommendations
from src.data import load_dataset
from src.utils import get_logger

logger = get_logger(__name__)

# load movies dataset once at startup
datasets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load datasets at startup."""
    logger.info("Loading datasets at startup...")
    data = load_dataset()
    datasets["movies"] = data["movies"]
    logger.info("Datasets loaded successfully")
    yield
    logger.info("Shutting down API...")
    datasets.clear()

app =FastAPI(
    title="Movie Recommendation API",
    description="API for movie recommendations",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """Get movie recommendations for a user."""
    movies = datasets.get("movies")
    if movies is None:
        logger.error("Movies dataset not loaded")
        raise HTTPException(status_code=500, detail="Movies dataset not loaded")

    recommendations = get_recommendations(
        user_id=request.user_id,
        movies=movies,
        n=request.n
    )
    if not recommendations:
        logger.warning("No recommendations found for the user.")
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
             
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=[
            MovieRecommendation(**r) for r in recommendations
        ]
    )