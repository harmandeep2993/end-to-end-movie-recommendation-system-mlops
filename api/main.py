"""
FastAPI application for movie recommendations system.

"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import RecommendationRequest, RecommendationResponse, MovieRecommendation, HistoryResponse
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
    datasets["ratings"] = data["ratings"]
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

@app.get("/user/{user_id}/history", response_model=HistoryResponse)
def get_user_history(user_id: int):
    
    movies = datasets.get("movies")
    ratings = datasets.get("ratings")
    
    if movies is None or ratings is None:
        raise HTTPException(status_code=500, detail="Datasets not loaded")
    
    # get user ratings
    user_ratings = ratings[ratings["user_id"] == user_id]
    
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # get top rated movies
    top_rated = user_ratings.sort_values("rating", ascending=False).head(10)
    
    history = []
    for _, row in top_rated.iterrows():
        title = movies[movies["movie_id"] == row["movie_id"]]["title"].values
        if len(title) > 0:
            history.append({
                "movie_id": int(row["movie_id"]),
                "title": title[0],
                "rating": float(row["rating"])
            })
    
    return HistoryResponse(user_id=user_id, history=history)