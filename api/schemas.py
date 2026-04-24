""""
Pydantic schemas for API request and response validation.

"""

from pydantic import BaseModel
from typing import List, Optional

class RecommendationRequest(BaseModel):
    """Schema for recommendation request."""
    user_id: int
    n : int = 10

class MovieRecommendation(BaseModel):
    """Schema for a single movie recommendation."""
    movie_id: int
    title: str
    predicted_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    """Schema for recommendation response."""
    user_id: int
    recommendations: List[MovieRecommendation]