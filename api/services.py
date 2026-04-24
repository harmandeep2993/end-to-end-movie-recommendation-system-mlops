"""
Services layer for the API application.
"""

import pandas
from src.models.predict import predict_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

def get_recommendations(user_id: int, movies: pandas.DataFrame, n: int = 10) -> list:
    """
    Get movie recommendations for a given user.

    Args:
        user_id: ID of the user to get recommendations for
        movies: movies dataframe
        n: number of recommendations to return

    Returns:
        List of recommended movies with predicted ratings
    """

    logger.info(f"Getting recommendations for user_id: {user_id}")

    recommendations = predict_pipeline(
        user_id= user_id,
        movies= movies,
        n= n
    )
    
    if recommendations is None:

        logger.warning(f"No recommendations found for user_id: {user_id}")
        return []
    
    return recommendations.to_dict(orient="records")