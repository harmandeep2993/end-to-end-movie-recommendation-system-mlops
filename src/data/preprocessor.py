# src/data/preprocessor.py

"""Preprocessing module for MovieLens 1M dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import get_logger

logger = get_logger(__name__)

# Missing functions for preprocessing steps
def _get_missing_values(dataframe: pd.DataFrame) -> tuple:
    """
    Calculate missing values in each column.

    Args:
        dataframe: input dataframe
    Returns:
        tuple: missing value counts and percentages
    """
    missing_values = dataframe.isnull().sum().sum()
    missing_percentage = (missing_values / len(dataframe)) * 100

    logger.info(f"Missing values: {missing_values}, Missing percentage: {missing_percentage:.2f}%")
    return missing_values, missing_percentage


# Remove duplicates
def _remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Args:
        dataframe: input dataframe
    Returns:
        dataframe with duplicates removed
    """
    duplicates = dataframe.duplicated().sum()
  
    logger.info(f"Duplicated rows: {duplicates}")
    return dataframe.drop_duplicates()


# Filter movies with low ratings
def filter_movies(ratings: pd.DataFrame, min_ratings: int = 10) -> pd.DataFrame:
    """
    Filter out movies with fewer than min_ratings ratings.
    
    Args:
        ratings: ratings dataframe
        min_ratings: minimum number of ratings required
    Returns:
        filtered ratings dataframe
    """
    logger.info(f"Movies before filtering: {ratings['movie_id'].nunique()}")

    # Filter logic: count ratings per movie and keep only those with enough ratings
    movie_counts = ratings.groupby("movie_id")["rating"].count()
    valid_movies = movie_counts[movie_counts >= min_ratings].index
    
    # Log the number of movies removed due to low ratings
    filtered = ratings[ratings["movie_id"].isin(valid_movies)]
    logger.info(f"Movies after filtering: {filtered['movie_id'].nunique()}")

    movies_removed = ratings["movie_id"].nunique() - filtered["movie_id"].nunique()
    logger.info(f"Movies removed: {movies_removed}")

    return filtered


# Train/test split for ratings
def train_test_split_ratings(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split ratings into train and test sets.

    Args:
        ratings: filtered ratings dataframe
        test_size: proportion for testing
        random_state: random seed
    Returns:
        tuple: train and test dataframes
    """
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings["user_id"]     # stratify by user_id to ensure all users are represented in both sets
    )

    logger.info(f"Train size: {train.shape}")
    logger.info(f"Test size: {test.shape}")

    return train, test


# Main preprocessing pipeline
def preprocess_pipeline(ratings: pd.DataFrame,movies: pd.DataFrame, users: pd.DataFrame) -> tuple:
    """
    Run all preprocessing steps.
    
    Args:
        ratings: raw ratings dataframe
        movies: raw movies dataframe
        users: raw users dataframe
    Returns:
        tuple: train, test, movies, users
    """
    # check missing values for all
    logger.info("Checking Missing Valuses...")
    _get_missing_values(ratings)
    _get_missing_values(movies)
    _get_missing_values(users)
    
    # remove duplicates from all
    logger.info("Duplicate Check and Removal...")
    ratings = _remove_duplicates(ratings)
    movies  = _remove_duplicates(movies)
    users   = _remove_duplicates(users)
    
    # filter low activity movies from ratings
    logger.info("Filtering low activity movies...")
    ratings = filter_movies(ratings)
    
    # train test split on ratings
    logger.info("Splitting ratings into train and test sets...")
    train, test = train_test_split_ratings(ratings)
    
    logger.info("Preprocessing pipeline completed!")
    
    return train, test, movies, users