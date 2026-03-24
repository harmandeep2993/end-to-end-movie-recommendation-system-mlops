# src/features/build_features.py

"""
This module contains functions to build features for the movie recommendation system.

Functions:
- build_user_features: Builds user features from the users DataFrame.
- build_movie_features: Builds movie features from the movies DataFrame.
- build_interaction_features: Builds interaction features from the ratings DataFrame.
"""

from pathlib import Path

import pandas as pd

from src.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

def build_user_item_matrix(train: pd.DataFrame) -> tuple:
    """
    Build sparse User-Item Matrix from training data.
    
    Args:
        train: training dataframe
    
    Returns:
        tuple: sparse matrix, user_map, item_map
    """
    from scipy.sparse import csr_matrix

    # create mappings
    user_ids = train["user_id"].unique()
    movie_ids = train["movie_id"].unique()
    
    # create user and item maps
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    # build matrix
    row = train["user_id"].map(user_map)
    col = train["movie_id"].map(item_map)
    values = train["rating"].values

    user_item_matrix = csr_matrix(
        (values, (row, col)),
        shape=(len(user_map), len(item_map))
    )
    
    logger.info(f"Matrix shape: {user_item_matrix.shape}")
    logger.info(f"Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
    
    return user_item_matrix, user_map, item_map


# Note: This function is not called in the main pipeline, but can be used for future feature engineering steps.
def save_features(user_item_matrix, user_map: dict, item_map: dict) -> None:
    """
    Save sparse matrix and mappings to disk.
    
    Args:
        user_item_matrix: sparse user-item matrix
        user_map: mapping of user_id to row index
        item_map: mapping of movie_id to col index
    """
   
    from scipy.sparse import save_npz

    matrix_path = Path(__file__).parent.parent.parent / "data" / "processed"
    mappings_path = matrix_path / "mappings"

    matrix_path.mkdir(parents=True, exist_ok=True)
    mappings_path.mkdir(parents=True, exist_ok=True)

    # save sparse matrix
    save_npz(matrix_path / "user_item_matrix.npz", user_item_matrix)

    # save mappings as csv
    pd.DataFrame(user_map.items(), columns=["user_id", "user_idx"]).to_csv(
        mappings_path / "user_map.csv", index=False
    )
    pd.DataFrame(item_map.items(), columns=["movie_id", "item_idx"]).to_csv(
        mappings_path / "item_map.csv", index=False
    )

    logger.info(f"Saved matrix to {matrix_path}")
    logger.info(f"Saved mappings to {mappings_path}")

def build_features_pipeline(train: pd.DataFrame) -> tuple:
    """
    Run all feature building steps in order.
    
    Args:
        train: training dataframe
    Returns:
        tuple: user_item_matrix, user_map, item_map
    """
    # build matrix
    user_item_matrix, user_map, item_map = build_user_item_matrix(train)
    
    # save to disk
    save_features(user_item_matrix, user_map, item_map)
    
    logger.info("Feature pipeline completed successfully!.")
    
    return user_item_matrix, user_map, item_map