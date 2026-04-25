# src/features/build_features.py

"""
This module contains functions to build features for the movie recommendation system.

Functions:
- build_user_features: Builds user features from the users DataFrame.
- build_movie_features: Builds movie features from the movies DataFrame.
- build_interaction_features: Builds interaction features from the ratings DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

from src.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


def build_user_item_matrix(train: pd.DataFrame) -> tuple:
    """
    Build sparse User-Item Matrix from training data.
    
    Args:
        train: training dataframe (ratings train data)
    Returns:
        tuple: sparse matrix, user_map, item_map
    """

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
    
    # calculate sparsity
    sparsity = 1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])

    # log matrix info
    logger.info(f"Matrix shape: {user_item_matrix.shape}")
    logger.info(f"Sparsity: {sparsity:.4f}")
    
    return user_item_matrix, user_map, item_map


# Marix normalization function
def normalize_matrix(user_item_matrix: csr_matrix) -> tuple:
    """
    Normalize user ratings by subtracting user mean.

    Args:
        user_item_matrix: sparse user-item matrix
    Returns:
        tuple: normalized matrix, user means
    """
    # lil_matrix for efficient row operations, then convert back to csr_matrix
    from scipy.sparse import lil_matrix
    
    # initialize user means array
    user_means = np.zeros(user_item_matrix.shape[0]) 
    
    # calculate user means
    for i in range(user_item_matrix.shape[0]):
        user_ratings = user_item_matrix[i].toarray().flatten()
        rated = user_ratings[user_ratings > 0]
        if len(rated) > 0:
            user_means[i] = rated.mean()
    
    # convert to lil_matrix for efficient row operations
    normalized = lil_matrix(user_item_matrix.shape, dtype=float)
    
    for i in range(user_item_matrix.shape[0]):
        row = user_item_matrix[i].toarray().flatten()
        nonzero_idx = np.where(row > 0)[0]
        if len(nonzero_idx) > 0 and user_means[i] > 0:
            normalized[i, nonzero_idx] = row[nonzero_idx] - user_means[i]
    
    # convert back to csr_matrix for efficient computations later
    normalized = normalized.tocsr()
    
    return normalized, user_means


# Functions to save features to disk
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


# Function to save normalized matrix and user means
def save_normalized_matrix(normalized_matrix: csr_matrix, user_means: np.ndarray) -> None:
    """
    Save normalized matrix to disk.
    
    Args:
        normalized_matrix: normalized sparse user-item matrix
        user_means: array of user means
    """
    from scipy.sparse import save_npz

    matrix_path = Path(__file__).parent.parent.parent / "data" / "processed"
    matrix_path.mkdir(parents=True, exist_ok=True)

    save_npz(matrix_path / "normalized_matrix.npz", normalized_matrix)
    np.save(matrix_path / "user_means.npy", user_means)

    logger.info(f"Saved normalized matrix to {matrix_path}")
    logger.info(f"Saved user means to {matrix_path}")


# Main pipeline function to run all feature building steps
def build_features_pipeline(train: pd.DataFrame) -> tuple:
    """
    Run all feature building steps in order.
    
    Args:
        train: training dataframe
    Returns:
        tuple: user_item_matrix, user_map, item_map, user_means, normalized_matrix
    """
    logger.info("=== START: FEATURE BUILD PIPELINE ===")

    # build matrix
    logger.info("Building user-item matrix...")
    user_item_matrix, user_map, item_map = build_user_item_matrix(train)
    
    # normalize matrix
    logger.info("Normalizing matrix...")    
    normalized_matrix, user_means = normalize_matrix(user_item_matrix)
    
    # save to disk
    logger.info("Saving features to disk...")
    save_features(user_item_matrix, user_map, item_map)
    save_normalized_matrix(normalized_matrix, user_means)
    
    logger.info("=== END: FEATURE BUILD PIPELINE ===")
    
    return user_item_matrix, user_map, item_map, user_means, normalized_matrix