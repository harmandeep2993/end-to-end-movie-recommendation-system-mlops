# src/data/loader.py

"""Data loading module for MovieLens 1M dataset."""

import pandas as pd
from typing import Dict
from pathlib import Path

from src.utils import get_logger


logger = get_logger(__name__)

# Define the path to the raw data directory and the expected header names for each dataset
FILE_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "ml-1m"

# Mapping of dataset names to their respective column headers
HEADER_NAMES = {
    "users": ["user_id", "gender", "age", "occupation", "zip_code"],
    "movies": ["movie_id", "title", "genres"],
    "ratings": ["user_id", "movie_id", "rating", "timestamp"]
}

def load_dataset(folder: Path = FILE_PATH) -> Dict[str, pd.DataFrame]:
    """
        Load all .dat files and assign headers based on filename.

    Args:
        folder (Path): The folder containing the .dat files.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping dataset names to DataFrames.
    """
    datasets = {}

    for file in folder.iterdir():
        if file.suffix == ".dat":
            name = file.stem
            logger.info(f"Loading dataset: {name} from {file}")
   
            # Load the dataset with appropriate parameters for .dat files
            df = pd.read_csv(
                file,
                sep="::",
                engine="python",
                header=None,
                encoding="latin-1"
            )

            if name in HEADER_NAMES:
                df.columns = HEADER_NAMES[name]

            datasets[name] = df

    logger.info(f"Loaded datasets: {list(datasets.keys())}")
    logger.info(f"Dataset shapes: { {name: df.shape for name, df in datasets.items()} }")
    return datasets