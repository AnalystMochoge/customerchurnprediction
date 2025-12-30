#This script is meant to load raw data from disk(or any other 
#future sources like APIs/DBs)

import pandas as pd
from pathlib import Path

def load_engineered_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load engineered churn dataset from CSV.
    """
    return pd.read_csv(file_path)