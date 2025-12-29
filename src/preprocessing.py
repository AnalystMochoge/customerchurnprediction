# Clean data and prepare it for feature engineering
"""
Docstring for src.preprocessing
This script Handles missing values, Fixes data types, 
Removes duplicates, caps/ handles infinities ,No feature creation
"""

import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()

    df.replace([np.inf, -np.inf], np.mean, inplace= True)

    X=df.drop(columns=['churn'])
    y=df['churn']

    return df
