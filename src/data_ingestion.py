#This script is meant to load raw data from disk(or any other 
#future sources like APIs/DBs)

import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=['customer_id', 'gender', 'country'], inplace=True)

    if 'Exited' not in df.columns:
        raise ValueError("Target column 'Exited' not found")
    return df
