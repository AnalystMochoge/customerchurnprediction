import pandas as pd

def drop_unused_columns(
        df: pd.DataFrame,
        columns_to_drop: list[str]
) -> pd.DataFrame:
    """
    Drop unused columns from dataframe.
    """
    return df.drop(columns=columns_to_drop)



def split_features_target(df, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y




from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size =test_size,
        random_state= random_state,
        stratify=y
    )

