#This script creates new variables that captures business signal


import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    It handles feature engineering only, No scaling, No encoding, 
    Deterministic transfomrations
    """
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0,30,40,50,60,120],
        labels=['18-30','31-40','41-50','51-60','60+'], include_lowest=True
    )
    df['inactive_single_product'] = (
        (df['active_member']==0) & 
        (df['products_number']==1)
    ).astype(int)

    df['products_per_tenure'] = df['products_number'] / (df['tenure'] + 1)
    df['zero_balance'] = (df['balance'] == 0).astype(int)
    df['high_balance'] = (df['balance'] > df['balance'].mean()).astype(int)

    df['balance_per_product'] = df['balance'] / (df['products_number'] + 1)

    df['credit_score_band'] = pd.cut(
        df['credit_score'],
        bins = [300,580,670, 740, 800, 900],
        labels = ['poor','fair','good','very good','excellent']
    )

    df['early_customer'] = (df['tenure'] <3).astype(int)

    #Composite risk Indicator
    df['churn_risk_score'] = (
        df['inactive_single_product'] +
        df['zero_balance'] +
        df['early_customer']
    )

    return df


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(numerical_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
    categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
]) 
    return ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numerical_features),
    ('cat',categorical_pipeline, categorical_features)
],remainder='passthrough'

)