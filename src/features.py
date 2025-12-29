#This script creates new variables that captures business signal

"""
It handles feature engineering only, No scaling, No encoding, 
Deterministic transfomrations
"""
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[18,30,40,50,60,100],
        labels=['18-30','31-40','41-50','51-60','60+']
    )
    df['inactive_single_prooduct'] = (
        (df['active_member']==0) & 
        (df['product_number']==1)
    ).astype(int)

    df['products_per_tenure'] = df['products_number'] / (df['tenure'] + 1)
    df['zero_balance'] = (df['balance'] == 0).astype(int)
    df['high_balance'] = (df['balance'] > df['balance'].mean()).astype(int)

    df['balance_per_product'] = (df['balance'] / df['products_number'] + 1)

    df['credit_score_band'] = pd.cut(
        df['credit_score'],
        bins = [300,580,670, 740, 800, 900],
        labels = ['poor','Fair','Good','Very Good','Excellent']
    )

    df['early_customer'] = (df['tenure'] <3).astype(int)

    #Composite risk Indicator
    df['churn_risk_score'] = (
        df['inactive_single_prooduct'] +
        df['zero_balance'] +
        df['early_customer']
    )

    return df

    