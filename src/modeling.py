# The main idea in this script is to train and save models

"""
Train-test split
Build pipelines
Train models
Persist best model
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def build_logisitc_regression(preprocessor):
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ("classifier", LogisticRegression(
                                        solver = 'saga',
                                        class_weight = 'balanced',
                                        max_iter = 10000,
                                        random_state = 42 
            ))
        ]
    )

def build_random_forest(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                                    n_estimators = 300,
                                    max_depth = 8,
                                    min_samples_leaf = 50,
                                    class_weight = 'balanced',
                                    random_state = 42,
                                    n_jobs = -1
            ))
        ]
    )

def build_gradient_boosting(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                                        n_estimators = 200,
                                        learning_rate = 0.05,
                                        max_depth = 3,
                                        random_state = 42
            ))
        ]
    )