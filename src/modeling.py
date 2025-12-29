# The main idea in this script is to train and save models

"""
Train-test split
Build pipelines
Train models
Persist best model
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_model(X,y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                     test_size=0.2,
                     stratify=y,
                     random_state=42
            )
    
    model = GradientBoostingClassifier(
        n_estimators= 200,
        learning_rate= 0.05,
        random_state= 42
    )
    
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    gb_pipeline.fit(X_train, y_train)

    joblib.dump(gb_pipeline, "model\gb_churn_model.joblib")

    return gb_pipeline, X_test, y_test