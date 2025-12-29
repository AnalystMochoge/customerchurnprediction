#This script evaluates trained models

"""
Compute metrics
Produce plots
No training logic
"""
from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(model,X_test, y_test):
    y_proba = model.predic_proba(X_test)[:, 1]
    auc =roc_auc_score(y_test, y_proba)

    return {
        "roc_auc": auc
    }