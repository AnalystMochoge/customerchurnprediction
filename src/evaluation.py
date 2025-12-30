#This script evaluates trained models

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import pandas as pd

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }


def compare_models(results: dict):
    return(
        pd.DataFrame(results).T.sort_values(by="roc_auc", ascending=False)
    )