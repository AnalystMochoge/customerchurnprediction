#Shared helpers used across scripts

"""
Logging
Path handling
Metric utilities
Threshild selection helpers
"""

import numpy as np

def apply_threshold(y_proba, threshold=0.35):
    return (y_proba>=threshold).astype(int)


from pathlib import Path

def save_model_inputs(X_train, X_test, y_train, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / 'X_train.csv', index=False)
    X_test.to_csv(output_dir / 'X_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train', index=False)
    y_test.to_csv(output_dir / 'y_test', index=False)