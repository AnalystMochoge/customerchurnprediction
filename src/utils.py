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