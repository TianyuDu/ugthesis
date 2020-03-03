"""
Training utilities.
"""
import numpy as np


def directional_accuracy(y_true, y_pred):
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))
