"""
Training utilities.
"""
import numpy as np


def directional_accuracy(y_true, y_pred):
    """
    The directionary accuracy metric.
    Identity {sign(target) == sign(prediction)}
    """
    assert all(y_true.shape == y_pred.shape)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def mape(y_true, y_pred):
    """
    Mean absolute percentage error.
    """
    assert all(y_true.shape == y_pred.shape)
    base = np.abs(y_true)
    base[base == 0.0] = np.inf  # To avoid infinite.
    err = np.abs(y_pred - y_true)
    return np.mean(err / base) * 100


def mse(y_true, y_pred):
    """
    Mean squared error.
    """
    assert all(y_true.shape == y_pred.shape)
    return np.mean((y_true - y_pred) ** 2)
