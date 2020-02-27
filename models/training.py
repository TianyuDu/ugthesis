"""
Training and model evaluation procedures.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn.model_selection import train_test_split


def check_data(data) -> None:
    """
    Validate dataset input.
    """
    raise NotImplementedError

def train(
    model,
    data: List[np.ndarray]
) -> dict():
    """
    Train and evaluate model.
    """
    # ==== Data Preprocessing ====
    X_train, y_train, X_test, y_test = data
    # Train validation split.
    # Note that the order of dataset returned by
    # train_test_split method is odd.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        train_size=0.75,
        shuffle=True
    )
    # ==== Training procedure ====
    raise NotImplementedError
