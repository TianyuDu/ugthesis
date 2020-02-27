"""
Training and model evaluation procedures.
"""
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from data_feed import regression_feed


def check_data(data) -> None:
    """
    Validate dataset input.
    """
    raise NotImplementedError


def training_pipeline(
    model,
    data: List[np.ndarray],
    task: Union["regression", "classification"],
    num_fold: int = 1,
) -> dict():
    """
    Train and evaluate model.
    """
    # ==== Data Preprocessing ====
    X_train_raw, y_train_raw, X_test, y_test = data
    check_data(data)
    val_loss_lst = list()
    for _ in range(num_fold):
        # Train validation split.
        # Note that the order of dataset returned by
        # train_test_split method is odd.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_raw, y_train_raw,
            train_size=0.75,
            shuffle=True
        )
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
