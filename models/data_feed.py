"""
Date: Feb. 22, 2020.
Data loaders for most models.
"""
import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from typing import List, Tuple
from datetime import datetime

from sklearn import model_selection

from utils.time_series_utils import gen_dataset_calendarday

# ============ Configs ============
MASTER_DIR = "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis"
TARGET_COL = "RETURN"
LAG_DAYS = 28
DF = pd.read_csv(
    MASTER_DIR + "/data/ready_to_use/returns_norm.csv",
    date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
    index_col=0
)
# Any tuple of X, y with y.date >= BOUNDARY will be classified as
# a testing case.
TRAINING_BOUNDARY = datetime(2019, 1, 1)
# ============== End ==============


def all_valid_verification(X: pd.DataFrame, y: pd.DataFrame) -> bool:
    """
    A verification method requires all entries in both feature set and label set to be non-null.
    This function depends on specific case.
    """
    if np.any(X.isnull()):
        return False
    elif np.any(y.isnull()):
        return False
    elif len(X) == 0:
        return False
    return True


def fix_failed(X: pd.DataFrame, y: pd.DataFrame, req_len: int) -> Tuple[pd.DataFrame]:
    """
    Fix training samples with missing data.
    """
    if np.any(y.isnull()):
        # When the target is missing, this tuple cannot be fixed.
        return (None, None)

    if X.shape[0] != req_len:
        return (None, None)

    fixed_X = X.interpolate(
        method="nearest",
        aixs=0,
    )

    fixed_X.fillna(
        method="bfill",
        inplace=True
    )

    return (fixed_X, y)


def split_train_test(
    ds: List[Tuple[pd.DataFrame]]
) -> List[List[Tuple[pd.DataFrame]]]:
    """
    Splits dataset according to TRAINING_BOUNDARY.
    Returns two dataset.
    """
    raise NotImplementedError
    for X, y in ds:
        pass


def regression_feed() -> List[np.ndarray]:
    """
    Feed training and testing sets to the model evaluation method.
    Note that validation set will be extracted from the training set.
    """
    feature_list, label_list, failed_feature_list, failed_label_list = gen_dataset_calendarday(
        DF,
        TARGET_COL,
        LAG_DAYS,
        verify=all_valid_verification
    )
    ds_passed = list(zip(feature_list, label_list))
    ds_failed = list(zip(failed_feature_list, failed_label_list))

    scope = max(len(x) for x in feature_list)
    print(f"Scope of features (num. of days): {scope}")

    ds_fixed = list(
        fix_failed(X, y, req_len=scope)
        for X, y in ds_failed
    )

    ds_fixed = list(filter(
        lambda z: z[0] is not None and z[0] is not None,
        ds_fixed
    ))

    ds_total = ds_passed + ds_fixed

    # Sort according to target's timestamp.
    ds_total.sort(key=lambda x: x[1].index)
