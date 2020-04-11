"""
Date: Feb. 22, 2020.
Data loaders for most models.
"""
import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from typing import List, Tuple, Union, Optional
from datetime import datetime

from sklearn import model_selection

from utils.time_series_utils import gen_dataset_calendarday

from data_proc.rpna_processing import convert_timestamp_wti

# ============ Configs ============
# MASTER_DIR = "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis"
MASTER_DIR = ".."
TARGET_COL = "RETURN"
LAG_DAYS = 28
# DF_RETURNS = pd.read_csv(
#     MASTER_DIR + "/data/ready_to_use/returns_norm.csv",
#     date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
#     index_col=0
# )

# Read the raw news.
# DF_RAW_NEWS = convert_timestamp_wti(
#     pd.read_csv(MASTER_DIR + "/data/ravenpack/crude_oil_all.csv")
# )

# DF_NEWS = pd.read_csv(
#     MASTER_DIR + "/data/ready_to_use/rpna_r0_wess.csv",
#     date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
#     index_col=0
# )

# DF_MASTER = pd.read_csv(
#     MASTER_DIR + "/data/ready_to_use/master_bothR0.csv",
#     date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
#     index_col=0
# )

# Any tuple of X, y with y.date >= BOUNDARY will be classified as
# a testing case.
TRAINING_BOUNDARY = datetime(2019, 1, 1)
# ============== End ==============


def align_datasets(
    main_df: pd.DataFrame,
    side_df: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Align multiple datasets, and merge them together.
    """
    index = main_df.index


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
    # FIXME: bettter fixing methods?
    if X.shape[0] < req_len:
        return (None, None)

    if np.any(y.isnull()):
        # When the target is missing, this tuple cannot be fixed.
        return (None, None)

    if X.shape[0] != req_len:
        return (None, None)

    try:
        fixed_X = X.interpolate(
            method="nearest",
            aixs=0,
        )

        fixed_X.fillna(
            method="bfill",
            inplace=True
        )

        fixed_X.fillna(
            method="ffill",
            inplace=True
        )
        return (fixed_X, y)
    except ValueError:
        return (None, None)


def check_ds(ds: List[List[Tuple[pd.DataFrame]]]) -> None:
    """
    An sanity check on the dataset.
    """
    # Check scope size.
    assert min(len(pair[0]) for pair in ds) == max(len(pair[0]) for pair in ds)
    for X, y in ds:
        target_date = y.index[0]
        last_feature_date = X.index[-1]
        assert target_date > last_feature_date
        assert not np.any(X.isnull())
        assert not np.any(y.isnull())


def split_train_test(
    ds: List[Tuple[pd.DataFrame]]
) -> Tuple[List[Tuple[pd.DataFrame]]]:
    """
    Splits dataset according to TRAINING_BOUNDARY.
    Returns two datasets.
    """
    train_set, test_set = [], []
    for X, y in ds:
        current_date = y.index[0]
        if current_date < TRAINING_BOUNDARY:
            # Training range.
            train_set.append((X, y))
        else:
            # Test range.
            test_set.append((X, y))
    return train_set, test_set


def convert_to_array(ds) -> Tuple[np.ndarray]:
    """
    Converts dataset (list of tuples) to arrays.
    """
    X_lst = [z[0].values for z in ds]
    y_lst = [z[1].values for z in ds]
    X = np.stack(X_lst)
    y = np.stack(y_lst)
    return (X, y)


def insert_days(ds) -> Tuple[np.ndarray]:
    """
    Insert weekdays of dates of X and y in the dataset.
    Note that this method is for reference and debugging
    purpose only.
    """
    X, y = ds[0].copy(), ds[1].copy()
    X.insert(loc=0, column="DAY", value=X.index.day_name())
    y.insert(loc=0, column="DAY", value=y.index.day_name())
    return (X, y)


def day_filter(
    X_lst: List[pd.DataFrame],
    y_lst: List[pd.DataFrame],
    day: Union[str, List[str]]
) -> Tuple[List[pd.DataFrame]]:
    """
    Returns only samples with y.index matching
    the provided day of week.
    day should be in formats:
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
    """
    X_sel, y_sel = [], []
    for X, y in zip(X_lst, y_lst):
        t = y.index.day_name().item()
        if isinstance(day, list):
            if t in day:
                X_sel.append(X)
                y_sel.append(y)
        if isinstance(day, str):
            if t == day:
                X_sel.append(X)
                y_sel.append(y)
    return X_sel, y_sel


def feed(
    include: str = "master",
    day: Union[None, str, List[str]] = None,
    task: Union["regression", "classification"] = "regression"
) -> List[np.ndarray]:
    """
    Feed training and testing sets to the model evaluation method.
    Note that validation set will be extracted from the training set.
    Returns:
        (X_train, y_train, X_test, y_test)
    """
    if include == "master":
        feature_list, label_list, failed_feature_list, failed_label_list = gen_dataset_calendarday(
            DF_MASTER,
            TARGET_COL,
            LAG_DAYS,
            verify=all_valid_verification
        )
    elif include == "return":
        feature_list, label_list, failed_feature_list, failed_label_list = gen_dataset_calendarday(
            DF_RETURNS,
            TARGET_COL,
            LAG_DAYS,
            verify=all_valid_verification
        )
    # Subset only one particular weekday.
    if day is not None:
        feature_list, label_list = day_filter(feature_list, label_list, day)
        failed_feature_list, failed_label_list = day_filter(
            failed_feature_list, failed_label_list, day
        )

    ds_passed = list(zip(feature_list, label_list))
    print(f"Number of observations passed: {len(ds_passed)}")
    ds_failed = list(zip(failed_feature_list, failed_label_list))
    print(f"Number of observations failed: {len(ds_failed)}")

    scope = max(x.shape[0] for x in feature_list)
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
    ds_total.sort(key=lambda x: x[1].index)
    print(f"Total number of training pairs (X, y) generated: {len(ds_total)}")
    # Sort according to target's timestamp.
    train_set, test_set = split_train_test(ds_total)
    check_ds(train_set)
    # check_ds(test_set)
    X_train, y_train = convert_to_array(train_set)
    X_test, y_test = convert_to_array(test_set)
    if task == "classification":
        print("Convert labels to {0, 1} for classification purpose.")
        y_train = convert_to_onehot(y_train)[:, 1].squeeze()
        y_test = convert_to_onehot(y_test)[:, 1].squeeze()
    print(f"X_train @ {X_train.shape}\ny_train @ {y_train.shape}\nX_test @ {X_test.shape}\ny_test @ {y_test.shape}")
    return (X_train, y_train, X_test, y_test)


def convert_to_onehot(y: np.ndarray) -> np.ndarray:
    N = y.shape[0]
    y = y.reshape(-1,)
    onehot = np.zeros((N, 2))
    onehot[:, 0] = (y < 0.0)
    onehot[:, 1] = (y >= 0.0)
    onehot = onehot.astype(np.int32)
    return onehot


def direct_feed(
    src: str,
    test_start = pd.to_datetime("2019-01-01"),
    day: Optional[Union[str, List[str]]] = None,
    return_array: bool = False
) -> Tuple[Union[np.ndarray, pd.DataFrame]]:
    """
    Feed (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(
        src,
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )
    original_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped rows involving Nan: {original_len - len(df)}")
    if day is not None:
        if isinstance(day, str):
            day = [day]
        day_mask = df.index.day_name().isin(day)
        df = df[day_mask]

    # Split training and testing subsets.
    test_mask = (df.index >= test_start)
    df_train, df_test = df[~ test_mask], df[test_mask]

    X_train = df_train.drop(columns=["TARGET"])
    y_train = df_train["TARGET"]

    X_test = df_test.drop(columns=["TARGET"])
    y_test = df_test["TARGET"]

    if day is not None:
        # Double check day names in the index.
        for s in X_train, X_test, y_train, y_test:
            assert set(s.index.day_name()) == set(day)

    if return_array:
        # Convert to numpy arrays
        X_train, X_test, y_train, y_test = map(
            lambda x: x.values,
            (X_train, X_test, y_train, y_test)
        )

    return X_train, X_test, y_train, y_test


def rnn_feed(
    src: str,
    train_size: float = 0.8
) -> Tuple[np.ndarray]:
    """
    Feed (X_train, X_test, y_train, y_test) to models.
    X_{train, test} @ (batch_size, seq_len, num_fea)
    y_{train, test} @ (batch_size,)
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=train_size, shuffle=True)
    return X_train, X_test, y_train, y_test
