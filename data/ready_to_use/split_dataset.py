"""
Split the crude oil and RPNA datasets to training set and testing set.
"""
from typing import Tuple, Union

from datetime import datetime


import numpy as np
import pandas as pd


DATE_RANGE_TYPE = Tuple[Union[str, datetime], Union[str, datetime]]


def load_dataset(
    path: str,
    date_col: str = "DATE"
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        index_col=0,
        header=0,
        parse_dates=[date_col],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    return df


def main(
    train_range: DATE_RANGE_TYPE,
    test_range: DATE_RANGE_TYPE,
    pack_dir: str
) -> None:
    """
    Splits the package of datasets into training and testing packages.

    NOTE: this method only split the dataset into two subsets, validation/dev
    set is NOT constructed.

    The pipeline of isolating a validation set for hyper-parameter tuning is
    included in model scripts.

    Arguments:
        train_range {DATE_RANGE_TYPE} -- [date range for training set.]
        test_range {DATE_RANGE_TYPE} -- [date range for testing set.]
        pack_dir {str} -- [the directory to ]
    """
    if not pack_dir.endswith("/"):
        pack_dir += "/"
        print(f"Generated datasets are saved to {pack_dir}")
    # STOPPED HERE


if __name__ == "__main__":
    main()
