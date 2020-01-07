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
    train_range: DATE_RANGE_TYPE = ("2000-01-01", "2017-12-31"),
    test_range: DATE_RANGE_TYPE = ("2018-01-01", "2019-09-30"),
    src_dir: str = "./data/ready_to_use/",
    pack_dir: str = "./data/ready_to_use/sample_pack/"
) -> None:
    """
    Splits the package of datasets into training and testing packages.

    NOTE: this method only split the dataset into two subsets, validation/dev
    set is NOT constructed.

    The pipeline of isolating a validation set for hyper-parameter tuning is
    included in model scripts.

    Arguments:
        Range format: datetime or string in format: %Y-%m-%d.
        train_range {DATE_RANGE_TYPE} -- [date range for training set.]
        test_range {DATE_RANGE_TYPE} -- [date range for testing set.]
        src_dir {str} -- [the directory where the source is located.]
        pack_dir {str} -- [the directory to save generated data package.]
    """
    if not src_dir.endswith("/"):
        src_dir += "/"
        print(f"Generated datasets are saved to {src_dir}")

    if not pack_dir.endswith("/"):
        pack_dir += "/"
        print(f"Generated datasets are saved to {pack_dir}")
    # Convert to datetime instance.
    parse = lambda t: datetime.strptime(t, "%Y-%m-%d") if isinstance(t, str) else t
    train_range = map(parse, train_range)
    test_range = map(parse, test_range)

    train_start, train_end = train_range
    test_start, test_end = test_range
    # Validate range.
    if not (train_start <= train_end and test_start <= test_end):
        raise ValueError("Invalid range.")
    train_idx = pd.date_range(train_start, train_end, freq="D")
    test_idx = pd.date_range(test_start, test_end, freq="D")
    if len(train_idx.intersection(test_idx)) > 0:
        raise ValueError("Training set and testing set overlap.")

    # Split raw crude oil price dataset.
    df_oil = load_dataset(src_dir + "DCOILWTICO_FILLED.csv")

    df_oil_train = df_oil.loc[train_idx].dropna()
    df_oil_test = df_oil.loc[test_idx].dropna()

    df_oil_train.to_csv(pack_dir + "WTI_price_train.csv")
    df_oil_test.to_csv(pack_dir + "WTI_price_test.csv")

    # Split real oil price datast.
    df_rprice = load_dataset(src_dir + "wti_crude_oil_price_real.csv")

    df_rprice_train = df_rprice.loc[train_idx].dropna()
    df_rprice_test = df_rprice.loc[test_idx].dropna()

    df_rprice_train.to_csv(pack_dir + "WTI_real_price_train.csv")
    df_rprice_test.to_csv(pack_dir + "WTI_real_price_test.csv")

    # RPNA dataset.
    df_rpna = load_dataset(src_dir + "rpna.csv")
    df_rpna.loc[train_idx].dropna().to_csv(pack_dir + "rpna_train.csv")
    df_rpna.loc[test_idx].dropna().to_csv(pack_dir + "rpna_test.csv")

    # Macro dataset.
    # TODO:

    # GMM Label set.
    # TODO:


if __name__ == "__main__":
    main()
