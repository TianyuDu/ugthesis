"""
Aggregate data processing utility, generate the ready to use dataset.
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pprint import pprint
from typing import List, Union

import numpy as np
import pandas as pd

sys.path.append("../")

import data_proc.fred_macro_features
import data_proc.information_flow
import data_proc.rpna_processing
from data_proc.rpna_processing import convert_timestamp_wti


MASTER_DIR = "../"
TARGET_COL = "RETURN"
LAG_DAYS = 28
DF_RETURNS = pd.read_csv(
    MASTER_DIR + "/data/ready_to_use/returns_norm.csv",
    date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
    index_col=0
)

# Read the raw news.
DF_RAW_NEWS = convert_timestamp_wti(
    pd.read_csv(MASTER_DIR + "/data/ravenpack/crude_oil_all.csv")
)
DF_RAW_NEWS["TIMESTAMP_WTI"] = DF_RAW_NEWS["TIMESTAMP_WTI"].apply(
    lambda x: x.strftime("%m/%d/%Y, %H:%M:%S")).astype("datetime64")


DF_NEWS = pd.read_csv(
    MASTER_DIR + "/data/ready_to_use/rpna_r0_wess.csv",
    date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
    index_col=0
)

DF_NEWS = DF_NEWS.asfreq("D")

DF_MASTER = pd.read_csv(
    MASTER_DIR + "/data/ready_to_use/master_bothR0.csv",
    date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
    index_col=0
)


def _load_rpna(
    src_file: str
) -> pd.DataFrame:
    """
    Loads RPNA dataset from disk.
    """
    print(f"Reading RPNA dataset from {src_file}")
    rpna = pd.read_csv(
        src_file,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    return rpna


def _load_wti(src_file: str) -> pd.DataFrame:
    oil_price = pd.read_csv(
        src_file,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    return oil_price


def _load_macro(src_file: str) -> pd.DataFrame:
    macro_panel = fred_macro_features.align_dataset(src=src_file)
    return macro_panel


def _generate_lags(
    df: pd.DataFrame,
    lags: Union[int, List[int]]
) -> pd.DataFrame:
    df = df.copy()
    # Constructs variables with lagged values.
    if isinstance(lags, int):
        lags = range(lags)
    cols = df.columns

    collection = list()

    for L in lags:
        df_lagged = df.shift(L)
        df_lagged.columns = [
            x + f"_L{L}"
            for x in cols
        ]
        collection.append(df_lagged)
    merged = pd.concat(collection, axis=1)
    cols = merged.columns
    merged = merged[sorted(merged.columns)]
    return merged


def _check_df_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    out = list()
    for col in df1.columns:
        out.append(np.sum(df1[col] - df2[col]) < 1e-12)
    for col in df2.columns:
        out.append(np.sum(df1[col] - df2[col]) < 1e-12)
    assert set(df1.columns) == set(df2.columns), "Columns not matched."
    assert all(out), "Value difference."


def _add_target(
    features: pd.DataFrame,
    targets: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    # TODO: do this.
    raise NotImplementedError


def generate_pairs(
    config: dict
) -> Tuple[List[pd.DataFrame]]:
    """
    Generates the (X, y) training pairs.
    Returns: (X, y)
        where X is the collection of predictors[t-k, ...,t-1]
        and y is the return at day t.
    """
    dates = pd.bdate_range(config["index.start_date"], config["index.end_date"])
    dates = dates.intersection(DF_RETURNS.index)
    original_len = len(dates)
    # Subset returns.
    df_returns = DF_RETURNS.loc[dates]
    X_lst, y_lst = list(), list()

    one_day = timedelta(days=1)

    for t, r in zip(df_returns.index, df_returns["RETURN"].values):
        fea_t = list()
        if np.isnan(r):
            # For nan returns, skip this.
            continue
        # look into summary table of RPNA.
        lags = timedelta(days=config["rpna.lags"])
        rg = pd.date_range(t - lags, t - one_day)
        fea_rpna = DF_NEWS.loc[rg]
        fea_rpna.fillna(value=0.0, inplace=True)
        fea_t.append(fea_rpna.values.reshape(-1,))

        # look into raw RPNA news.
        # get the information flow.
        subset = (t - lags <= DF_RAW_NEWS["TIMESTAMP_WTI"]) & (DF_RAW_NEWS["TIMESTAMP_WTI"] <= t - one_day)
        if_t = DF_RAW_NEWS[subset]
        if_features = information_flow.extract_IF(if_t)


def main(
    config: dict,
) -> None:
    df_lst = list()
    if config["oil.include"]:
        # Load crude oil price.
        df_returns = _load_wti(src_file=config["oil.src"])
        df_returns = df_returns.asfreq(config["index.master_freq"])
        df_lst.append(df_returns)
    else:
        warnings.warn("Oil dataset is NOT included.")

    if config["rpna.include"]:
        # Load RPNA dataset on crude oil.
        df_rpna_oil = _load_rpna(src_file=config["rpna.crude_oil.src"])
        df_rpna_oil = df_rpna_oil.asfreq(config["index.master_freq"])
        df_lst.append(df_rpna_oil)
    else:
        warnings.warn("RPNA dataset is NOT included.")

    if config["fred.include"]:
        raise NotImplementedError
        # Load macro variables from Fred.
        df_macro = _load_macro(
            src_file=config["fred.src"]
        )
        df_macro = df_macro.asfreq(config["index.master_freq"])

        # TODO: fix this.
        # NOTE: macro variables are already lagged variables
        # which indicate measures in the previous measuring period (e.g., month).
        df_lst.append(df_macro)
        df_lst.append(
            _generate_lags(
                df_macro,
                config["fred.lags"]
            ))
    else:
        warnings.warn("FRED dataset is NOT included.")

    # Combine datasets
    df_merged = pd.concat(df_lst, axis=1)

    # Select subset
    def parser(x):
        return datetime.strptime(x, "%Y-%m-%d")

    start_date, end_date = map(
        parser,
        (config["index.start_date"], config["index.end_date"])
    )

    def select_subset(x):
        s = np.logical_and(x.index >= start_date, x.index <= end_date)
        return s

    df_merged = df_merged[select_subset(df_merged)]

    # Format frequency and data type.
    df_merged = df_merged.asfreq(config["index.master_freq"])
    df_merged = df_merged.astype(np.float32)

    # report summary
    __report_na(df_merged)
    return df_merged


# Testing utilities
def __load_default_config():
    with open("./dt_config.json", "r") as f:
        return json.load(f)


def __report_na(df) -> None:
    avg = np.mean(df.isna().sum(axis=1) > 0)
    total = np.sum(df.isna().sum(axis=1) > 0)
    print(f"Dates with invalid entries: {total}({avg * 100: 0.3f}%).")


def __validate_dataset(
    df: pd.DataFrame,
    config: dict,
    prop_rand: float = 0.1,
) -> None:
    """A sanity check on whether feature dataset is correctly
    constructed.
    """
    sampled_idx = np.random.choice(
        df.index,
        size=int(prop_rand * len(df))
    )
    for t in sampled_idx:
        current = df.loc[t, :]
        return (t, current)
    raise NotImplementedError
    # TODO: Stopped here.

# t, current = __validate_dataset(df, config)


if __name__ == "__main__":
    # Add configuration.
    parser = argparse.ArgumentParser()
    now_str = datetime.strftime(datetime.now(), "%Y-%m-%d-%H:%M:%S")
    parser.add_argument(
        "--save_to",
        default=f"../data/ready_to_use/master_dataset_{now_str}.csv",
        type=str
    )
    parser.add_argument("--config", default=None, type=str)
    args = parser.parse_args()

    if args.config is not None:
        assert os.path.exists(args.config)
        print(f"Read configuration from {args.config}")
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        print("Read default configuration file.")
        config = __load_default_config()
    print("===============CONFIG===============")
    pprint(config)
    print("====================================")
    df = main(config)
    df.to_csv(args.save_to)
