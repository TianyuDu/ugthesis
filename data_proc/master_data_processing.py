"""
Aggregate data processing utility, generate the ready to use dataset.
"""
import argparse
import json
import os
import warnings
from datetime import datetime
from pprint import pprint
from typing import List, Union

import numpy as np
import pandas as pd

import fred_macro_features
import rpna_processing


# HPTD: add docstrings.


def _load_rpna(
    src_file: str,
    radius: float
) -> pd.DataFrame:
    threshold = (-radius, radius)
    print(f"Constructing news count with threshold = {threshold}")
    df = pd.read_csv(src_file)
    p = rpna_processing.main(df, threshold)
    for y in ["ESS", "WESS"]:
        print(f"News type composition using {y}.")
        total = sum(
            p[f"NUM_{x}_{y}"].sum()
            for x in ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        )
        for x in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
            perc = p[f"NUM_{x}_{y}"].sum() / total
            print(f"{x}: {perc * 100: 0.2f}%")
    # Convert to freq=D, this may add nan data to weekends.
    p = p.asfreq("B")
    return p


def _load_wti(src_file: str) -> pd.DataFrame:
    oil_price = pd.read_csv(
        src_file,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    oil_price = oil_price.asfreq("B")
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

# df2 = _generate_lags(df, 3)


def _check_df_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    out = list()
    for col in df1.columns:
        out.append(np.sum(df1[col] - df2[col]) < 1e-12)
    for col in df2.columns:
        out.append(np.sum(df1[col] - df2[col]) < 1e-12)
    assert set(df1.columns) == set(df2.columns), "Columns not matched."
    assert all(out), "Value difference."


def main(
    config: dict,
) -> None:
    df_lst = list()
    if config["oil.include"]:
        # Load crude oil price.
        df_oil_price = _load_wti(
            src_file=config["oil.src"]
        )
        df_oil_price = df_oil_price.asfreq(config["index.master_freq"])

        df_lst.append(
            _generate_lags(
                df_oil_price,
                config["oil.lags"]
            ))
    else:
        warnings.warn("Oil dataset is NOT included.")

    if config["rpna.include"]:
        # Load RPNA dataset on crude oil.
        df_rpna_oil = _load_rpna(
            src_file=config["rpna.crude_oil.src"],
            radius=config["rpna.radius"]
        )

        df_rpna_oil = df_rpna_oil.asfreq(config["index.master_freq"])

        df_rpna_oil.fillna(0.0, inplace=True)

        df_lst.append(
            _generate_lags(
                df_rpna_oil,
                config["rpna.lags"]
            ))
    else:
        warnings.warn("RPNA dataset is NOT included.")

    if config["fred.include"]:
        # Load macro variables from Fred.
        df_macro = _load_macro(
            src_file=config["fred.src"]
        )
        df_macro = df_macro.asfreq(config["index.master_freq"])

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
    merged = pd.concat(df_lst, axis=1)
    merged = merged[sorted(merged.columns)]

    # Select subset
    def parser(x):
        return datetime.strptime(x, "%Y-%m-%d")

    start_date, end_date = map(
        parser,
        (config["index.start_date"], config["index.end_date"])
    )

    def subset(x):
        s = np.logical_and(
            x.index >= start_date,
            x.index <= end_date
        )
        return s

    merged = merged[subset(merged)]

    # Format frequency and data type.
    merged = merged.asfreq(config["index.master_freq"])
    merged = merged.astype(np.float32)

    # report summary
    __report_na(merged)
    return merged


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
    # TODO: Stopped here.

# t, current = __validate_dataset(df, config)


if __name__ == "__main__":
    # Add configuration.
    parser = argparse.ArgumentParser()
    now_str = datetime.strftime(datetime.now(), "%D-%T")
    parser.add_argument(
        "--save_to",
        default=f"./master_dataset_{now_str}.csv",
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
