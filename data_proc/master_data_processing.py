"""
Aggregate data processing utility, generate the ready to use dataset.
"""
import argparse
import json
import os
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

import fred_macro_features
import rpna_processing


def _load_rpna(
    src_file: str,
    radius: float
) -> pd.DataFrame:
    threshold = (-radius, radius)
    print(f"Constructing news count with threshold = {threshold}")
    df = pd.read_csv(src_file)
    p = rpna_processing.preprocessing(df, threshold)
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
    p = p.asfreq("D")
    return p


def _load_wti(src_file: str) -> pd.DataFrame:
    oil_price = pd.read_csv(
        src_file,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    oil_price = oil_price.asfreq("D")
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
    cols = mereged.columns
    merged = merged(sorted(merged.columns))
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
    master_freq: str = "B"
) -> None:
    df_lst = list()
    # Load crude oil price.
    df_oil_price = _load_wti(
        src_file=config["oil.src"]
    )
    df_oil_price = df_oil_price.asfreq(master_freq)

    df_lst.append(
        _generate_lags(
            df_oil_price,
            config["oil.lags"]
        ))

    # Load RPNA dataset on crude oil.
    df_rpna_oil = _load_rpna(
        src_file=config["rpna.crude_oil.src"],
        radius=config["rpna.radius"]
    )
    df_rpna_oil = df_rpna_oil.asfreq(master_freq)

    df_lst.append(
        _generate_lags(
            df_rpna_oil,
            config["rpna.lags"]
        ))

    # Load macro variables from Fred.
    df_macro = _load_macro(
        src_file=config["fred.src"]
    )
    df_macro = df_macro.asfreq(master_freq)

    # NOTE: macro variables are already lagged variables
    # which indicate measures in the previous measuring period (e.g., month).
    df_lst.append(df_macro)
    df_lst.append(
        _generate_lags(
            df_macro,
            config["fred.lags"]
        ))

    # Convert to business days.
    # df_lst = [d.asfreq("B") for d in df_lst]

    merged = pd.concat(df_lst, axis=1)
    merged = merged[sorted(merged.columns)]
    return merged


# Testing utilities
def __load_default_config():
    with open("./dt_config.json", "r") as f:
        return json.load(f)
    

def __validate_dataset(
    df: pd.DataFrame,
    config: dict
) -> None:
    # TODO: stopped here.


if __name__ == "__main__":
    # Add configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", default="./master_dataset.csv", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    assert os.path.exists(args.config)
    print(f"Read configuration from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)
        print(config)
    # src = args.src
    # if not src.endswith("/"):
    #     src += "/"
    main(config)
