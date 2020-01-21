"""
Aggregate data processing utility, generate the ready to use dataset.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

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


def main(
    config: dict
) -> None:
    df_lst = list()

    df_rpna_oil = _load_rpna(
        src_file=config["rpna.crude_oil.src"],
        radius=config["rpna.radius"]
    )
    df_lst.append(df_rpna_oil)


# Testing utilities
def _load_default_config():
    with open("./dt_config.json", "r") as f:
        return json.load(f)


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
