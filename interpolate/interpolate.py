"""
Main file
"""
import argparse
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

import interpolate_utils as utils


def main(
    ts_dir: str,
    save_dir: str,
    figure_dir: str,
    arima_order: Tuple[int] = (7, 2, 0),
    start = datetime(2000, 1, 1),
    end = datetime(2019, 9, 30),
    verbose: bool = True
) -> None:
    """
    Processes the time series data.
    ts_dir:
        *.csv file of the raw dataset.
    save_dir:
        *.csv file location to save the generated dataset.
    """
    # Load dataset
    df = pd.read_csv(
        ts_dir,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    df.replace(".", np.NaN, inplace=True)
    df = df.astype(np.float32)
    if verbose:
        print("** Raw Dataset **")
        df.info()
        print(f"Nan ratio: {np.mean(np.isnan(df.values.squeeze())) * 100: .2}%")

    # select subset.
    def _select_range(df):
        return df[np.logical_and(
            df.index >= start, df.index <= end
        )]
    df = _select_range(df)
    if verbose:
        print("** Raw Dataset **")
        df.info()

    df_filled = utils.arima_interpolate(
        raw=df,
        arima_order=arima_order,
        verbose=verbose
    )
    df_filled.to_csv(save_dir)
    # Visualize Interpolation results.
    utils.visualize_interpolation(df, df_filled, figure_dir)
    # return df, df_filled
    return None


def visualize_from_file(
    ts_dir: str,
    filled_dir: str,
    figure_dir: str
) -> None:
    """
    Visualize local file only without running the interpolation.
    """
    df = pd.read_csv(
        ts_dir,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    ).replace(".", np.NaN).astype(np.float32)

    df_filled = pd.read_csv(
        filled_dir,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    ).replace(".", np.NaN).astype(np.float32)

    utils.visualize_interpolation(df, df_filled, figure_dir)


if __name__ == "__main__":
    print(sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpolate", type=int, default=1)
    args = parser.parse_args()
    if args.interpolate:
        main(
            ts_dir="./data/fred/DCOILWTICO.csv",
            save_dir="./data/read_to_use/DCOILWTICO_FILLED.csv",
            figure_dir="./figures/arima_intropolate_",
            arima_order=(7, 2, 0),
            start=datetime(2000, 1, 1),
            end=datetime(2019, 9, 30),
        )
    else:
        visualize_from_file(
            ts_dir="./data/ready_to_use/DCOILWTICO.csv",
            filled_dir="./data/ready_to_use/DCOILWTICO_FILLED.csv",
            figure_dir="./figures/arima_intropolate_"
        )
