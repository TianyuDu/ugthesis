"""
Main file
"""
import sys

from datetime import datetime

import numpy as np
import pandas as pd

import interpolate_utils as utils


def main(
    ts_dir: str,
    save_dir: str,
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
        df.info()
        print(f"Nan ratio: {np.mean(np.isnan(df.values.squeeze())) * 100: .2}%")
    #  dataset.
    df.astype()


if __name__ == "__main__":
    print(sys.version)
    main(
        ts_dir="/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/fred/DCOILWTICO.csv",
        save_dir="./test_file.csv"
    )
