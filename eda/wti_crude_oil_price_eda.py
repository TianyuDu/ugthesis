"""
The script for crude oil market eda.
"""
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import norm

plt.style.use("seaborn-dark")


def main(
    start=datetime(2000, 1, 1),
    end=datetime(2019, 9, 30),
    verbose: bool = True,
    write_to_disk: bool = False
) -> None:
    df_wti_raw = pd.read_csv(
        "../data/fred/DCOILWTICO.csv",
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )

    df_cpi_raw = pd.read_csv(
        "../data/fred/CPIAUCSL.csv",
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )

    def select_range(df):
        return df[np.logical_and(
            df.index >= start, df.index <= end
        )]

    df_wti, df_cpi = map(select_range, (df_wti_raw, df_cpi_raw))
    df_wti = df_wti[df_wti.values != "."]
    df_wti, df_cpi = map(lambda x: x.astype(np.float32), (df_wti, df_cpi))
    
    if verbose:
        print(df_wti.info())
        print(df_wti.head())
        print(df_wti.tail())
        print(df_cpi.info())
        print(df_cpi.head())
        print(df_cpi.tail())


if __name__ == "__main__":
    print(sys.version)
    main()
