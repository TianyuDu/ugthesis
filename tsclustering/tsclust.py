"""
Time series clustering.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tsclust_utils


def main(
    df_wti_real: pd.DataFrame,
    df_wti_return: pd.DataFrame,
    verbose: bool = True
) -> None:
    if verbose:
        print("REAL PRICE series received:")
        print(df_wti_real.info())
        print("LOG-DIFF RETURN ON REAL PRICE series received:")
        print(df_wti_return.info())
    mo_subseqs, mo_stats = tsclust_utils.monthly_subsequence(df_wti_return)
    df_gmm_labels = tsclust_utils.time_series_clustering(
        mo_stats,
        n_clusters=3,
        normalize=True
    )
    df_wti_return_labelled, df_wti_real_labelled = map(
        lambda x: tsclust_utils.broadcast_monthly_clustering(
            x, df_gmm_labels
        ),
        (df_wti_return, df_wti_real)
    )
    plt.close()
    tsclust_utils.regime_plot(
        df_wti_real_labelled,
        df_wti_real_labelled["label"].values,
        "DCOILWTICO_REAL",
        color_map="auto",
        save_dir="./figures/wti_real_regime.png"
    )
    plt.close()
    tsclust_utils.regime_plot(
        df_wti_return_labelled,
        df_wti_return_labelled["label"].values,
        "DCOILWTICO_RETURN_REAL",
        color_map="auto",
        save_dir="./figures/wti_return_regime.png"
    )


if __name__ == "__main__":
    # Matplotlib Conifg
    plt.rcParams["figure.figsize"] = (15, 3)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.grid"] = True
    # Load data
    df_wti_real = pd.read_csv(
        "./data/ready_to_use/wti_crude_oil_price_real.csv",
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    df_wti_return = pd.read_csv(
        "./data/ready_to_use/wti_crude_oil_return_real.csv",
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    main(df_wti_real, df_wti_return)
