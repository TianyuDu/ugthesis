"""
The script for crude oil market eda.
"""
import sys
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import norm

plt.style.use("seaborn-dark")


def construct_return(raw_price: pd.DataFrame):
    """Transform raw price to log-difference (return)"""
    log_price = np.log(raw_price).dropna()
    ret = log_price.diff().dropna()
    ret.columns = ["DCOILWTICO_REAL_RETURN"]
    return ret


def main(
    wti_price_dir: str = "./data/fred/DCOILWTICO.csv",
    cpi_dir: str = "./data/fred/CPIAUCSL.csv",
    start=datetime(2000, 1, 1),
    end=datetime(2019, 9, 30),
    verbose: bool = True,
    write_to_disk: bool = False,
    save_dir: Union[str, None] = None
) -> None:
    print(f"write to {save_dir}")
    df_wti_raw = pd.read_csv(
        wti_price_dir,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )

    df_cpi_raw = pd.read_csv(
        cpi_dir,
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
        df_wti.info()
        print(df_wti.head())
        print(df_wti.tail())
        df_cpi.info()
        print(df_cpi.head())
        print(df_cpi.tail())

    # Normalizing using CPIAUCSL to construct real crude oil price.
    # Creating CPI referencing series
    df_norm_cpi = df_cpi / df_cpi.iloc[0, 0]  # 2000.01.01 as 1.00 index.
    df_norm_cpi_daily = df_norm_cpi.resample(
        "d", fill_method="ffill", label="left")
    # fig = plt.figure(figsize=(15, 3), dpi=300)
    # plt.scatter(df_norm_cpi.index, df_norm_cpi.values, label="monthly", alpha=0.7, s=1, color="red")
    # plt.plot(df_norm_cpi_daily, label="daily", alpha=0.3)
    # plt.legend()
    # plt.show()
    df_norm_cpi = df_norm_cpi_daily

    # Normalize operation, sequential implementation.
    df_wti_real = pd.DataFrame(
        columns=["DCOILWTICO_REAL"], index=df_wti.index).astype(np.float32)
    for t in df_wti_real.index:
        try:
            real_price = df_wti["DCOILWTICO"][t] / df_norm_cpi["CPIAUCSL"][t]
            df_wti_real["DCOILWTICO_REAL"][t] = real_price
        except KeyError:
            print(f"Skipped: {t}")

    plt.plot(df_wti_real, linewidth=0.5, label="WTI Crude Oil Price (Real)")
    plt.plot(df_wti, linewidth=0.5, label="WTI Crude Oil Price (Nominal)")
    plt.legend()

    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "nominal_and_real_wti_price.png", dpi=300)
    plt.close()
    # Construct returns.
    df_wti_return = construct_return(df_wti_real)
    # Histogram of returns
    sns.distplot(
        df_wti_return.values,
        # bins=80,
        fit=norm,
        fit_kws={"lw": 1, "label": "Gaussian Fit"},
        kde=True,
        kde_kws={"lw": 1, "label": "KDE"},
        label=f"Return (diff-log) on real crude oil price (N={len(df_wti_return.values)})")
    plt.legend()
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "hist_wti_return.png", dpi=300)
    plt.close()
    # The return as a series.
    plt.plot(
        df_wti_return,
        linewidth=0.5,
        label="Return (diff-log) on real crude oil prices"
    )
    plt.legend()
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_return.png", dpi=300)
    plt.close()
    # ACF and PACF.
    # For real prices.
    sm.tsa.graphics.plot_acf(df_wti_real.dropna().values, lags=32)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_real_acf.png", dpi=300)
    plt.close()
    sm.tsa.graphics.plot_pacf(df_wti_real.dropna(), lags=32)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_real_pacf.png", dpi=300)
    plt.close()
    # For change on real prices.
    sm.tsa.graphics.plot_acf(df_wti_real.diff().dropna().values, lags=32)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_diff_real_aacf.png", dpi=300)
    plt.close()
    sm.tsa.graphics.plot_pacf(df_wti_real.diff().dropna(), lags=32)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_diff_real_pacf.png", dpi=300)
    plt.close()
    # For returns on real prices.
    sm.tsa.graphics.plot_acf(df_wti_return.dropna().values, lags=32)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_return_acf.png", dpi=300)
    plt.close()
    sm.tsa.graphics.plot_pacf(df_wti_return.dropna(), lags=32)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir + "wti_return_pacf.png", dpi=300)
    plt.close()
    if write_to_disk:
        # Save generated results
        df_wti_real.dropna().to_csv("../data/ready_to_use/wti_crude_oil_price_real.csv")
        df_wti_return.dropna().to_csv("../data/ready_to_use/wti_crude_oil_return_real.csv")


if __name__ == "__main__":
    print(sys.version)
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.grid"] = True
    main(save_dir="./figures/")
    # main(save_dir=None)
