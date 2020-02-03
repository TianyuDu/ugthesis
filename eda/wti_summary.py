"""
Summary statistics for crude oil dataset.
"""
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

plt.style.use("grayscale")


def plot_overview(
    df: pd.DataFrame,
    returns: pd.DataFrame,
    path: str
) -> None:
    plt.plot(df)
    plt.xlabel("Date")
    plt.ylabel("WTI Crude Oil Price")
    plt.savefig(
        path + "prices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.plot(df)
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.savefig(
        path + "returns.png",
        dpi = 300,
        bbox_inches = "tight"
    )


def summary(
    df: pd.DataFrame,
    path: str
) -> None:
    raise NotImplementedError


def acf_pacf(
    df: pd.DataFrame,
    returns: pd.DataFrame,
    path: str
) -> None:
    """
    Plots the ACF and PACF of two main datasets.
    df:
        DataFrame for prices.
    returns:
        DataFrame for returns.
    """
    sm.tsa.graphics.plot_acf(df)
    plt.savefig(path + "prices_acf.png", dpi=300, bbox_inches="tight")
    sm.tsa.graphics.plot_pacf(df)
    plt.savefig(path + "prices_pacf.png", dpi=300, bbox_inches="tight")

    sm.tsa.graphics.plot_acf(returns)
    plt.savefig(path + "returns_acf.png", dpi=300, bbox_inches="tight")
    sm.tsa.graphics.plot_pacf(returns)
    plt.savefig(path + "returns_pacf.png", dpi=300, bbox_inches="tight")


def main(
    df: pd.DataFrame,
    path: str
) -> None:
    filtered = df["DCOILWTICO"].dropna()
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str,
        default="/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/figures/wti_summary"
    )
    args = parser.parse_args()
    path = args.path
    if not path.endswith("/"):
        path += "/"
    df = pd.read_csv(
        "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/DCOILWTICO.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )
