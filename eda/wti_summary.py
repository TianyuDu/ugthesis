"""
Summary statistics for crude oil dataset.
"""
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm

plt.style.use("grayscale")


def plot_overview(
    df: pd.DataFrame,
    returns: pd.DataFrame,
    path: str
) -> None:
    # US Recession dates.
    rece2k1_bgn = datetime(2001, 3, 1)
    rece2k1_end = datetime(2001, 11, 1)

    rece2k8_bgn = datetime(2007, 12, 1)
    rece2k8_end = datetime(2009, 6, 1)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(df, label="Price")
    plt.xlabel("Date")
    plt.ylabel("WTI Crude Oil Price")
    plt.legend(loc="upper right")
    ax.axvspan(rece2k1_bgn, rece2k1_end, alpha=0.3)
    ax.axvspan(rece2k8_bgn, rece2k8_end, alpha=0.3)
    plt.savefig(
        path + "prices.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(returns, label="return")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend(loc="upper right")
    ax.axvspan(rece2k1_bgn, rece2k1_end, alpha=0.3)
    ax.axvspan(rece2k8_bgn, rece2k8_end, alpha=0.3)
    plt.savefig(
        path + "returns.png",
        dpi = 300,
        bbox_inches = "tight"
    )
    plt.close()


def summary(
    df: pd.DataFrame
) -> None:
    YEARS = list(range(
        int(min(df.index.strftime("%Y"))),
        int(max(df.index.strftime("%Y"))) + 1
    ))

    def core(values: np.ndarray, year: str) -> None:
        # Compute stats
        N = len(values)
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        _min, _max = np.min(values), np.max(values)
        acfs = sm.tsa.acf(values, fft=False)
        # Report
        print(f"{year} & {N} & {mean:0.3f} & {median:0.3f} & {std:0.3f} & {_min:0.3f} & {_max:0.3f} & {acfs[1]:0.3f} & {acfs[3]:0.3f} & {acfs[5]:0.3f} \\\\")

    print("======== Year Distribution ========")
    print("Year & Num. Obs. & Mean & Median & Std. & Min & Max & ACF(1) & ACF(3) & ACF(5) \\\\")
    for y in YEARS:
        year_index = df.index.strftime("%Y")
        mask = year_index == str(y)
        current_year = df[mask].dropna()  # care about valid data only.
        core(current_year.values.squeeze(), year=str(y))
    print("======== Total Distribution ========")
    core(df.dropna().values.squeeze(), year="Total")


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
    plt.close()
    sm.tsa.graphics.plot_acf(df)
    plt.savefig(path + "prices_acf.png", dpi=300, bbox_inches="tight")
    sm.tsa.graphics.plot_pacf(df)
    plt.savefig(path + "prices_pacf.png", dpi=300, bbox_inches="tight")

    plt.close()
    sm.tsa.graphics.plot_acf(returns)
    plt.savefig(path + "returns_acf.png", dpi=300, bbox_inches="tight")
    sm.tsa.graphics.plot_pacf(returns)
    plt.savefig(path + "returns_pacf.png", dpi=300, bbox_inches="tight")


def plot_return_hist(
    returns: pd.DataFrame,
    path: str
) -> None:
    plt.close()
    sns.distplot(
        returns.values,
        # bins=80,
        fit=norm,
        fit_kws={"lw": 1, "label": "Gaussian Fit"},
        kde=True,
        kde_kws={"lw": 1, "label": "KDE"},
        label=f"Returns (N={len(returns.values)})"
    )
    plt.legend()
    plt.savefig(path + "return_hist.png", dpi=300, bbox_inches="tight")


def main(
    df: pd.DataFrame,
    path: str
) -> None:
    print("Processing dataset...")
    print(f"Raw dataset length: {len(df)}")
    filtered = df["DCOILWTICO"].dropna()
    print(f"Valid dataset length: {len(filtered)}")
    returns = np.log(filtered).diff().dropna().rename("RETURN")
    print(f"Return dataset length: {len(returns)}")

    print("======== Summary Statistics for Prices ========")
    summary(df)
    print("======== Summary Statistics for Returns ========")
    summary(returns)

    print("Plotting Return Distribution Contrast...")
    plt.close()
    plot_return_hist(returns, path=path)

    print("Plotting overview...")
    plot_overview(filtered, returns, path=path)

    print("Plotting ACF and PACF for prices and returns...")
    acf_pacf(filtered, returns, path=path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str,
        default="/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/figures/wti_summary/"
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
    main(df, path=path)
