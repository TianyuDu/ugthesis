"""
Analyze the day of week effect.

Jan. 31, 2019.
"""
import argparse
from datetime import datetime
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, moment
import seaborn as sns

plt.style.use("grayscale")


def compute_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the accumulated return from last trading day.
    """
    raw = df["DCOILWTICO"].copy().dropna()
    ret = pd.DataFrame(np.log(raw).diff().dropna())
    ret.columns = ["RETURN"]
    ret["DAY"] = ret.index.day_name()
    return ret


def Kolmogorov_Smirnov_test(
    collection: Dict[str, pd.DataFrame]
) -> None:
    """
    Kolmogorov Smirnov test to determine whether the distribution of oil prices or oil returns are from
    the same distribution.

    H0: G(x) = F(x)
    """
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday"]
    days_short = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    fmt = "{}  {}  {}  {}  {}  {}"
    print(fmt.format("///", *[x + " " * 10 for x in days_short]))
    for d1 in days:
        results = []
        for d2 in days:
            s1, s2 = collection[d1].values, collection[d2].values
            stats, pval = ks_2samp(
                s1, s2,
                alternative="two-sided",
                mode="exact"
            )
            results.append(f"{stats:0.3f}({pval:0.3f}) &")
        print(fmt.format(d1[:3], *results))


def day_effect(
    df: pd.DataFrame,
    df_returns: pd.DataFrame,
    path: Union[str, None],
) -> Tuple[dict]:
    """
    Generates a list of distributions based on days and delta values.
    Test whether there are day effects.
    """
    df["DAY"] = df.index.day_name()

    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]

    df_returns["DAY"] = df_returns.index.day_name()

    prices, returns = dict(), dict()
    for day in days:
        mask = df["DAY"] == day
        p = df[mask]["DCOILWTICO"]
        prices[day] = p

        mask = df_returns["DAY"] == day
        r = df_returns[mask]["RETURN"]
        returns[day] = r

    assert sum(len(x) for x in prices.values()) == len(df)
    assert sum(len(x) for x in returns.values()) == len(df_returns)

    # Plot prices
    for day in days:
        if day not in ["Saturday", "Sunday"]:
            fig, ax = plt.subplots()
            ax.hist(
                prices[day], bins=40, label="price", alpha=0.5
            )
            plt.title(f"{day} (N={len(prices[day])})")
            ax.set_xlim([-20, 140])
            if path is None:
                plt.show()
            else:
                plt.savefig(
                    path + f"dist_prices_{day}.png",
                    bbox_inches="tight",
                    dpi=300
                )
            plt.close()

    # Plot returns.
    for day in days:
        if day not in ["Saturday", "Sunday"]:
            fig, ax = plt.subplots()
            plt.hist(
                returns[day], bins=40, label="return", alpha=0.5
            )
            plt.title(f"{day} (N={len(returns[day])})")
            ax.set_xlim([-20.0, 20.0])
            plt.legend()
            if path is None:
                plt.show()
            else:
                plt.savefig(
                    path + f"dist_returns_{day}.png",
                    bbox_inches="tight",
                    dpi=300
                )
            plt.close()

    return prices, returns


def _save_results(
    collection: Dict[str, pd.DataFrame],
    path: str
) -> None:
    if not path.endswith("/"):
        path += "/"
    for day, df in collection.items():
        print(f"{day}: length: {len(df)}")
        df.to_csv(
            path + day + ".csv"
        )


def summary_stats(
    collection: Dict[str, pd.DataFrame]
) -> None:
    """
    Create summary statistics for prices/returns in
    each day of week.
    """
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday"]
    days_short = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print("Day, N, mean, std, moment")
    for d in days:
        values = collection[d].dropna().values
        print(
            "{} & {} & {:0.3f} & {:0.3f} & {:0.3f} \\\\".format(
                d,
                len(collection[d].dropna()),
                np.mean(values),
                np.std(values),
                moment(values, moment=3) / (np.std(values) ** 3)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/figures/day_of_week_effect"
    )
    args = parser.parse_args()

    path = args.fig_dir
    if not path.endswith("/"):
        path += "/"

    df = pd.read_csv(
        "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/DCOILWTICO.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )

    df_returns = pd.read_csv(
        "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/returns_norm.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )

    prices, returns = day_effect(df, df_returns, path=path)
    print("Summary Statistics for Prices")
    summary_stats(prices)
    print("\n")
    print("Summary Statistics for Returns")
    summary_stats(returns)
    print("\n")
    print("Kolmogorov Smirnov test on Prices")
    Kolmogorov_Smirnov_test(prices)
    print("\n")
    print("Kolmogorov Smirnov test on Returns")
    Kolmogorov_Smirnov_test(returns)
    _save_results(
        returns,
        path="/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/day_returns"
    )
    print(sum(len(v) for v in returns.values()))
    print(len(df_returns))
