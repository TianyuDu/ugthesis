"""
Analyze the day of week effect.

Jan. 31, 2019.
"""
import argparse
from datetime import datetime
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

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
            results.append(f"{stats:0.3f}({pval: 0.3f})")
        print(fmt.format(d1[:3], *results))


def day_effect(
    df: pd.DataFrame,
    path: Union[str, None],
) -> None:
    """
    Generates a list of distributions based on days and delta values.
    Test whether there are day effects.
    """
    df["DAY"] = df.index.day_name()

    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]

    df_returns = compute_return(df)

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
                prices[day], bins=40, label="price"
            )
            plt.title(day)
            ax.set_xlim([-20, 140])
            if path is None:
                plt.show()
            else:
                plt.savefig(
                    path + f"{day}_prices.png",
                    bbox_inches="tight",
                    dpi=300
                )
            plt.close()

    # Plot returns.
    for day in days:
        if day not in ["Saturday", "Sunday"]:
            fig, ax = plt.subplots()
            plt.hist(
                returns[day], bins=40, label="return"
            )
            plt.title(f"{day} (N={len(returns[day])})")
            ax.set_xlim([-0.2, 0.2])
            plt.legend()
            if path is None:
                plt.show()
            else:
                plt.savefig(
                    path + f"{day}_returns.png",
                    bbox_inches="tight",
                    dpi=300
                )
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig_dir", type=str)
    args = parser.parse_args()

    path = args.fig_dir
    if not path.endswith("/"):
        path += "/"

    df = pd.read_csv(
        "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/DCOILWTICO.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )

    day_effect(df, path=path)
