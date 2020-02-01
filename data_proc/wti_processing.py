"""
Jan. 31, 2019
Processing utilities for crude oil price dataset.
"""
from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

plt.style.use("grayscale")


def compute_time_lags(
    df: pd.DataFrame,
    append_to_original: bool = False
) -> pd.DataFrame:
    """
    For each time period, computes the number of business days
    since last valid observations.
    """
    _report_missing_days(df)
    df_filled = df.dropna()
    _report_missing_days(df_filled)

    # Compute delta values for the entire dataset.
    delta = pd.DataFrame(index=df.index.copy())
    delta["DELTA"] = pd.NA

    # Leave Nan for t s.t. df[t] = Nan.
    for i, t in enumerate(df_filled.index):
        if i == 0:
            continue
        curr, prev = t, df_filled.index[i - 1]
        dlt = (curr - prev).days
        # insert values
        delta["DELTA"][t] = dlt
    if append_to_original:
        combined = pd.concat([df, delta], axis=1)
        return combined
    else:
        return delta


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
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday"]
    days_short = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    fmt = "{}  {}  {}  {}  {}  {}"
    print(fmt.format("///", *[x + " " * 10 for x in days_short]))
    for d1 in days:
        results = []
        for d2 in days:
            s1, s2 = collection[d1].values, collection[d2].values
            stats, pval = ks_2samp(s1, s2)
            results.append(f"{stats:0.3f}({pval: 0.3f})")
        print(fmt.format(d1[:3], *results))


def day_effect(
    df: pd.DataFrame,
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
            plt.show()
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
            plt.show()
            plt.close()


# ============ Testing Utilities ============
def _report_missing_days(df: pd.DataFrame) -> None:
    """
    Helper function.
    """
    mask = df.isna().values
    print(f"{np.mean(mask) * 100: 0.3f}% days are missing")
    days = df[mask].index.day_name()
    print(days.value_counts())


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/DCOILWTICO.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )
    df = df.asfreq("D")
    df = compute_time_lags(df, append_to_original=True)
