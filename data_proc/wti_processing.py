"""
Jan. 31, 2019
Processing utilities for crude oil price dataset.
"""
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# plt.style.use("grayscale")

import interpolate.interpolate_utils as utils


def compute_returns(
    df: pd.DataFrame,
) -> Set[pd.DataFrame]:
    """
    Compute returns from the raw dataset.
    Returns are computed using the difference between log
    closing prices between consecutive trading days.

    Args:
        df: dataset for crude oil price.
    Return:
        returns: returns with missing values.
        returns
    """
    # Drop na to calculate difference between prices on two consecutive trading days.
    filtered = df["DCOILWTICO"].dropna()
    returns = np.log(filtered).diff()
    returns = returns.rename("RETURN")
    # Change to business day frequency.
    returns = returns.asfreq("B")
    returns = pd.DataFrame(returns)
    print(f"Length of return series: {len(returns)}")
    print(f"Percentage Missing in returns: {np.mean(returns.isna()) * 100: 0.3f} %.")
    print("Missing value information")
    z = np.logical_not(returns.isnull().values)
    print(returns[z].index.day_name().value_counts())
    returns_filled = utils.arima_interpolate(
        raw=returns,
        arima_order=(1, 0, 1),
        verbose=True
    )
    return returns, returns_filled


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


def main(df: pd.DataFrame) -> None:
    """
    main function reporting results.
    """
    df = df.copy()
    df["DAY"] = df.index.day_name()
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]
    print(df.head())

    def report(today):
        print(f"Total day: {len(today)}")
        print(f"Trading day: {len(today.dropna())}")
        print(today["DELTA"].describe())
        print(today["DELTA"].value_counts())

    for day in days:
        print(f"================{day}================")
        mask = (df["DAY"] == day)
        report(df[mask])
    print("================Overall================")
    report(df)

    # Detailed reports on missing dates.
    mask = df["DCOILWTICO"].isna()
    # Missing WEEKdays, weekends are always missing.
    df_missing = df[mask].copy()
    df_missing["MONTH_DAY"] = df_missing.index.strftime(
        "%B-%d")
    fmt = "================{}================"
    print(fmt.format("Missing Day"))
    df_missing["MONTH_DAY"].describe()
    print(fmt.format("Top 20 Missing Day"))
    print(df_missing["MONTH_DAY"].value_counts().head(40))

    print(fmt.format("Missing Day without Weekends"))
    is_weekend = np.logical_or(
        df_missing["DAY"] == "Saturday",
        df_missing["DAY"] == "Sunday"
    )
    df_missing_weekedays = df_missing[np.logical_not(is_weekend)]
    df_missing_weekedays["MONTH_DAY"].describe()
    print(fmt.format("Top 20 Missing Day (exclude Weekends)"))
    print(df_missing_weekedays["MONTH_DAY"].value_counts().head(40))


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
    df_delta = compute_time_lags(
        df.asfreq("D"),
        append_to_original=True
    )
    main(df_delta)
