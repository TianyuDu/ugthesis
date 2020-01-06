"""
Utilities for processing RPNA dataset.
"""
import argparse
from typing import Tuple, Union

import numpy as np
import pandas as pd


def convert_timestamp_wti(
    df: pd.DataFrame,
    utc_col: str = "TIMESTAMP_UTC"
) -> pd.DataFrame:
    """
    Converts UTC time to local time at WTI, Central Standard Time (US)
    """
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    dates_utc = pd.to_datetime(df[utc_col], format=fmt)
    dates_wti = dates_utc.dt.tz_localize("UTC").dt.tz_convert("US/Central")
    df.insert(loc=0, column="TIMESTAMP_WTI", value=dates_wti)
    return df


def aggregate_daily(
    raw: pd.DataFrame,
    attr_col: str,
    date_col: str = "TIMESTAMP_WTI",
    add_events: bool = True
) -> pd.DataFrame:
    df = raw.copy()
    dates = df[date_col].dt.strftime("%Y-%m-%d")
    df.insert(loc=0, value=dates, column="DATE")
    mean_ess = pd.DataFrame(
        df.groupby("DATE").mean()[attr_col]
    )
    total_ess = pd.DataFrame(
        df.groupby("DATE").sum()[attr_col]
    )
    num_events = pd.DataFrame(
        df.groupby("DATE").size()
    )
    if add_events:
        daily = pd.concat([mean_ess, total_ess, num_events], axis=1)
        daily.columns = [f"{attr_col}_MEAN", f"{attr_col}_TOTAL", "NUM_EVENTS"]
    else:
        daily = pd.concat([mean_ess, total_ess], axis=1)
        daily.columns = [f"{attr_col}_MEAN", f"{attr_col}_TOTAL"]
    daily.index = pd.to_datetime(daily.index, format="%Y-%m-%d")
    return daily


def separate_count(
    raw: pd.DataFrame,
    attr_col: str,
    date_col: str = "TIMESTAMP_WTI",
    threshold: Tuple[float] = (-10, 10)
) -> pd.DataFrame:
    """
    Generate numbers of positive, negative, and neutral news in each day.
    """
    df = raw.copy()
    dates = df[date_col].dt.strftime("%Y-%m-%d")
    df.insert(loc=0, value=dates, column="DATE")
    low, high = threshold
    df["POS_LABEL"] = (df[attr_col] > high).astype(np.int32)
    df["NEG_LABEL"] = (df[attr_col] < low).astype(np.int32)
    df["NEU_LABEL"] = np.logical_and(
        df[attr_col] <= high,
        df[attr_col] >= low
    ).astype(np.int32)
    count_lst = []
    for col in ["POS_LABEL", "NEG_LABEL", "NEU_LABEL"]:
        count = pd.DataFrame(
            df.groupby("DATE").sum()[col]
        )
        count_lst.append(count)
    df_count = pd.concat(count_lst, axis=1)
    df_count.columns = [f"{x}_{attr_col}" for x in ["NUM_POSITIVE", "NUM_NEGATIVE", "NUM_NEUTRAL"]]
    df_count.index = pd.to_datetime(df_count.index, format="%Y-%m-%d")
    return df_count


def select_features():
    raise NotImplementedError


def _check_equal(df1, df2) -> None:
    if not np.all(df1.index == df2.index):
        raise IndexError("Two dataframes should share the same index.")
    a1, a2 = df1.values, df2.values
    num_fea = a1.shape[1]
    print(
        f"Percentage same: {np.mean(np.sum(a1 == a2, axis=1) == num_fea)* 100: 0.2f} % ")


def preprocessing(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gathers above methods, the complete pipeline.
    """
    df = raw.copy()
    df["ESS"] = df["ESS"] - 50  # Normalize
    # convert to wti us central timezone.
    df = convert_timestamp_wti(df)
    # Aggregate daily ESS.
    daily_ess = aggregate_daily(df, attr_col="ESS", date_col="TIMESTAMP_WTI", add_events=False)
    # Construct ENS weighted ess.
    df["WESS"] = df["ENS"] * df["ESS"] / 100
    daily_weighted_ess = aggregate_daily(
        df,
        attr_col="WESS",
        date_col="TIMESTAMP_WTI",
        add_events=True
    )
    # Count numbers of positive, negative, and neutral news based on two criteria.
    ess_count = separate_count(df, attr_col="ESS")
    wess_count = separate_count(df, attr_col="WESS")
    _check_equal(ess_count, wess_count)

    assert np.all(daily_ess.index == daily_weighted_ess.index)
    assert np.all(ess_count.index == wess_count.index)
    assert np.all(daily_ess.index == ess_count.index)

    df_daily = pd.concat([
        daily_ess, daily_weighted_ess,
        ess_count, wess_count
    ], axis=1)
    return df_daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", type=str, default=None)
    args = parser.parse_args()

    src_file = "./data/ravenpack/crude_oil_all.csv"
    print(f"Read raw dataset from {src_file}")
    df = pd.read_csv(src_file)
    p = preprocessing(df)
    p.info()
    print(p.head())

    if args.save_to is not None:
        p.to_csv(args.save_to)
