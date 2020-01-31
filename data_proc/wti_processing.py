"""
Jan. 31, 2019
Processing utilities for crude oil price dataset.
"""
from typing import Union

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

