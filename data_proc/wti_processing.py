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
    freq: str = "B",
    append_to_original: bool = False
) -> pd.DataFrame:
    """
    For each time period, computes the number of business days
    since last valid observations.
    """
    _report_missing_days(df)


def _report_missing_days(df: pd.DataFrame) -> None:
    



if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/DCOILWTICO.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )
