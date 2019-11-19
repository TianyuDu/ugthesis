# Formatting stock data, generate the dataset for supervised learning problem.
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import CONSTANTS


def load_local_dataset(symb: str) -> pd.DataFrame:
    """
    Loads stock info from local disk, serves as a helper function.
    Args:
        symb: symbol to identify stock.
    """
    df = pd.read_csv(
        CONSTANTS.DIR_STOCK_DATA + symb + ".csv"
    )
    df.index = pd.to_datetime(df["Date"])
    df.drop(columns=["Date"], inplace=True)
    return df


def gen_classification_label(
    df: pd.DataFrame,
    lookback: int = 1
) -> pd.DataFrame:
    """
    Frames the problem into a supervised binary classification problem.
    Args:
        lookback: the number of days looked back while constructing trend.
        trend[t] := closePrice[t] - closePrice[t - lookback]
    """
    shift = pd.Timedelta(-lookback, unit="d")
    pc = pd.Series(index=df.index)
    for i, day in enumerate(df.index):
        if i < lookback:
            continue
        base = df.index[i - lookback]
        price_change = df["Adj Close"][day] - df["Adj Close"][base]
        pc[day] = price_change
    # binary label
    bin_label = pd.Series(index=df.index)
    bin_label[pc >= 0] = 1.0
    bin_label[pc < 0] = 0
    # ternary label
    ter_label = pd.Series(index=df.index)
    ter_label[pc == 0] = 0
    ter_label[pc > 0] = 1
    ter_label[pc < 0] = -1
    merged = pd.concat([
        df,
        pd.DataFrame(bin_label, columns=["bin_label"]),
        pd.DataFrame(ter_label, columns=["ter_label"])
    ], axis=1)
    return merged


def gen_lagged_values(
    df: pd.DataFrame,
    lookback: int = 5
) -> pd.DataFrame:
    """
    Generates historical values for the supervised learning problem.
    lookback:
        df[t-lookback] ~ df[t-1] are used to predict df[t]
    """
    raise NotImplementedError


if __name__ == "__main__":
    # For debugging purpose.
    pass
