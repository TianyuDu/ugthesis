# Formatting stock data, generate the dataset for supervised learning problem.
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    columns: List[str],
    lookback: int = 5
) -> pd.DataFrame:
    """
    Generates historical values for the supervised learning problem.
    lookback:
        df[t-lookback] ~ df[t-1] are used to predict df[t]
    """
    # ==== Check ====
    assert np.all([c in df.columns for c in columns]), "Column requested not found in data frame."
    # ==== Core ====
    new_features = []
    days = df.index
    for col in columns:
        new_columns = [f"{col}_lag{x + 1}" for x in range(lookback)]
        features = pd.DataFrame(columns=new_columns, index=days, dtype=np.float64)
        for i, day in tqdm(enumerate(days)):
            if i < lookback:
                continue
            features.loc[day, :] = [
                df.loc[days[i - lag - 1], col]
                for lag in range(lookback)
            ]
        new_features.append(features)
    df_result = pd.concat([df] + new_features, axis=1)
    return df_result


if __name__ == "__main__":
    # For debugging purpose.
    df = load_local_dataset("AAPL")
    df = gen_classification_label(df)
    df = gen_lagged_values(df, columns=["Adj Close"], lookback=5)
