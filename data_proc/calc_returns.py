"""
Calculate effective daily returns.
"""
import numpy as np
import pandas as pd
from datetime import datetime


def calculate_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    assert "DCOILWTICO" in df.columns and "DELTA" in df.columns
    filtered = df["DCOILWTICO"].dropna()
    returns = np.log(filtered).diff()
    returns = returns.rename("RETURN")
    # Change to business day frequency.
    returns = returns.asfreq("B")
    returns = pd.DataFrame(returns)
    # normalize return using delta values.
    delta = df["DELTA"][returns.index]
    returns = pd.DataFrame(returns["RETURN"] / delta * 100)  # Change to percentage returns.
    returns.columns = ["RETURN"]
    return returns


if __name__ == "__main__":
    DATA_DIR = "/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use"
    df = pd.read_csv(
        DATA_DIR + "/DCOILWTICO_with_delta.csv",
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        index_col=0
    )

    returns = calculate_returns(df)
    print("Save returns to: " + DATA_DIR + "/returns_norm.csv")
    returns.to_csv(DATA_DIR + "/returns_norm.csv")
