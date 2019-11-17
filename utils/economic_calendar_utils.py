from typing import List

import numpy as np
import pandas as pd

import sys
sys.path.append("./")

import CONSTANTS

import tqdm

"""
DATA desc
column - Event date
column - Event time (time New York)
column - Country of the event
column - The degree of volatility (possible fluctuations in currency, indices, etc.) caused by this event
column - Description of the event
column - Evaluation of the event according to the actual data, which came out better than the forecast, worse or correspond to it
column - Data format (%, K x103, M x106, T x109)
column - Actual event data
column - Event forecast data
column - Previous data on this event (with comments if there were any interim changes).
"""

COLUMNS = [
    "EventDate",
    "EventTime",
    "Country",
    "Volatility",
    "Description",
    "Evaluation",
    "DataFormat",
    "ActualData",
    "ForecastData",
    "PreviousData"
]

DATASET_NAME = "economic_event.csv"


def generate_economic_events(
    file_dir: str = "./",
    save_dir: str = None
) -> pd.DataFrame:
    # Load dataset
    # Note; the first file is comma-separated, the other two are column-separated.
    cluster1 = [pd.read_csv(file_dir + "D2011-13.csv", sep=",", names=COLUMNS)]
    load_file = lambda f: pd.read_csv(f, sep=";", names=COLUMNS)
    cluster2 = list(map(load_file, [file_dir + "D2014-18.csv", file_dir + "D2019.csv"]))
    df_merged = pd.concat(cluster1 + cluster2, axis=0)
    # Parse datetime and group them daily basis.
    df_merged.index = pd.to_datetime(df_merged["EventDate"], format="%Y/%m/%d")
    df_merged.drop(columns=["EventDate"], inplace=True)
    print(df_merged.head())
    print(f"Total number of events loaded: {len(df_merged)}")
    if save_dir is not None:
        print(f"Merged event file is saved to: {save_dir + DATASET_NAME}")
        df_merged.to_csv(save_dir + DATASET_NAME)
    return df_merged


def construct_daily_vm(
    file_dir: str = "./",
    save_dir: str = None
) -> pd.DataFrame:
    # Convert volatility to numerical values.
    df = pd.read_csv(file_dir + DATASET_NAME)
    df["Volatility"].value_counts()
    df["VolatilityMeasure"] = 0
    df["VolatilityMeasure"][df["Volatility"] == "Low Volatility Expected         "] = 1
    df["VolatilityMeasure"][df["Volatility"] == "Moderate Volatility Expected    "] = 2
    df["VolatilityMeasure"][df["Volatility"] == "High Volatility Expected        "] = 3
    # Upsample to daily basis.
    # Direct solution (robust implementation)
    # days = list(set(df["EventDate"]))
    # values = []
    # for d in days:
    #     mask = (df["EventDate"] == d)
    #     values.append(
    #         df[mask]["VolatilityMeasure"].sum()
    #     )
    # df2 = pd.DataFrame(np.array(values), index=[pd.to_datetime(d, format="%Y-%m-%d") for d in days])
    # df2.sort_index(inplace=True)
    daily_vm = df.groupby(["EventDate"]).sum()
    assert len(daily_vm) == len(set(df["EventDate"]))
    if save_dir is not None:
        print(f"Merged event file is saved to: {save_dir + 'daily_vm.csv'}")
        daily_vm.to_csv(save_dir + 'daily_vm.csv')
    return daily_vm



if __name__ == "__main__":
    df = generate_economic_events(
        file_dir=CONSTANTS.DIR_ECONOMIC_CALENDAR_DATA,
        save_dir=CONSTANTS.DIR_ECONOMIC_CALENDAR_DATA
    )
