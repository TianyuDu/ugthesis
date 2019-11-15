from typing import List

import numpy as np
import pandas as pd

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
    "EventDate", "EventTime",
    "Country", "Volatility", "Description", 
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
    # Parse datetime (daily basis)
    df_merged.index = pd.to_datetime(df_merged["Event date"], format="%Y/%m/%d")
    print(f"Total number of events loaded: {len(merged)}")
    if save_dir is not None:
        df_merged.to_csv(save_dir + DATASET_NAME)
    return df_merged


def load_economic_events(file_dir: str = "./") -> pd.DataFrame:
    df = pd.read_csv(file_dir + DATASET_NAME)
    

def extract_info() -> pd.DataFrame:
    df = load_economic_events()


if __name__ == "__main__":
    df = parser()