import sys
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def monthly_subsequence(
    df: pd.DataFrame,
    statistic: callable = series_summary
) -> (Dict[str, pd.DataFrame], pd.DataFrame):
    """
    Groups the entire series to monthly subsequences.
    then creates features describing the monthly subsequence,
    these features can be used for time series clustering.
    """
    yr_range = sorted(set(df.index.year))
    mo_range = sorted(set(df.index.month))
    subsequences = dict()
    stats = list()
    for yr in yr_range:
        for mo in mo_range:
            mask = np.logical_and(df.index.year == yr, df.index.month == mo)
            str_mo = "0" + str(mo) if len(str(mo)) == 1 else str(mo)
            time = str(yr) + "-" + str_mo
            subseq = df[mask]
            if len(subseq) > 0:
                subsequences[time] = subseq
                # Summary statistic
                stats_month = statistic(subseq)
                stats_month["Date"] = [time]
                stats.append(stats_month)
    stats_all = pd.concat(stats, axis=0)
#     stats_all.reset_index(inplace=True, drop=True)
    stats_all.index = pd.to_datetime(stats_all["Date"], format="%Y-%m")
    stats_all.drop(columns=["Date"], inplace=True)
    return (subsequences, stats_all)


def time_series_clustering(
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clusters months into clusters using created features.
    """
    raise NotImplementedError
