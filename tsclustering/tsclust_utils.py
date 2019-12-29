import sys
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _series_summary(
    series: pd.DataFrame,
    acf_lags: int = 5,
    pacf_lags: int = 5
) -> pd.DataFrame:
    """
    Extracts features for time series data.
    This function should not be called directly.
    """
    collection = pd.DataFrame()
    collection["mean"] = np.mean(series).values
    collection["std"] = np.std(series).values
    collection["median"] = np.std(series).values

    # Use acf and pacf to characterize the dynamics of series
    acf = pd.DataFrame(
        sm.tsa.stattools.acf(series, nlags=acf_lags)[1:].reshape(1, -1),
        columns=[f"acf_lag_{i}" for i in range(1, acf_lags + 1)]
    )
    pacf = pd.DataFrame(
        sm.tsa.stattools.pacf(series, nlags=pacf_lags)[1:].reshape(1, -1),
        columns=[f"pacf_lag_{i}" for i in range(1, pacf_lags + 1)]
    )
    collection = pd.concat([collection, acf, pacf], axis=1)
    return collection


def monthly_subsequence(
    df: pd.DataFrame,
    statistic: callable = _series_summary
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
    n_clusters: int = 3,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Clusters months into clusters using created features.
    However, standardizing procedures are executed before
    clustering, standardization/scaling is necessary for k-mean
    but unnecessary for other methods including Gaussian mixture.

    Returns the clustered label dataframe, which uses the same indices
    as df_features.
    """
    if normalize:
        # Normalize features.
        scaler = StandardScaler()
        norm_fea = scaler.fit_transform(df_features.values)
    else:
        norm_fea = df_features.values

    gmm = GaussianMixture(n_components=n_clusters)
    gmm_labels = gmm.fit_predict(norm_fea)
    df_gmm_labels = pd.DataFrame(
        data={"label": gmm_labels},
        index=df_features.index
    )
    return df_gmm_labels
