import colorsys
import sys
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-dark")


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
                stats_month["DATE"] = [time]
                stats.append(stats_month)
    stats_all = pd.concat(stats, axis=0)
#     stats_all.reset_index(inplace=True, drop=True)
    stats_all.index = pd.to_datetime(stats_all["DATE"], format="%Y-%m")
    stats_all.drop(columns=["DATE"], inplace=True)
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


def broadcast_monthly_clustering(
    df_data_raw: pd.DataFrame,
    df_label: pd.DataFrame
) -> pd.DataFrame:
    """
    Casts monthly clustering assignments to daily level.
    """
    if "label" in df_data_raw.columns:
        raise KeyError("df should not have label column.")
    df_data = df_data_raw.copy()
    df_data["label"] = -1
    parsed_date_data = np.array(df_data.index.strftime("%Y-%m"), dtype=str)
    parsed_date_label = np.array(df_label.index.strftime("%Y-%m"), dtype=str)
    for t, label in zip(parsed_date_label, df_label.values):
        # Replace the day to 01, so represents the month.
        df_data["label"][parsed_date_data == t] = label
    assert np.all(df_data["label"] != -1), f"labels found: {set(df_data['label'])}"
    return df_data


def regime_plot(
    df: pd.DataFrame,
    labels: np.array,
    plot_col: str,
    color_map: Union[dict, "auto"] = {0: "b", 1: "g", 2: "r", 3: "c", 4: "m", 5: "y", 6: "k", 7: "w"},
    save_dir: Union[str, None] = None
) -> None:
    """
    Plots the time series while using different colors for different
    regimes(labels).
    """
    if color_map == "auto":
        # Automatically create color mapping dictionary.
        num_labels = len(set(labels))
        hsv = np.linspace(0, 360, num_labels + 1)
        color_map = dict()
        for i in range(num_labels):
            color_map[i] = colorsys.hsv_to_rgb(hsv[i] / 360, 1, 1)

    if not (len(df) == len(labels)):
        raise ValueError("labels and dataframe should have the same length.")
    if not (len(color_map) >= len(set(labels))):
        raise ValueError("Color map is not sufficient for labels.")
    for r, current_label in enumerate(set(labels)):
        current_df = df.copy()
        # Replace other label as None, so not plotted.
        mask = (labels == current_label)
        current_df[np.logical_not(mask)] = None
        plt.plot(
            current_df[plot_col],
            color=color_map[current_label],
            linewidth=0.5,
            label=f"{plot_col} cluster #{r}",
            alpha=0.7
        )
    plt.xlabel("Date")
    plt.ylabel(df.columns[0])
    plt.legend()
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir, dpi=300)
        print(f"Output figure is saved to {save_dir}")
