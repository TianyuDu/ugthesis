import numpy as np
import pandas as pd

from typing import Tuple, Union

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

from tqdm import tqdm


def arima_interpolate(
    raw: pd.DataFrame,
    arima_order: Tuple[int],
    verbose: bool = True,
    cache_dir: Union[str, None] = "./_arima_interpolate_cache.csv"
) -> pd.DataFrame:
    """
    Intropolate time series using an ARIMA model.
    Args:
    raw:
        Raw dataset.
    arima_order:
        p, d, q parameters for ARIMA model.
    verbose:
        whether to print information.
    cache_dir:
        save intermediate results, since the interpolation process tends to be time-consuming.
    """
    df = raw.copy().astype(np.float32)
    assert len(df.index) == len(set(df.index)), "Non-unique index detected."
    print(f"Interpolate using ARIMA({arima_order})")
    # Interpolate first few datapoints directly.
    rg = arima_order[0] * 100
    num_missing = np.sum(np.isnan(df.iloc[:rg, :].values))
    if verbose:
        print(f"Spline (order=5) interpolation range: first {rg} observations, total {num_missing} missing values.")
    df_filled = df.iloc[:rg, :].interpolate(
        method="spline",
        order=5
    )
    # The first or last element, because they cannot be interpolated using spline, use the nearest value.
    fail_to_fill = np.sum(np.isnan(df_filled.values))
    if verbose:
        print(f"Number of missing values failed to be filled using spline: {fail_to_fill}")
        print(f"Replace using forward + backward fillinng.")
    df_filled.fillna(method="ffill", inplace=True)
    df_filled.fillna(method="bfill", inplace=True)
    assert not any(np.isnan(df_filled.values))
    df.iloc[:rg, :] = df_filled
    # Interpolate using arima.
    succ_counter = 0
    err_counter = 0
    total = len(df.index)
    # for i, (t, v) in tqdm(enumerate(zip(df.index, df.values))):
    for i in tqdm(range(total)):
        t, v = df.index[i], df.values[i]
        # print(f"{i}/{total}")
        if np.isnan(v.item()):
            try:
                succ_counter += 1
                # Interpolate if this point is null.
                history = df[df.index < t].values.squeeze()
                assert not any(np.isnan(history)),\
                    f"All previous values at {t} should be filled, nan detected! Check if implemented correctly."
                model = ARIMA(history, order=arima_order)
                model_fit = model.fit(disp = 1 if verbose else 0)
                pred = model_fit.forecast(steps=1)
                yhat = pred[0]  # pred = (yhat, std, (low_ci, high_ci))
                # Insert interpolated value
                df[df.index == t] = yhat
            except np.linalg.LinAlgError:
                err_counter += 1
                # Use forward filling.
                df[df.index == t] = df.iloc[i - 1, :].values.item()
    print(
        f"Number of missing values filled using ARIMA({arima_order}): {succ_counter}.")
    print(f"Number of missing values cannot be filled using ARIMA, used forward filling instead: {err_counter}.")
    if cache_dir is not None:
        df.to_csv(cache_dir)
    _check_output(df)
    return df


def _check_output(df: pd.DataFrame) -> None:
    if np.sum(np.isnan(df.values)) > 0:
        raise Exception("Dataframe is not completely filled.")
    return None
