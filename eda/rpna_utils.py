"""
Utilities for processing RPNA dataset.
"""
import pandas as pd


def convert_timestamp_wti(
    df: pd.DataFrame,
    utc_col: str = "TIMESTAMP_UTC"
) -> pd.DataFrame:
    """
    Converts UTC time to local time at WTI, Central Standard Time (US)
    """
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    dates_utc = pd.to_datetime(df[utc_col], format=fmt)
    dates_wti = dates_utc.dt.tz_localize("UTC").dt.tz_convert("US/Central")
    df.insert(loc=0, column="TIMESTAMP_WTI", value=dates_wti)
    return df


def aggregate_daily(
    raw: pd.DataFrame,
    attr_col: str,
    date_col: str = "TIMESTAMP_WTI"
) -> pd.DataFrame:
    df = raw.copy()
    dates = df[date_col].dt.strftime("%Y-%m-%d")
    df.insert(loc=0, value=dates, column="DATE")
    mean_ess = pd.DataFrame(
        df.groupby("DATE").mean()[attr_col]
    )
    total_ess = pd.DataFrame(
        df.groupby("DATE").sum()[attr_col]
    )
    num_events = pd.DataFrame(
        df.groupby("DATE").size()
    )
    daily = pd.concat([mean_ess, total_ess, num_events], axis=1)
    daily.columns = ["MEAN_ESS", "TOTAL_ESS", "NUM_EVENTS"]
    daily.index = pd.to_datetime(daily.index, format="%Y-%m-%d")
    return daily


def select_features():
    raise NotImplementedError


def preprocessing(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gathers above methods, the complete pipeline.
    """
    df = raw.copy()
    # convert to wti us central timezone.
    df = convert_timestamp_wti(df)
