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
