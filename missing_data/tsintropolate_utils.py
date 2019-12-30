import numpy as np
import pandas as pd

from typing import Tuple

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


def arima_intropolate(
    raw: pd.DataFrame,
    arima_order: Tuple[int]
) -> pd.DataFrame:
    """
    Intropolate time series using an ARIMA model.
    Args:
    raw:
        Raw dataset.
    arima_order:
        p, d, q parameters for ARIMA model.
    """
    df = raw.copy()
    # TODO: check blog and complete this section.
    model = ARIMA(history, order=arima_order)
