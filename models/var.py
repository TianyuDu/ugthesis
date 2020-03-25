"""
Vector Autocorrelation.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

import data_feed


df_returns = data_feed.DF_RETURNS.copy()
