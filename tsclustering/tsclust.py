"""
Time series clustering.
"""
import pandas as pd
from tsclust_utils import *

if __name__ == "__main__":
    df_wti_real.dropna().to_csv("../data/ready_to_use/wti_crude_oil_price_real.csv")
    df_wti_return.dropna().to_csv("../data/ready_to_use/wti_crude_oil_return_real.csv")
