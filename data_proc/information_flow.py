"""
April 2, 2020
Extract information from information flow.
"""
import numpy as np
import pandas as pd


def extract_IF(
    IF: pd.dataframe
) -> dict:
    """
    Extract meaningful features from the information flow.
    An information flow is a subset (rows) of the RPNA dataset.
    """
    IF = IF.copy()
    X = dict()

    for fea, val in IF["ESS"].describe().items():
        X["ess_" + fea] = val

    IF["WESS"] = IF["ENS"] * IF["ESS"] / 100.0
    for fea, val in IF["WESS"].describe().items():
        X["wess_" + fea] = val

    
