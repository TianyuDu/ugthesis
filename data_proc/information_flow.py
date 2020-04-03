"""
April 2, 2020
Extract information from information flow.
"""
import numpy as np
import pandas as pd


def extract_IF(
    IF: pd.DataFrame
) -> dict:
    """
    Extract meaningful features from the information flow.
    An information flow is a subset (rows) of the RPNA dataset.
    """
    IF = IF.copy()
    IF["ESS"] = IF["ESS"] - 50
    IF["ESS"].fillna(value=0.0, inplace=True)
    X = dict()

    # For all events
    for fea, val in IF["ESS"].describe().items():
        X["ess_" + fea] = val

    for fea, val in (IF["ESS"] ** 2).describe().items():
        X["ess_squared_" + fea] = val

    IF["WESS"] = IF["ENS"] * IF["ESS"] / 100.0
    for fea, val in IF["WESS"].describe().items():
        X["wess_" + fea] = val

    for fea, val in (IF["WESS"] ** 2).describe().items():
        X["wess_squared_" + fea] = val

    # Seperate positive and negative events.
    POS = IF[IF["ESS"] > 0]
    NEG = IF[IF["ESS"] < 0]
    for label, info in zip(["pos_", "neg_"], [POS, NEG]):
        X[label + "count"] = len(info)
        X[label + "ess_mean"] = np.mean(info["ESS"])
        X[label + "wess_mean"] = np.mean(info["WESS"])

    # For extreme events
    IF_POS = IF[IF["ESS"] > 18]
    IF_NEG = IF[IF["ESS"] < -15]
    for label, info in zip(["ex_pos_", "ex_neg_"], [IF_POS, IF_NEG]):
        X[label + "count"] = len(info)
        X[label + "ess_mean"] = np.mean(info["ESS"])
        X[label + "wess_mean"] = np.mean(info["WESS"])
    return X
