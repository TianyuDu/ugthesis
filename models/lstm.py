"""
Models of LSTM family.
"""
import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

import utils.time_series_utils as ts_utils


class BasicLstm(nn.Module):
    def __init__(self):
        super(BasicLstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x):
        out, (h_n, h_n) = None


def preprocessing(
    df: pd.DataFrame,
):
    """
    Normalizes the dataset.
    """
    data = df.values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler


def train(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7
):
    raise NotImplementedError


def predict():
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    print(f"Load dataset from: {args.data_dir}")
    df = pd.read_csv(
        args.data_dir,
        index_col=0,
        header=0,
        parse_dates=["DATE"],
        date_parser=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )
    data_normalized, scaler = preprocessing(df)
    X, y = ts_utils.create_inout_sequences(
        input_data=data_normalized
    )