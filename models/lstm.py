"""
Models of LSTM family.
"""
import argparse
import sys
from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

sys.path.append("../")
sys.path.append("./")

import utils.time_series_utils as ts_utils
import utils.training_utils as train_utils


class StackedLstm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        drop_prob: float = 0.5
    ) -> None:
        super(StackedLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # dropout=drop_prob,
            batch_first=True  # Only affect input tensor and output tensor
        )

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.hidden_cell = (None, None)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            # input_seq.view(len(input_seq) ,1, -1),
            # input of shape (batch, seq_len, input_size)
            input_seq,
            self.hidden_cell
        )
        # lstm output of shape (batch, seq_len, num_directions * hidden_size)
        out = self.dropout(lstm_out)
        pred = self.fc(out)
        # pred of shape (batch, seq_len, output_size)
        return pred[:, -1, :]

    def reset_hidden(self, batch_size) -> None:
        # both hidden h and cell c.
        self.hidden_cell = (
            torch.randn(self.num_layers, batch_size, self.hidden_size),
            torch.randn(self.num_layers, batch_size, self.hidden_size)
        )


def preprocessing(
    df: pd.DataFrame,
) -> (np.ndarray, "MinMaxScaler"):
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
    model_config: dict,
    epoch: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    train_size: float = 0.7,
    shuffle: bool = False,
) -> (nn.Module, Tuple[np.ndarray]):
    """
    # TODO: docstring :P
    """
    # Split dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_size,
        shuffle=shuffle
    )
    X_train, X_val, y_train, y_val = map(
        lambda z: torch.Tensor(z.astype(np.float32)),
        (X_train, X_val, y_train, y_val)
    )
    # Construct model.
    model = StackedLstm(**model_config)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    batch_index_lst = train_utils.batch_sampler(
        batch_size=batch_size,
        data_size=X_train.shape[0]
    )

    for e in range(epoch):
        for (low, high) in batch_index_lst:
            seq = X_train[low: high, :, :]
            lab = y_train[low: high, :, :]
            lab = lab.reshape(-1, 1)
            optimizer.zero_grad()
            # Initialize hidden states and cell states.
            model.reset_hidden(batch_size=lab.shape[0])
            y_pred = model(seq)
            batch_loss = loss_function(y_pred, lab)
            batch_loss.backward()
            optimizer.step()
        # validation
        if e % 5 == 1:
            with torch.no_grad():
                model.reset_hidden(batch_size=y_val.shape[0])
                val_pred = model(X_val)
                y_val = y_val.reshape(-1, 1)
                val_loss = loss_function(val_pred, y_val)
            print(f"epoch: {e: 3} train loss: {batch_loss.item(): 10.8f} validation loss: {val_loss.item(): 10.8f}")
    return model, (X_train, X_val, y_train, y_val)


def predict(
    model: nn.Module,
    data: Tuple[np.ndarray] = "X_train, X_val, y_train, y_val",
    report_str: bool = False,
    log_dir: Union[str, None] = None
) -> Union[str, None]:
    X_train, X_val, y_train, y_val = data
    # TODO: stopped here.

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
        input_data=data_normalized,
        lags=365
    )

    model_config = dict(input_size=1,
                        hidden_size=128,
                        output_size=1,
                        num_layers=2,
                        drop_prob=0.5)

    train(
        X, y,
        model_config=model_config
    )
