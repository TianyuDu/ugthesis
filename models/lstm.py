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

from data_feed import rnn_feed
from training_utils import directional_accuracy, mape, mse

sys.path.append("../")
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
    train_size: float = 0.8,
    shuffle: bool = True,
) -> (nn.Module, Tuple[np.ndarray]):
    """
    Training the LSTM model.
    """
    # Split dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_size,
        shuffle=shuffle
    )
    # Convert to tensors.
    X_train, X_val, y_train, y_val = map(
        lambda z: torch.Tensor(z.astype(np.float32)),
        (X_train, X_val, y_train, y_val)
    )

    print(f"X_train @ {X_train.shape}")
    print(f"y_train @ {y_train.shape}")
    print(f"X_val @ {X_val.shape}")
    print(f"y_val @ {y_val.shape}")

    # Construct model.
    model = StackedLstm(**model_config)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    batch_index_lst = train_utils.batch_sampler(
        batch_size=batch_size,
        data_size=X_train.shape[0]
    )
    print(f"Number of mini-batches: {len(batch_index_lst)} with batch size {batch_size}")

    for e in range(epoch):
        for (low, high) in batch_index_lst:
            seq = X_train[low: high, :, :]
            lab = y_train[low: high]
            lab = lab.reshape(-1, 1)
            optimizer.zero_grad()
            # Initialize hidden states and cell states.
            model.reset_hidden(batch_size=lab.shape[0])
            y_pred = model(seq)
            batch_loss = loss_function(y_pred, lab.view(-1,))
            batch_loss.backward()
            optimizer.step()
            train_acc = directional_accuracy(
                lab.detach().numpy(),
                y_pred.detach().numpy()
            )
            train_mape = mape(lab.detach().numpy(), y_pred.detach().numpy())
        print(f"epoch: {e: 3} train loss: {batch_loss.item(): 10.8f}, DA: {train_acc * 100: 2.1f}%, mape: {train_mape: 2.1f}%")
        # validation
        if e % 5 == 1:
            with torch.no_grad():
                model.reset_hidden(batch_size=y_val.shape[0])
                val_pred = model(X_val)
                y_val = y_val.reshape(-1, 1)
                val_loss = loss_function(val_pred, y_val.view(-1,))
                val_acc = directional_accuracy(
                    y_val.detach().numpy(),
                    val_pred.detach().numpy()
                )
                val_mape = mape(
                    y_val.detach().numpy(),
                    val_pred.detach().numpy()
                )
            print(f"[Validation] epoch: {e: 3} val loss: {val_loss.item(): 10.8f}, DA: {val_acc * 100: 2.1f} %, mape: {val_mape: 2.1f}%")
    return model, (X_train, X_val, y_train, y_val)


def predict(
    model: nn.Module,
    data: Tuple[np.ndarray] = "X_train, X_val, y_train, y_val",
    report_str: bool = False,
    log_dir: Union[str, None] = None
) -> Union[str, None]:
    """
    Generates out-of-sample prediction.
    """
    # Load the dataset.
    X_train, X_val, y_train, y_val = data
    # TODO: Stopped here


def main():
    src = "../data/ready_to_use/xrt/"
    X_train, X_test, y_train, y_test = rnn_feed(
        src=src,
        test_start=pd.to_datetime("2019-01-01")
    )

    print(f"X_train @ {X_train.shape}")
    print(f"y_train @ {y_train.shape}")
    print(f"X_test @ {X_test.shape}")
    print(f"y_test @ {y_test.shape}")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    model_config = dict(
        input_size=X_train.shape[-1],
        hidden_size=32,
        output_size=1,
        num_layers=1,
        drop_prob=0.5
    )

    model, (X_train, X_val, y_train, y_val) = train(
        X, y,
        model_config=model_config,
        epoch=250,
        batch_size=512,
        lr=0.0003,
        train_size=0.8
    )


if __name__ == "__main__":
    # main()
    raise NotImplementedError
