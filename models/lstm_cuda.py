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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        lstm_drop_prob: float = 0.5,
        fc_drop_prob: float = 0.5
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
            dropout=lstm_drop_prob,
            batch_first=True  # Only affect input tensor and output tensor
        )

        self.dropout = nn.Dropout(fc_drop_prob)

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
            torch.randn(self.num_layers, batch_size, self.hidden_size).cuda(),
            torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
        )


def train(
    X: np.ndarray,
    y: np.ndarray,
    model_config: dict,
    epoch: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    train_size: float = 0.8,
    shuffle: bool = True,
) -> (nn.Module, Tuple[float], Tuple[np.ndarray]):
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
        lambda z: torch.Tensor(z.astype(np.float32)).cuda(),
        (X_train, X_val, y_train, y_val)
    )

    print(f"X_train @ {X_train.shape}")
    print(f"y_train @ {y_train.shape}")
    print(f"X_val @ {X_val.shape}")
    print(f"y_val @ {y_val.shape}")

    # Construct model.
    model = StackedLstm(**model_config)
    model = model.cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    batch_index_lst = train_utils.batch_sampler(
        batch_size=batch_size,
        data_size=X_train.shape[0]
    )
    print(f"Number of mini-batches: {len(batch_index_lst)} with batch size {batch_size}")

    for e in range(1, epoch + 1):
        for (low, high) in batch_index_lst:
            seq = X_train[low: high, :, :]
            lab = y_train[low: high]
            # lab = lab.reshape(-1, 1)
            optimizer.zero_grad()
            # Initialize hidden states and cell states.
            model.reset_hidden(batch_size=lab.shape[0])
            y_pred = model(seq)
            batch_loss = loss_function(y_pred.view(-1, 1), lab.view(-1, 1))
            batch_loss.backward()
            optimizer.step()
            train_acc = directional_accuracy(
                lab.cpu().detach().numpy(),
                y_pred.cpu().detach().numpy()
            )
            train_mape = mape(lab.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        print(f"epoch: {e: 3} train loss: {batch_loss.item(): 10.8f}, DA: {train_acc * 100: 2.1f}%, mape: {train_mape: 2.1f}%")
        # validation
        if e % 3 == 0:
            with torch.no_grad():
                model.reset_hidden(batch_size=y_val.shape[0])
                val_pred = model(X_val)
                val_loss = loss_function(val_pred.view(-1, 1), y_val.view(-1, 1))
                val_acc = directional_accuracy(
                    y_val.cpu().detach().numpy(),
                    val_pred.cpu().detach().numpy()
                )
                val_mape = mape(
                    y_val.cpu().detach().numpy(),
                    val_pred.cpu().detach().numpy()
                )
            print(f"[Validation] epoch: {e: 3} val loss: {val_loss.item(): 10.8f}, DA: {val_acc * 100: 2.1f} %, mape: {val_mape: 2.1f}%")
    return model, (val_loss, val_acc, val_mape), (X_train, X_val, y_train, y_val)


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


def main(config: dict) -> str:
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

    # Transform data.
    scalers = {}
    for i in range(X_train.shape[-1]):
        scalers[i] = StandardScaler()
        X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])

    for i in range(X_test.shape[-1]):
        X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])

    model_config = dict(
        input_size=X_train.shape[-1],
        hidden_size=int(config["nn.hidden_size"]),
        output_size=int(config["nn.output_size"]),
        num_layers=int(config["nn.num_layer"]),
        lstm_drop_prob=float(config["nn.lstm_drop_prob"]),
        fc_drop_prob=float(config["nn.fc_drop_prob"]),
    )

    model, (val_mse, val_acc, val_mape), _ = train(
        X_train, y_train,
        model_config=model_config,
        epoch=int(config["train.epoch"]),
        batch_size=int(config["train.batch_size"]),
        lr=float(config["train.lr"]),
        train_size=0.7
    )
    model.reset_hidden(batch_size=X_test.shape[0])
    pred_test = model(torch.Tensor(X_test.astype(np.float32)).cuda()).cpu().detach().numpy().squeeze()
    test_mse = mse(y_test, pred_test)
    test_acc = directional_accuracy(y_test, pred_test)
    test_mape = mape(y_test, pred_test)

    report = f"{val_mse}\t{val_acc}\t{val_mape}\t{test_mse}\t{test_acc}\t{test_mape}"
    return report


def sample_config(config_scope: dict) -> dict:
    """
    Randomly sample a configuration from the config_scope.
    """
    sampled = dict()
    for k, v in config_scope.items():
        sampled[k] = np.random.choice(v)
    return sampled


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    config_scope = {
        "nn.hidden_size": [32, 64, 128, 256, 512, 1024],
        "nn.output_size": [1],
        "nn.num_layer": [1, 2, 3],
        "nn.lstm_drop_prob": [0.0, 0.25, 0.5],
        "nn.fc_drop_prob": [0.0, 0.25, 0.5],
        "train.epoch": list(range(5, 20)) + [5 * x for x in range(4, 41)],
        "train.batch_size": [32, 128, 512],
        "train.lr": [10**(-x) for x in range(1, 6)] + [3 * 10**(-x) for x in range(1, 6)],
    }

    # A smaller configuration scope for debugging purpose
    # config_scope = {
    #     "nn.hidden_size": [32, 64],
    #     "nn.output_size": [1],
    #     "nn.num_layer": [1],
    #     "nn.drop_prob": [0.0, 0.25, 0.5],
    #     "train.epoch": [5, 6, 7],
    #     "train.batch_size": [32, 128, 512],
    #     "train.lr": [10**(-x) for x in range(1, 6)] + [3 * 10**(-x) for x in range(1, 6)],
    # }

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="./lstm_result.txt")
    args = parser.parse_args()

    start_time = datetime.now()
    with open(args.log_dir, "w") as f:
        f.write("sample_id\tval_mse\tval_acc\tval_mape\ttest_mse\ttest_acc\ttest_mape\tconfig\n")
        for i in range(args.N):
            config = sample_config(config_scope)
            print("=======================================================")
            print(f"======     Current Round Config: {i + 1} out of {args.N}     ======")
            print("=======================================================")
            print(config)
            repr_str = main(config)
            repr_str_extended = f"{i + 1}\t" + repr_str + "\t" + str(config) + "\n"
            f.write(repr_str_extended)
    print(f"Training {args.N} random profiles, time taken{datetime.now() - start_time}")
