import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def predict():
    raise NotImplementedError
