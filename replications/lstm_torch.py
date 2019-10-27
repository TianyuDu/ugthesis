"""
A LSTM class built on pytorch.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
plt.style.use("seaborn-dark")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, future=0, y=None):
        outputs = []
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.float32)

        for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]

        # Optional
        # for i in range(future):
        #     if y is not None and random.random() > 0.5:
        #         output = y[:, [i]]  # teacher forcing
        #     h_t, c_t = self.lstm(output, (h_t, c_t))
        #     output = self.linear(h_t)
        #     outputs += [output]
        # outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == "__main__":
    # Random data for testing purpose.
    L = 300  # seq length.
    N = 10  # num observations.
    x = np.array([np.arange(L) for _ in range(N)]).astype(np.float32)
    y = np.sin(x / 10) + np.random.randn(N, L) * 0.5
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    model = LSTM(1, 32, 1)
    y_hat = model(x)
    assert y_hat.shape == y.shape
