import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init
from torchsummary import summary

import tqdm


class BasicLSTM(nn.Module):
    """Basic LSTM binary classification model with internal dropout"""
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        # Dropout layer
        self.input_dropout = nn.Dropout(0.5)
        self.hidden_dropout = nn.Dropout(0.0)
        self.output_dropout = nn.Dropout(0.0)
        # Classification layers
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax()

    def forward(self, inputs, future=0, y=None):
        outputs = []
        # Input shape: (batch_size, sequence_length, input_size)
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        batch_size = inputs.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        # h_t = torch.randn(batch_size, self.hidden_size, dtype=torch.float32)
        # c_t = torch.randn(batch_size, self.hidden_size, dtype=torch.float32)

        for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):
            input_t = self.input_dropout(input_t)
            # Drop the time dimension.
            input_t = input_t.view(input_t.shape[0], input_t.shape[-1])
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            h_t = self.hidden_dropout(h_t)
            output = self.linear(h_t)
            output = self.softmax(output)
            outputs += [output]

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        final_output = outputs[:, -1, :].view(batch_size, self.output_size)
        return final_output


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.train_accuracy = []
        self.val_losses = []
        self.val_accuracy = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        if batch_size == -1:
            return x, y, 1
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i: i + batch_size]
            y_batch = y[i: i + batch_size]
            yield x_batch, y_batch, batch

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=32,
        n_epochs=15
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []
            batch_train_losses = []
            batch_accuracies = []
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                y_pred = self.model(x_batch)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                batch_train_losses.append(loss.item())
                # Accuracy.
                bin_y_pred = torch.argmax(y_pred, axis=1)
                accuracy = torch.mean((bin_y_pred == y_batch).double())
                batch_accuracies.append(accuracy.item())
#             self.scheduler.step()
            train_loss = np.mean(batch_train_losses)
            self.train_losses.append(train_loss)
            accuracy = np.mean(batch_accuracies)
            self.train_accuracy.append(accuracy)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            if x_val is not None:
                print(
                    "Epoch {}, Train CE: {:.6f}, Train Acc: {:3f}. Val CE: {:.6f}, Val Acc: {:3f}. Elapsed time: {:.2f}s.".format(
                        epoch + 1,
                        self.train_losses[-1], self.train_accuracy[-1],
                        self.val_losses[-1], self.val_accuracy[-1],
                        elapsed
                    )
                )
            else:
                print(
                    "Epoch {}, Train CE Loss: {:.6f}, Train Accuracy: {:3f}, Elapsed time: {:.2f}s.".format(
                        epoch + 1, train_loss, accuracy, elapsed
                    )
                )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return None
        with torch.no_grad():
            # Metrics across batches.
            batch_val_losses = []
            batch_accuracies = []
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                batch_val_losses.append(loss.item())
                bin_y_pred = torch.argmax(y_pred, axis=1)
                batch_accuracies.append(torch.mean(
                    ((bin_y_pred) == y_batch).double()
                ).item())
            val_loss = np.mean(batch_val_losses)
            self.val_losses.append(val_loss)
            val_accuracy = np.mean(batch_accuracies)
            self.val_accuracy.append(val_accuracy)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")

    def plot_accuracies(self):
        plt.plot(self.train_accuracy, label="Training accuracy")
        plt.plot(self.val_accuracy, label="Validation accuracy")
        plt.legend()
        plt.title("Accuracies")
