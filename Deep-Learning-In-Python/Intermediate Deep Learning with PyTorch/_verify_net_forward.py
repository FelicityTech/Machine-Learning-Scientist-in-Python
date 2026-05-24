import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def create_sequences(df, seq_length):
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        x = df.iloc[i : (i + seq_length), 1]
        y = df.iloc[i + seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


df = pd.read_csv("electricity_consump/electricity_train.csv")
X_train, y_train = create_sequences(df, seq_length=24 * 4)
dataset_train = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).float(),
)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

net = Net()
seqs, labels = next(iter(dataloader_train))
seqs = seqs.view(seqs.size(0), seqs.size(1), 1)
out = net(seqs)
print(out.shape)
