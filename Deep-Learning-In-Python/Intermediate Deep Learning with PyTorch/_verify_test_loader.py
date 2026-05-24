import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_sequences(df, seq_length):
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        x = df.iloc[i : (i + seq_length), 1]
        y = df.iloc[i + seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


df_test = pd.read_csv("electricity_consump/electricity_test.csv")
X_test, y_test = create_sequences(df_test, seq_length=24 * 4)
dataset_test = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test).float(),
)

test_loader = DataLoader(
    dataset_test,
    batch_size=32,
    shuffle=False,
)

for seqs, labels in test_loader:
    print(labels.shape)
    break
