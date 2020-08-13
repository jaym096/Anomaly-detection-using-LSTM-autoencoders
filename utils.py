import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from arff2pandas import a2p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(filename):
    with open(filename) as f:
        train = a2p.load(f)
    return train

def plot_time_series_class(data, class_name, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.125
  )
  ax.set_title(class_name)


def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  print("shape of dataset: ", dataset[0].shape)
  print("sequence: ", len(sequences[0]))
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features


def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses