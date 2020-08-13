import torch
import utils
import numpy as np
from torch import nn
import copy
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import RecurrentAutoencoder

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

rcParams['figure.figsize'] = 12, 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history


if __name__ == "__main__":
    train_data = utils.load_data('ECG5000/ECG5000_TRAIN.arff')
    test_data = utils.load_data('ECG5000/ECG5000_TEST.arff')

    # create a single dataframe
    df = train_data.append(test_data)
    df = df.sample(frac=1.0)

    class_names = ['Normal', 'R on T', 'PVC', 'SP', 'UB']

    new_columns = list(df.columns)
    new_columns[-1] = 'target'
    df.columns = new_columns

    print("*** exploring data ***")
    print(df.head(n=5))
    print(df.target.value_counts())

    print("*** save plots ***")
    plt.bar(class_names, df.target.value_counts())
    plt.savefig("results/plot1.png")

    classes = df.target.unique()

    fig, axs = plt.subplots(
        nrows=len(classes) // 3 + 1,
        ncols=3,
        sharey=True,
        figsize=(14, 8)
    )

    for i, cls in enumerate(classes):
        ax = axs.flat[i]
        data = df[df.target == cls] \
            .drop(labels='target', axis=1) \
            .mean(axis=0) \
            .to_numpy()
        utils.plot_time_series_class(data, class_names[i], ax)

    fig.delaxes(axs.flat[-1])
    fig.tight_layout()
    fig.savefig("results/plot2.png")

    CLASS_NORMAL = 1
    # take normal examples into one dataframe
    normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
    print("shape: ", normal_df.shape)

    # rest of the classes, we'll consider anomalies and will make them into one dataframe
    anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)
    print("shape: ", anomaly_df.shape)

    # make train, test, and validate splits
    train_df, val_df = train_test_split(
        normal_df,
        test_size=0.15,
        random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        val_df,
        test_size=0.33,
        random_state=RANDOM_SEED
    )

    train_dataset, seq_len, n_features = utils.create_dataset(train_df)
    val_dataset, _, _ = utils.create_dataset(val_df)
    test_normal_dataset, _, _ = utils.create_dataset(test_df)
    test_anomaly_dataset, _, _ = utils.create_dataset(anomaly_df)

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model = model.to(device)
    
    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        n_epochs=150
    )
    
    print("*** finished training ***")
    
    MODEL_PATH = 'model.pth'
    torch.save(model, MODEL_PATH)

#     !gdown --id 1jEYx5wGsb7Ix8cZAw3l5p5pOwHs3_I9A
#     model = torch.load('model.pth')
#     model = model.to(device)

    # clear the plot
    plt.clf()

    # we can look at the reconstruction loss on the training data.
    # Then we can decide on a threshold and turn the problem into simple
    # binary classification. If reconstruction loss > threshold the data
    # point is an anomaly else it is a normal heartbeat.
    _, losses = utils.predict(model, train_dataset)
    sns.distplot(losses, bins=50, kde=True)
    plt.grid()
    plt.title("reconstruction loss for training set")
    plt.xlabel("loss")
    plt.ylabel("Probability density")
    plt.savefig("results/plot3.png")

    # based on the plot we decide on the threshold
    THRESH = 26

    plt.clf()

    predictions, pred_losses = utils.predict(model, test_normal_dataset)
    sns.distplot(pred_losses, bins=50, kde=True)
    plt.grid()
    plt.title("reconstruction loss on test set (normal heartbeats)")
    plt.xlabel("loss")
    plt.ylabel("Probability density")
    plt.savefig("results/plot4.png")

    plt.clf()

    correct = sum(l <= THRESH for l in pred_losses)
    print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

    # make the length of anomaly test data set equal to the length of test normal data set
    anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

    predictions2, pred_losses = utils.predict(model, anomaly_dataset)
    sns.distplot(pred_losses, bins=50, kde=True)
    plt.grid()
    plt.title("reconstruction loss on test set (anomaly heartbeats)")
    plt.xlabel("loss")
    plt.ylabel("Probability density")
    plt.savefig("results/plot5.png")

    correct = sum(l > THRESH for l in pred_losses)
    print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')
