from binn import BINN, Network, SuperLogger, BINNExplainer, ImportanceNetwork;
import pandas as pd;
from sklearn import preprocessing;
import numpy as np;
from typing import Union;
import torch;
from pytorch_lightning import Trainer
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib


def fit_data_matrix_to_network_input(data_matrix: pd.DataFrame, features, feature_column="Protein") -> pd.DataFrame:
    nr_features_in_matrix = len(data_matrix.index)
    if len(features) > nr_features_in_matrix:
        features_df = pd.DataFrame(features, columns=[feature_column])
        data_matrix = data_matrix.merge(
            features_df, how='right', on=feature_column)
    if len(features) > 0:
        data_matrix.set_index(feature_column, inplace=True)
        data_matrix = data_matrix.loc[features]
    return data_matrix


def generate_data(data_matrix: pd.DataFrame, design_matrix: pd.DataFrame):
    GroupOneCols = design_matrix[design_matrix['group']
                                 == 0]['sample'].values
    GroupTwoCols = design_matrix[design_matrix['group']
                                 == 1]['sample'].values

    df1 = data_matrix[GroupOneCols].T
    df2 = data_matrix[GroupTwoCols].T
    y = np.array([0 for _ in GroupOneCols] + [1 for _ in GroupTwoCols])
    X = pd.concat([df1, df2]).fillna(0).to_numpy()
    #X = preprocessing.StandardScaler().fit_transform(X)
    return X, y


input_data = pd.read_csv("./data/maf_data.txt", sep="\t");
translation = pd.read_csv("./data/reactome_data.txt", sep="\t");
pathways = pd.read_csv("./data/pathways.tsv", sep="\t");
network = Network(input_data=input_data, pathways=pathways, mapping=translation, input_data_column="Protein",
                  source_column="child", target_column="parent")
torch.manual_seed(666)
binn = BINN(network=network, n_layers=4, dropout=0.5, validate=True, residual=False, learning_rate=0.001)
design_matrix = pd.read_csv('./data/sample_data.txt', sep="\t")
protein_matrix = fit_data_matrix_to_network_input(input_data, features=binn.features)
X, y = generate_data(protein_matrix, design_matrix=design_matrix)
dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
torch.manual_seed(666)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=27, shuffle=True, drop_last=False, num_workers=0)

optimizer = binn.configure_optimizers()[0][0]
num_epochs = 30
for epoch in range(num_epochs):
    binn.train()
    total_loss = 0.0
    total_accuracy = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(binn.device)
        targets = targets.to(binn.device).type(torch.LongTensor)
        optimizer.zero_grad()
        outputs = binn(inputs).to(binn.device)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += torch.sum(torch.argmax(outputs, axis=1) == targets) / len(targets)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f'Epoch {epoch}, Average Accuracy {avg_accuracy}, Average Loss: {avg_loss}')