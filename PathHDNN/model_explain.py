import pandas as pd
import os
import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np
from typing import Union;
import torch;
from pytorch_lightning import Trainer
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import joblib
import shap
import zipfile

from tensorflow.python.ops.partitioned_variables import fixed_size_partitioner

os.chdir("Work path")
from PathHDNN import binn,logger,network,importance_network,sklearn,explainer

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
    X = preprocessing.StandardScaler().fit_transform(X)
    return X, y

fixed_path="./data"
os.makedirs(fixed_path, exist_ok=True)
with zipfile.ZipFile("./data/144_train_model.zip", "r") as zip_ref:
    zip_ref.extractall(fixed_path)
binn= joblib.load("./data/train_model.joblib")
input_data = pd.read_csv("./data/maf_data.txt",sep="\t");
design_matrix = pd.read_csv('./data/sample_data.txt',sep="\t")
protein_matrix = fit_data_matrix_to_network_input(input_data, features=binn.features)
X, y = generate_data(protein_matrix, design_matrix=design_matrix)

from binn import BINNExplainer
torch.manual_seed(0)
explainer = BINNExplainer(binn)
test_data = torch.Tensor(X)
background_data = torch.Tensor(X)
torch.manual_seed(0)
importance_df = explainer.explain(test_data, background_data)
'''
column_names=importance_df.columns
header = np.array([column_names], dtype=object)
complete_data = np.vstack((header, importance_df))
with open('importance1_feature_shape.txt', 'w') as f:
    f.write('\t'.join(column_names) + '\n')
np.savetxt('./importance1_feature_shape.txt', complete_data, fmt='%s', delimiter='\t', newline='\n', comments='')
'''
from binn.importance_network import ImportanceNetwork
import kaleido
IG = ImportanceNetwork(importance_df,norm_method=False)
IG.plot_complete_sankey(show_top_n=10,multiclass=False, savename='./test.pdf', node_cmap='Reds', edge_cmap='Reds')
query_node="TP53_mut"
IG.plot_subgraph_sankey(query_node, upstream=False, savename='./TP53_mut.pdf', cmap='Reds')
IG.plot_subgraph_sankey("JAK2_mut", upstream=False, savename='./JAK2_mut.pdf', cmap='Reds')

