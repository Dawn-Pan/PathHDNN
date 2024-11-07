# PathHDNN - A pathway hierarchical-informed deep neural network framework for predicting immunotherapy response and mechanism interpretation
-----------------------------------------------------------------
Code by **Xiangmei Li** and **Bingyue Pan** at Harbin Medical University.
This repository contains source code and data for **PathHDNN** 

## 1. Introduction

**PathHDNN** is a pathway hierarchical-informed deep neural network (PathHDNN) framework to predict the therapeutic responses of cancer patients and identify key pathways associated with immunotherapy efficacy.
The PathHDNN facilitates the construction of a sparse neural network utilizing a pathway and input data. Documentation examples leverage the Reactome pathway database along with a genomic dataset for neural network generation. This tool not only enables training and network interpretation through SHAP but also offers plotting capabilities for creating sankey diagrams.

## 2. Design of PathHDNN

![alt text](image/workflow.tif "Design of PathHDNN")

Figure 1: Overall architecture of PathHDNN

## 3. Installation

**PathHDNN** relies on [Python (version 3.9)](https://www.python.org/downloads/release/python-390/) environments.

## 4. Usage

We have made available all the code necessary to execute **PathHDNN** in this GitHub repository. Please ensure that you replace the input data paths in the code with your own storage locations.

**1. Code**: The complete code for **PathHDNN** is located at folder ``PathHDNN/``.

**2. Data**: The datasets used to train **PathHDNN** are located at folder ``data/``
| File                              | Description                                                                   |
|------------------------------------|------------------------------------------------------------------------|
| maf_data.txt                             | Alteration features of genes                            |
| pathways.tsv                           | Hierarchical structure of reactome|
| reactome_data.txt                           | Relationships between alteration features and biological entities                             |
| sample_data.txt | Patient lable                                       |

**3. Output**: Get the best trained model. The models used in this paper are located at folder ``data/``,named ``144_train_model/``.
```sh
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
    ```


## 5. Interpretation of the PathHDNN model

To further clarify the decision-making process of PathHDNN and identify important genes or pathways influencing immunotherapy response prediction, SHapley Additive exPlannations (**SHAP**) algorithm was employed to interpret the PathHDNN model.

The code for calculating **SHAP value** is in the file ``model_explain.py`` which located at folder ``PathHDNN/``.

```sh
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
```
![alt text](image/test.jpg "test")
![alt text](image/TP53_mut.jpg "TP53_mut")
