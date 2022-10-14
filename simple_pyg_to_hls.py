# %% [markdown]
# # Installation
# Please follow the instructions on README.mk file for installing the necessary packages to run this notebook

# %% [markdown]
# This walkthrough has few instructions. It's mainly just code to help the user to understand the pytorch geometric to hls4ml pipeline. If there's any confusion, please email me at yun79@purdue.edu

# %% [markdown]
# ### Imports

# %%
import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn

from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model
import hls4ml

from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# locals
from utils.models.interaction_network_pyg import GENConvBig
from model_wrappers import model_wrapper
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size
import time
import pickle as pkl


# %% [markdown]
# ### PyTorch Model

# %%
def number_of_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print(f"N Model Parameters: {pp}")

# %%
"""
We intialize our custom pytorch geometric(pyg) model
"""
n_layers = 1
out_channels = 1
torch_model = GENConvBig(
    n_layers, 
    flow = "source_to_target",
    out_channels = out_channels,
    debugging = True
).eval() # eval mode for bathnorm

# %%
number_of_parameters(torch_model)

# %%
torch_model.node_encoder_norm.weight = nn.Parameter(
    torch_model.node_encoder_norm.norm.weight
)

torch_model.node_encoder_norm.bias = nn.Parameter(
    torch_model.node_encoder_norm.norm.bias
)

torch_model.node_encoder_norm.running_mean = nn.Parameter(
    torch_model.node_encoder_norm.norm.running_mean
)

torch_model.node_encoder_norm.running_var = nn.Parameter(
    torch_model.node_encoder_norm.norm.running_var
)


# %%
torch_model.edge_encoder_norm.weight = nn.Parameter(
    torch_model.edge_encoder_norm.norm.weight
)

torch_model.edge_encoder_norm.bias = nn.Parameter(
    torch_model.edge_encoder_norm.norm.bias
)

torch_model.edge_encoder_norm.running_mean = nn.Parameter(
    torch_model.edge_encoder_norm.norm.running_mean
)

torch_model.edge_encoder_norm.running_var = nn.Parameter(
    torch_model.edge_encoder_norm.norm.running_var
)


# %%
Betas = []
for nodeblock_idx in range(n_layers):
    gnn = torch_model.gnns[nodeblock_idx]
    Betas.append(float(gnn.beta))

# %% [markdown]
# ### HLS Model

# %% [markdown]
# hls4ml cannot infer the *order* in which these submodules are called within the pytorch model's "forward()" function. We have to manually define this information in the form of an ordered-dictionary.

# %%
"""
forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
"""
forward_dict = OrderedDict()
forward_dict["node_encoder"] = "NodeEncoder"
forward_dict["edge_encoder"] = "EdgeEncoder"
forward_dict["node_encoder_norm"] = "NodeEncoderBatchNorm1d"
forward_dict["edge_encoder_norm"] = "EdgeEncoderBatchNorm1d"
for nodeblock_idx in range(n_layers):
    forward_dict[f"O_{nodeblock_idx}"] = "NodeBlock"
forward_dict["pool"] = "MeanPool"  
forward_dict["fc_out"] = "fc_out"

# %% [markdown]
# hls4ml creates a hardware implementation of the GNN, which can only be represented using fixed-size arrays. This restriction also applies to the inputs and outputs of the GNN, so we must define the size of the graphs that this hardware GNN can take as input**, again in the form of a dictionary. 
# 
# **Graphs of a different size can be padded or truncated to the appropriate size using the "fix_graph_size" function. In this notebook, padding/truncation is  done in the "Data" cell. 

# %%
"""
we define additional parameters.
"""
common_dim = out_channels
graph_dims = {
        "n_node": 28,
        "n_edge": 37,
        "node_attr": 3,
        "node_dim": common_dim,
        "edge_attr": 4,
    "edge_dim":common_dim
}

misc_config = {"Betas" : Betas}

# %% [markdown]
# Armed with our pytorch model and these two dictionaries**, we can create the HLS model. 

# %%
from hls4ml.model.optimizer import get_available_passes
"""
We initialize hls model from pyg model
"""
output_dir = "test_GNN"
config = config_from_pyg_model(torch_model,
                                   default_precision="ap_fixed<52,20>",
                                   default_index_precision='ap_uint<16>', 
                                   default_reuse_factor=8)

# config["Optimizers"] = get_available_passes()

print(f"config: {config}")
hls_model = convert_from_pyg_model(torch_model,
                                       n_edge=graph_dims['n_edge'],
                                       n_node=graph_dims['n_node'],
                                       edge_attr=graph_dims['edge_attr'],
                                       node_attr=graph_dims['node_attr'],
                                       edge_dim=graph_dims['edge_dim'],
                                       node_dim=graph_dims['node_dim'],
                                       misc_config = misc_config,
                                       forward_dictionary=forward_dict, 
                                       activate_final='sigmoid', #sigmoid
                                       output_dir=output_dir,
                                       hls_config=config)

# %% [markdown]
# ## hls_model.compile() builds the C-function for the model

# %%
hls_model.compile()
# hls_model.build()
"""
compile
build
implementation
"""

# %% [markdown]
# # Evaluation and prediction: hls_model.predict(input)

# %%
class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index, target):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index.transpose(0,1)

        node_attr, edge_attr, edge_index = self.x.detach().cpu().numpy(), self.edge_attr.detach().cpu().numpy(), self.edge_index.transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        node_attr, edge_attr, edge_index = np.ascontiguousarray(node_attr), np.ascontiguousarray(edge_attr), np.ascontiguousarray(edge_index)
        self.hls_data = [node_attr, edge_attr, edge_index]

        self.target = target
        self.np_target = np.reshape(target.detach().cpu().numpy(), newshape=(target.shape[0],))

def load_graphs(graph_indir, graph_dims, n_graphs):
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs_total = len(graph_files)
    IDs = np.arange(n_graphs_total)
    print(f"IDS: {IDs}")
    dataset = GraphDataset(graph_files=graph_files[IDs])

    graphs = []
    for data in dataset[:n_graphs]:
        node_attr, edge_attr, edge_index, target, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                             data.y,
                                                                             n_node_max=graph_dims['n_node'],
                                                                             n_edge_max=graph_dims['n_edge'])
        if not bad_graph:
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
#         graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
    print(f"graphs length: {len(graphs)}")

    print("writing test bench data for 1st graph")
    data = graphs[0]
    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(
        0, 1).detach().cpu().numpy().astype(np.int32)
    os.makedirs('tb_data', exist_ok=True)
    input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)], axis=1)
    np.savetxt('tb_data/input_data.dat', input_data, fmt='%f', delimiter=' ')

    return graphs


graph_indir = "trackml_data/processed_plus_pyg_small"

graphs = load_graphs(graph_indir, graph_dims, n_graphs=10)

# %%
with open('test_data.pickle', 'rb') as f:
    graphs= pkl.load(f) 

MSEs = []
for data in graphs:
    torch_pred = torch_model(data)
    torch_pred = torch_pred.detach().cpu().numpy().flatten()
    hls_pred = hls_model.predict(data.hls_data)
    MSE = mean_squared_error(torch_pred, hls_pred)
    MSEs.append(MSE)
    
print(f"MSEs: \n {MSEs}")
print(f"Average MSEs: \n {np.mean(MSEs)}")
