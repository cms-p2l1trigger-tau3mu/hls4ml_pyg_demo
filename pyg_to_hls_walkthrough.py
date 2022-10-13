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
        graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
    print(f"graphs length: {len(graphs)}")

    print("writing test bench data for 1st graph")
    data = graphs[0]
    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(
        0, 1).detach().cpu().numpy().astype(np.int32)
    os.makedirs('tb_data', exist_ok=True)
    input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)], axis=1)
    np.savetxt('tb_data/input_data.dat', input_data, fmt='%f', delimiter=' ')

    return graphs




"""
We intialize our custom pytorch geometric(pyg) model
"""
n_layers = 8
torch_model = GENConvBig(
    n_layers, 
    flow = "source_to_target",
    out_channels = 128,
    debugging = True
).eval() # eval mode for bathnorm
"""
We obtain the state dict(trained parameters) from Siqi Miao, PhD student of Prof Pan Li
"""
state_dict = torch.load('./model.pt', map_location="cpu")

"""
load/transfer the state dict into our pyg model
"""

"""
load/transfer the state dict into our pyg model
"""


torch_model.node_encoder.weight = nn.Parameter(state_dict['model_state_dict']['node_encoder.weight'])
torch_model.node_encoder.bias = nn.Parameter(state_dict['model_state_dict']['node_encoder.bias'])
torch_model.edge_encoder.weight = nn.Parameter(state_dict['model_state_dict']['edge_encoder.weight'])
torch_model.edge_encoder.bias = nn.Parameter(state_dict['model_state_dict']['edge_encoder.bias'])


torch_model.edge_encoder_norm.weight = nn.Parameter(
    state_dict['model_state_dict']['bn_edge_feature.weight']
)
torch_model.edge_encoder_norm.norm.weight = torch_model.edge_encoder_norm.weight # this is temporary soln to the structure of the class

torch_model.edge_encoder_norm.bias = nn.Parameter(
    state_dict['model_state_dict']['bn_edge_feature.bias']
)
torch_model.edge_encoder_norm.norm.bias = torch_model.edge_encoder_norm.bias # this is temporary soln to the structure of the class

torch_model.edge_encoder_norm.running_mean = nn.Parameter(
    state_dict['model_state_dict']['bn_edge_feature.running_mean']
)
torch_model.edge_encoder_norm.norm.running_mean = torch_model.edge_encoder_norm.running_mean # this is temporary soln to the structure of the class

torch_model.edge_encoder_norm.running_var = nn.Parameter(
    state_dict['model_state_dict']['bn_edge_feature.running_var']
)
torch_model.edge_encoder_norm.norm.running_var = torch_model.edge_encoder_norm.running_var # this is temporary soln to the structure of the class




# now the nodeblocks and betas
original_layer_idxs = [0,1,4] # don't ask me why it jumps from 1 to 4
new_layer_mlp_idxs = [0,1,3] # we skip 2 bc that's activation
Betas = []
for nodeblock_idx in range(n_layers):
    gnn = torch_model.gnns[nodeblock_idx]
    gnn.beta = state_dict['model_state_dict'][f'convs.{nodeblock_idx}.t']
    Betas.append(float(gnn.beta[0]))
    
    mlp_name = f"mlps.{nodeblock_idx}."
    
    for idx in range(len(original_layer_idxs)):
        original_layer_idx = original_layer_idxs[idx]
        new_layer_mlp_idx = new_layer_mlp_idxs[idx]
        nodeblock_name = f"O_{nodeblock_idx}"
        nodeblock = getattr(torch_model, nodeblock_name)
        module = nodeblock.layers[new_layer_mlp_idx]
        if (module.__class__.__name__ == 'Linear') or (module.__class__.__name__ == 'BatchNorm1d'):
            module.weight = nn.Parameter(
                state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.weight"]
            )
            module.bias = nn.Parameter(
                state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.bias"]
            )
        if (module.__class__.__name__ == 'BatchNorm1d'):
            module.running_mean = nn.Parameter(
                state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.running_mean"]
            )
            module.running_var = nn.Parameter(
                state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.running_var"]
            )
        
    
torch_model.fc_out.weight = nn.Parameter(state_dict['model_state_dict']['fc_out.weight'])
torch_model.fc_out.bias = nn.Parameter(state_dict['model_state_dict']['fc_out.bias'])

batchnorm_st_dict = OrderedDict()
batchnorm_st_dict["weight"] = state_dict['model_state_dict']['bn_node_feature.weight']
batchnorm_st_dict["bias"] = state_dict['model_state_dict']['bn_node_feature.bias']
batchnorm_st_dict["running_mean"] = state_dict['model_state_dict']['bn_node_feature.running_mean']
batchnorm_st_dict["running_var"] = state_dict['model_state_dict']['bn_node_feature.running_var']
torch_model.node_encoder_norm.norm.load_state_dict(batchnorm_st_dict)

"""
Just some code to test if the transfer was successful
"""
"""
Just some code to test if the transfer was successful
"""
for nodeblock_idx in range(n_layers):
    gnn = torch_model.gnns[nodeblock_idx]
    boolean_val = gnn.beta == state_dict['model_state_dict'][f'convs.{nodeblock_idx}.t']
#     print(f"beta: {gnn.beta}")
    print(f"beta loading for layer {idx} successful: {boolean_val}")
    
    mlp_name = f"mlps.{nodeblock_idx}."
    for idx in range(len(original_layer_idxs)):
        original_layer_idx = original_layer_idxs[idx]
        new_layer_mlp_idx = new_layer_mlp_idxs[idx]
        nodeblock_name = f"O_{nodeblock_idx}"
        nodeblock = getattr(torch_model, nodeblock_name)
        module = nodeblock.layers[new_layer_mlp_idx]
        if (module.__class__.__name__ == 'Linear') or (module.__class__.__name__ == 'BatchNorm1d'):
            boolean_val = torch.all(
                module.state_dict()["weight"] == state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.weight"]
            )
            print(f"weight loading for nodeblock {nodeblock_idx} layer {idx} successful: {boolean_val}")
            
            boolean_val = torch.all(
                module.state_dict()["bias"] == state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.bias"]
            )
            print(f"bias loading for nodeblock {nodeblock_idx} layer {idx} successful: {boolean_val}")
            
        if (module.__class__.__name__ == 'BatchNorm1d'):
            boolean_val = torch.all(
                module.state_dict()["running_mean"] == state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.running_mean"]
            )
            print(f"running_mean loading for nodeblock {nodeblock_idx} layer {idx} successful: {boolean_val}")
            
            boolean_val = torch.all(
                module.state_dict()["running_var"] == state_dict['model_state_dict'][mlp_name+f"{original_layer_idx}.running_var"]
            )
            print(f"running_var loading for nodeblock {nodeblock_idx} layer {idx} successful: {boolean_val}")
            
            

"""
siqi's model
"""
from Tau3MuGNNs.src.models import Model
import torch

config = {
    "bn_input": True,                 # Batch normalization on input features? This is to normalize the input features
  "n_layers": 8    ,                # Number of GNN layers
  "out_channels": 128  ,            # Number of hidden channels for each GNN layer
  "dropout_p": 0.5  ,               # Dropout probability
  "readout": "pool"  ,                # Specify the method to use for the readout layer. One can also use 'lstm', 'vn' or 'jknet'
  "norm_type": "batch"   ,            # Specify the type of normalization to use. One can also use 'instance', 'layer' or 'graph'
  "deepgcn_aggr": "softmax"          # Aggregation function for the DeeperGCN layers. Please refer to their documentation for more details
}
x_dim = 3
edge_attr_dim = 4

model_siqi = Model(x_dim, edge_attr_dim, True, config).eval()

state_dict = torch.load('./model.pt', map_location="cpu")

# model = torch.jit.load('./model.pt', map_location="cpu")

model_siqi.load_state_dict(state_dict['model_state_dict'])




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

"""
we define additional parameters.
"""
common_dim = 128
graph_dims = {
        "n_node": 28,
        "n_edge": 37,
        "node_attr": 3,
        "node_dim": common_dim,
        "edge_attr": 4,
    "edge_dim":common_dim
}

misc_config = {"Betas" : Betas}


"""
We initialize hls model from pyg model
"""
output_dir = "test_GNN"
config = config_from_pyg_model(torch_model,
                                   default_precision="ap_fixed<52,20>",
                                   default_index_precision='ap_uint<16>', 
                                   default_reuse_factor=8)
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



hls_model.compile()
    



graph_indir = "trackml_data/processed_plus_pyg_small"

graphs = load_graphs(graph_indir, graph_dims, n_graphs=10)


"""
Here we are testing hls model output compared to pyg model.
We are using Mean Squared Error (MSE) to calculate the differences 
in the output of the two models.
"""
MSE_l = []
batch = None
siqi_data = None
for data in graphs:
    torch_pred = torch_model(data)
    torch_pred = torch_pred.detach().cpu().numpy().flatten()
    hls_pred = hls_model.predict(data.hls_data)
    siqi_pred = model_siqi(
        x = data.x, edge_index = data.edge_index, edge_attr = data.edge_attr, batch = None, data = None
    )
    siqi_pred = siqi_pred.detach().cpu().numpy().flatten()
    print(f"torch_pred.shape: {torch_pred.shape}")
    print(f"hls_pred.shape: {hls_pred.shape}")
    MSE = mean_squared_error(torch_pred, hls_pred)
#     print(np.testing.assert_almost_equal(torch_pred, hls_pred))
    print(f"torch vs siqi: {mean_squared_error(torch_pred, siqi_pred)}")
    MSE_l.append(MSE)

MSE_l = np.array(MSE_l)
print(f"MSE_l: {MSE_l}")
print(f"Mean of all MSEs: {np.mean(MSE_l)}")



with open('test_data.pickle', 'rb') as f:
    graphs= pkl.load(f) 

MSEs = []
for data in graphs:
    torch_pred = torch_model(data)
    torch_pred = torch_pred.detach().cpu().numpy().flatten()
    hls_pred = hls_model.predict(data.hls_data)
    MSE = mean_squared_error(torch_pred, hls_pred)
    MSEs.append(MSE)
    siqi_pred = model_siqi(
        x = data.x, edge_index = data.edge_index, edge_attr = data.edge_attr, batch = None, data = None
    )
    siqi_pred = siqi_pred.detach().cpu().numpy().flatten()
    print(f"torch vs siqi: {mean_squared_error(torch_pred, siqi_pred)}")
    
print(f"MSEs: \n {MSEs}")



# """
# Now let's load some of tau3mu data from our group (Prof Mia Liu).
# This is still a smaller sample of the total data, but it's good enough. 

# NOTE: this will take some time (<15mins)
# """
# import timeit

# MSEs = []
# stages = ["train", "valid", "test"]
# # turn off debugging here
# torch_model.SetDebugMode(False)

# for stage in stages:
#     with open(f'tau3mu_data/test_BIG_data_{stage}.pickle', 'rb') as f:
#         graphs= pkl.load(f) 
        
#     counter = 0
#     start = timeit.default_timer()
#     for data in graphs:
#         # use counter to just keep track of the progress. Nothing fancy
#         if counter%500 ==0 and counter != 0:
#             print(f"counter: {counter}")
#         counter += 1
#         torch_pred = torch_model(data)
#         torch_pred = torch_pred.detach().cpu().numpy()
#         hls_pred = hls_model.predict(data.hls_data)
#         MSE = mean_squared_error(torch_pred, hls_pred)
#         MSEs.append(MSE)
#     end = timeit.default_timer()
#     print(f"time taken: {(end - start)/ 60} mins")
# MSEs = np.array(MSEs)
# print(f"MSE means: {np.mean(MSEs)}")
# print(f"MSE max: {np.max(MSEs)}")
# print(f"n_total: {MSEs.shape[0]}")


# """
# Now let's graph the MSE distribution
# """
# import numpy as np
# import matplotlib.pyplot as plt

# n_total = MSEs.shape[0]
# mean_val = np.mean(MSEs)

# plt.hist(MSEs, density=True, bins=50, label=f"Mean value: {mean_val}\n max val outlier removed") 
# plt.ylabel('Occurrence')
# plt.xlabel('MSE');
# plt.title(f'MSE of hls vs torch prediction (n_total: {n_total})')
# plt.legend()
# plt.show()
# plt.savefig('MSEs.png')
"""

You can see from the graph above that the error is very small (order of magnitude -7). This will obviously get bigger once you use more realistic ap_fixed parameters, but this proves that the conversion itself is working as intended.

So this is the latest progress on the pyg to hls conversion. The current model is only one layer out of eight pyg layers from the original Siqi's model. More work is on the way, but hopefully this gives you a good idea of how the conversion pipeline works.

For any questions, please email me at yun@purdue.edu, or slack if you already have me on it. Thank you!

Biography

This walkthrough and other local files were taken from Mr Abd Elabd's code at https://github.com/abdelabd/manual_GNN_conversion
The hls4ml pyg support's starting code has been taken from Mr. Abd Elabd and Prof Javier Duarte's work: https://github.com/fastmachinelearning/hls4ml/tree/pyg_to_hls_rebase_w_dataflow
"""