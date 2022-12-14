
import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, global_mean_pool

from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
from torch_scatter import scatter, scatter_softmax
import pickle as pkl
import pandas as pd
import numpy as np

# for typing
from torch import Tensor
from typing import Optional, Union

import sys
# import quant batchnorm1d
sys.path.append('../../Tau3MuGNNs')
from Tau3MuGNNs.src.models.custom_bv import BatchNorm1dToQuantScaleBias, getCustomQuantizer

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)



class NodeEncoderBatchNorm1d(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, quantizer_dict):
        super(NodeEncoderBatchNorm1d, self).__init__()
        self.norm = BatchNorm1dToQuantScaleBias(
            input_size,
            weight_quant= quantizer_dict["weight"],
            bias_quant= quantizer_dict["bias"],
            input_quant= quantizer_dict["act"],
            output_quant= quantizer_dict["act"],
            return_quant_tensor= False
        )

    def forward(self, x):
        return self.norm(x)



class EdgeEncoderBatchNorm1d(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, quantizer_dict):
        super(EdgeEncoderBatchNorm1d, self).__init__()
        self.norm = BatchNorm1dToQuantScaleBias(
            input_size,
            weight_quant= quantizer_dict["weight"],
            bias_quant= quantizer_dict["bias"],
            input_quant= quantizer_dict["act"],
            output_quant= quantizer_dict["act"],
            return_quant_tensor= False
        )


    def forward(self, x):
        return self.norm(x)



class BvGENConvSmall(MessagePassing):
    def __init__(
            self, 
            flow='source_to_target', 
            out_channels=128, 
            nodeblock = None,
            id = 0,
            debugging = False
        ):
        super(BvGENConvSmall, self).__init__(flow=flow)
        self.out_channels = out_channels
        self.hidden_size = 2*self.out_channels

        # self.res_block = ResidualBlock(self.out_channels)
        
        if nodeblock == None:
            self.O = ObjectModel(self.out_channels, self.out_channels, self.hidden_size)
        else:
            self.O = nodeblock

        self.id = id
        self.debugging = debugging
        

        # print("nodeblock state dict: ", self.O.state_dict)
        self.beta = 0.01 # inverse temperature for softmax aggr. Has to match with hls version
        self.eps = 1e-07 # for message passing
        torch.set_printoptions(precision=8)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        with torch.no_grad():  

            
            # print(f"residualBlock input1: {residual}")

            # print(f"node attr: {x}")
            # print(f"attempt at x_j: {x[edge_index[0]]}")
            # print(f"edge_index[0]: {edge_index[0]}")
            # print(f"edge_index[1]: {edge_index[1]}")
            x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            # print(f"x_tilde.shape: {x_tilde.shape}")
            # print(f"x_tilde: {x_tilde}")
            

            if self.flow=='source_to_target':
                r = edge_index[1]
                s = edge_index[0]
            else:
                r = edge_index[0]
                s = edge_index[1]

            # m2 = torch.cat([x_tilde[r],
            #                 x_tilde[s],
            #                 self.E], dim=1)
            # x_j =  x_tilde[s]
            # print("forwarding")
            # output = self.R2(m2)
            # # print(f"R2 output max: {torch.max(output)}")
            # print(f"R2 output mean: {torch.mean(output)}, std: {torch.std(output)}")
            # # print(f"x_tilde[r]: {x_tilde[r]}")
            # # print(f"x_tilde[r] shape: {x_tilde[r].shape}")
            # output = torch.sigmoid(output.flatten())
            
            output = x_tilde
            # print(f"residualBlock input2: {output}")
            
            # print(f"model output mean: {torch.mean(output)}, std: {torch.std(output)}")
            return output

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming, target
        # x_j --> outgoing, source        
        # print(f"x_i: {x_i.shape}")
        # print(f"x_j: {x_j.shape}")
        # print(f"edge_attr: {edge_attr}")
        # m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        # print(f"R1 output max: {torch.max(self.E)}")
        # print(f"R1 output mean: {torch.mean(self.E)}, std: {torch.std(self.E)}")
        # print(f"x_j: {x_j}")
        # print(f"edge_attr: {edge_attr}")
        
        if self.debugging:
            df = pd.DataFrame(x_j.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_message_x_j.csv", index=False) # for debugging
            df = pd.DataFrame(edge_attr.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_message_edge_attr.csv", index=False) # for debugging


        msg = x_j + edge_attr
        # print(f"x_j + edge_attr: {msg}")
        msg = F.relu(msg) + self.eps
        # print(f"msg after: {msg}")
        # print("message passing")
        if self.debugging:
            df = pd.DataFrame(msg.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_message_output.csv", index=False) # for debugging
        return msg

    # def message(self, x_j: Tensor, edge_attr: OptTensor, edge_atten=None) -> Tensor:
    #     msg = x_j if edge_attr is None else x_j + edge_attr
    #     msg = F.relu(msg) + self.eps
    #     return msg

    def aggregate(self, inputs, index, dim_size = None):
        # print(f"inputs: {inputs}")
        # print(f"max abs inputs: {torch.max(torch.abs(inputs))}")
        # print(f"self.node_dim: {self.node_dim}")
        # print(f"index: {index}")
        # print(f"self.beta: {self.beta}")
        out = scatter_softmax(inputs * self.beta, index, dim=self.node_dim)
        # print(f"out: {out}")
        
        
        output = scatter(inputs * out, index, dim=self.node_dim,
                        dim_size=dim_size, reduce='sum')
        # print(f"aggregate output: {output}")
        if self.debugging:
            df = pd.DataFrame(output.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_aggregate_output.csv", index=False) # for debugging
        # print(f"aggregating")
        return output

    def update(self, aggr_out, x):
        # c = torch.cat([x, aggr_out], dim=1)

        if self.debugging:
            df = pd.DataFrame(aggr_out.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_update_aggregate_output.csv", index=False) # for debugging
            df = pd.DataFrame(x.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_update_x.csv", index=False) # for debugging
        # print(f"update x: {x}")

        # c = x + aggr_out
        c = aggr_out
        if self.debugging:
            df = pd.DataFrame(c.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_mlp_input.csv", index=False) # for debugging
        
        # print(f"x: {x}")

        
        output = c
        idx = 0
        for layer in self.O.layers:
            output_old = output
            output = layer(output)
            if self.debugging:
                df = pd.DataFrame(output.cpu().numpy())
                df.to_csv(f"./debugging/{self.id}_update_mlp{idx}.csv", index=False) # for debugging
                idx += 1
            # if layer.__class__.__name__ == 'BatchNorm1d':
                # print(f"BatchNorm1d input: {output_old}")
                # print(f"BatchNorm1d output: {output}")
            
            # print(f"layer {layer} output: {output}")

        # print(f"O output mean: {torch.mean(output)}, std: {torch.std(output)}")
        # print(f"O output {output}")
        
        # print(f"node update output shape: {output.shape}")
        # print("updating")
        return output





class BvGENConvBig(nn.Module):
    def __init__(
            self,
            n_layers : int, 
            int_bitwidth = 4,
            fractional_bitwidth = 4,
            flow = 'source_to_target',
            out_channels = 128,
            debugging = False
        ):
        super().__init__()
        self.flow = flow
        self.debugging = debugging
        self.n_layers = n_layers
        self.out_channels = out_channels
        self.quantizer_dict = getCustomQuantizer(int_bitwidth, fractional_bitwidth)
        self.hidden_size = 2*self.out_channels
        self.node_encoder = nn.Linear(3, self.out_channels)
        self.edge_encoder = nn.Linear(4, self.out_channels)

        self.node_encoder_norm = NodeEncoderBatchNorm1d(self.out_channels, self.quantizer_dict)
        self.edge_encoder_norm = EdgeEncoderBatchNorm1d(self.out_channels, self.quantizer_dict)

        self.gnns = nn.ModuleList() # this is where we keep our GNN layers
        # automate nodeblock creation
        for idx in range(self.n_layers):
            # this assigns a nodeblock variable to class variable with varying names
            layer_name = f'O_{idx}'
            # set attribute a certain name to make is easy for pyg_to_hls converter
            setattr(self, layer_name, ObjectModel(self.out_channels, self.out_channels, self.hidden_size))
            gnn = BvGENConvSmall(
                flow = self.flow, 
                out_channels = out_channels, 
                nodeblock = getattr(self, layer_name),
                id = idx,
                debugging = self.debugging
            )
            self.gnns.append(gnn)
        assert(len(self.gnns) == self.n_layers)

        self.pool = global_mean_pool
        self.fc_out =  nn.Linear(self.out_channels, 1)


        

    def forward(self, data):
        with torch.no_grad():
            x = data.x
            # print(f"Node Encoder input {x}")
            x = self.node_encoder(x)
            if self.debugging:
                df = pd.DataFrame(x.cpu().numpy())
                df.to_csv(f"./debugging/node_encoder_output.csv", index=False) 
            
            # print(f"Node Encoder output max: {torch.max(x)}")
            # print(f"Node Encoder output abs means: {torch.mean(torch.abs(x))}")
            # print(f"Node Encoder output mean: {torch.mean(x)}. std: {torch.std(x)}")
            # print(f"Node Encoder output {x}")


            
            edge_index, edge_attr = data.edge_index, data.edge_attr
            # print(f"Edge Encoder input {edge_attr}")
            edge_attr = self.edge_encoder(edge_attr)
            if self.debugging:
                df = pd.DataFrame(edge_attr.cpu().numpy())
                df.to_csv(f"./debugging/edge_encoder_output.csv", index=False) 
                # print(f"Edge Encoder output max: {torch.max(edge_attr)}")
                # print(f"Edge Encoder output abs means: {torch.mean(torch.abs(edge_attr))}")
                # print(f"Edge Encoder output mean: {torch.mean(edge_attr)}. std:{torch.std(edge_attr)}")
                # print(f"Edge Encoder output {edge_attr}")
                # print(f"edge_index: {edge_index.shape}")
                # print(f"edge_attr: {edge_attr.shape}")
                # print(f"edge_attr: {edge_attr}")

                # now batchnorm the encoder

                # print(f"node_encoder_norm state_dict: {self.node_encoder_norm.state_dict()}")
                df = pd.DataFrame(x.cpu().numpy())
                df.to_csv(f"./debugging/node_encoder_norm_input.csv", index=False) 

            x = self.node_encoder_norm(x)
            if self.debugging:
                df = pd.DataFrame(x.cpu().numpy())
                df.to_csv(f"./debugging/node_encoder_norm_output.csv", index=False) 
            # print(f"node_encoder_norm output: {x}")
            # print(f"node_encoder_norm.norm weight: {self.node_encoder_norm.norm.weight}")
            edge_attr = self.edge_encoder_norm (edge_attr)

            if self.debugging:
                df = pd.DataFrame(edge_attr.cpu().numpy())
                df.to_csv(f"./debugging/edge_encoder_norm_output.csv", index=False) 
                # print(f"edge_encoder_norm output {edge_attr}")

            # now GNN layers
            residual = x
            counter = 0
            for gnn in self.gnns:
                residual = residual + gnn(residual, edge_index, edge_attr=edge_attr) # this acts as a "residual block"
                if self.debugging:
                    df = pd.DataFrame(residual.cpu().numpy())
                    df.to_csv(f"./debugging/{counter}_residual_output.csv", index=False) 
                    counter += 1# for debugging

            batch = None
            output = self.pool(residual, batch)
            output = self.fc_out(output)
            output = torch.sigmoid(output.flatten()) # final activation
            return output


    def SetDebugMode(self, debug_mode : bool):
        self.debugging = debug_mode
        for gnn in self.gnns:
            gnn.debugging = debug_mode