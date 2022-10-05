import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
from torch_scatter import scatter, scatter_softmax
import pickle as pkl
import pandas as pd
import numpy as np



class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)


class ResidualBlock(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size

    def forward(self, h_prev, h_after):
        assert(h_prev.shape == h_after.shape)
        return h_prev+h_after

class NodeEncoder(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, output_size):
        super(NodeEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.encoder(x)


class EdgeEncoder(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, output_size):
        super(EdgeEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.encoder(x)


class NodeEncoderBatchNorm1d(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size,):
        super(NodeEncoderBatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        return self.norm(x)


class EdgeEncoderBatchNorm1d(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size):
        super(EdgeEncoderBatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        return self.norm(x)


class InteractionNetwork(MessagePassing):
    def __init__(self, flow='source_to_target', out_channels=128):
        super(InteractionNetwork, self).__init__(flow=flow)
        self.out_channels = out_channels
        hidden_size = 2*self.out_channels
        self.n_neurons = hidden_size
        
        self.O = ObjectModel(self.out_channels, self.out_channels, hidden_size)
        self.res_block = ResidualBlock(self.out_channels)
        self.node_encoder = nn.Linear(3, self.out_channels)
        self.node_encoder_norm = NodeEncoderBatchNorm1d(self.out_channels)
        self.edge_encoder = nn.Linear(4, self.out_channels)
        self.edge_encoder_norm = EdgeEncoderBatchNorm1d(self.out_channels)

        self.beta = 0.01 # inverse temperature for softmax aggr. Has to match with hls version
        self.eps = 1e-07 # for message passing
        torch.set_printoptions(precision=8)

    def forward(self, data):
        with torch.no_grad():  
            x = data.x
            x = self.node_encoder(x)
            
            edge_index, edge_attr = data.edge_index, data.edge_attr
            edge_attr = self.edge_encoder(edge_attr)

            # now batchnorm the encoder
            x = self.node_encoder_norm(x)
            edge_attr = self.edge_encoder_norm (edge_attr)

            residual = x
            x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            

            if self.flow=='source_to_target':
                r = edge_index[1]
                s = edge_index[0]
            else:
                r = edge_index[0]
                s = edge_index[1]


            output = x_tilde
            output = self.res_block(residual, output) 
            output = torch.sigmoid(output.flatten())
            return output

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming, target
        # x_j --> outgoing, source        

        msg = x_j + edge_attr
        msg = F.relu(msg) + self.eps
        return msg

    def aggregate(self, inputs, index, dim_size = None):
        out = scatter_softmax(inputs * self.beta, index, dim=self.node_dim)

        

        output = scatter(inputs * out, index, dim=self.node_dim,
                        dim_size=dim_size, reduce='sum')
        return output

    def update(self, aggr_out, x):
        output = x + aggr_out
        idx = 0
        for layer in self.O.layers:
            output_old = output
            output = layer(output)
            idx += 1

        return output




