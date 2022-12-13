import torch
import torch.nn as nn
from torch_geometric.nn import InstanceNorm, LayerNorm, GraphNorm, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.typing import OptTensor

from torch import Tensor
from typing import Optional, Union

from .gen_conv_jit import GENConv


class Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, virtual_node, model_config):
        super(Model, self).__init__()
        self.out_channels = model_config['out_channels']
        self.n_layers: int = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.readout = model_config['readout']
        self.norm_type = model_config['norm_type']
        self.deepgcn_aggr = model_config['deepgcn_aggr']
        self.bn_input = model_config['bn_input']
        self.virtual_node = virtual_node

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        channels = [self.out_channels, self.out_channels*2, self.out_channels]
    
        #128, 256, 128
        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)
        if self.bn_input:
            self.bn_node_feature = nn.BatchNorm1d(self.out_channels)
            self.bn_edge_feature = nn.BatchNorm1d(self.out_channels)
        
        norm = nn.BatchNorm1d
        
        
        for _ in range(self.n_layers):
            self.convs.append(GENConv(self.out_channels, self.out_channels, aggr=self.deepgcn_aggr, learn_t=True, learn_p=True).jittable())
            
            mlp = nn.Sequential()
            
            for i in range(1, len(channels)):
                mlp.add_module('linear{0}'.format(i), nn.Linear(channels[i - 1], channels[i]))

                if i < len(channels) - 1:
                    mlp.add_module('norm{0}'.format(i), norm(channels[i]))
                    mlp.add_module('leakyrelu{0}'.format(i), nn.LeakyReLU())
                    mlp.add_module('dropout{0}'.format(i), nn.Dropout(self.dropout_p))
            
            self.mlps.append(mlp)

        if self.virtual_node:
            if self.readout == 'lstm':
                self.lstm = nn.LSTMCell(self.out_channels, self.out_channels)
            elif self.readout == 'jknet':
                self.downsample = nn.Linear(self.out_channels * self.n_layers, self.out_channels)
            elif self.readout == 'vn':
                pass
            elif self.readout == 'pool':
                self.pool = global_mean_pool
            else:
                raise NotImplementedError
        else:
            assert self.readout == 'pool'
            self.pool = global_mean_pool

        self.fc_out = nn.Linear(self.out_channels, 1)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: OptTensor = None) -> Tensor:

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        if self.bn_input:
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)
        
        
        for conv, mlp in zip(self.convs, self.mlps):
            identity = x
            
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = mlp(x)

            x += identity
    
        if isinstance(batch, Tensor):
            pool_out: Tensor = self.pool(x, batch)
        else:
            fake_batch: Tensor = torch.zeros(len(x), dtype=torch.int64)
            pool_out: Tensor = self.pool(x, fake_batch)
            
        out: Tensor = self.fc_out(pool_out).sigmoid()
        return out

    def get_emb(self, x, edge_index, edge_attr, batch, ptr):
        v_idx, v_emb = (ptr[1:] - 1, []) if self.virtual_node else (None, None)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        if self.bn_input:
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.mlps[i](x, batch)

            if self.virtual_node:
                if i == 0:
                    hx, cx = identity[v_idx], torch.zeros_like(identity[v_idx])
                if self.readout == 'lstm':
                    hx, cx = self.lstm(x[v_idx], (hx, cx))
                elif self.readout == 'jknet':
                    v_emb.append(x[v_idx])
            x += identity
        return x


class BatchSequential(nn.Sequential):
    def forward(self, inputs: Tensor, batch: Tensor) -> Tensor:
        for module in self.m:
            if isinstance(module, (GraphNorm, InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels: list, norm_type: str, dropout: float, bias: bool = True):

        if norm_type == 'batch':
            norm = nn.BatchNorm1d
        elif norm_type == 'layer':
            norm = LayerNorm
        elif norm_type == 'instance':
            norm = InstanceNorm
        elif norm_type == 'graph':
            norm = GraphNorm
        else:
            raise NotImplementedError

        self.m : list = []
        for i in range(1, len(channels)):
            self.m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                self.m.append(norm(channels[i]))
                self.m.append(nn.LeakyReLU())
                self.m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*self.m)
