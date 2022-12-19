import torch
import torch.nn as nn
from torch_geometric.nn import InstanceNorm, LayerNorm, GraphNorm, global_mean_pool, global_add_pool, global_max_pool

from .gen_conv import GENConv
import pandas as pd


class Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, virtual_node, model_config):
        super(Model, self).__init__()
        print(f"non bv model")
        self.out_channels = model_config['out_channels']
        self.n_layers = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.readout = model_config['readout']
        self.norm_type = model_config['norm_type']
        self.deepgcn_aggr = model_config['deepgcn_aggr']
        self.bn_input = model_config['bn_input']
        self.virtual_node = virtual_node

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        channels = [self.out_channels, self.out_channels*2, self.out_channels]

        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)
        if self.bn_input:
            self.bn_node_feature = nn.BatchNorm1d(self.out_channels)
            self.bn_edge_feature = nn.BatchNorm1d(self.out_channels)

        for idx in range(self.n_layers):
            self.convs.append(GENConv(self.out_channels, self.out_channels, aggr=self.deepgcn_aggr, learn_t=True, learn_p=True, id = idx))
            self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))

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

    def forward(self, x, edge_index, edge_attr, batch, data, edge_atten=None, node_atten=None):
        # v_idx, v_emb = (data.ptr[1:] - 1, []) if self.virtual_node else (None, None)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # df = pd.DataFrame(x.detach().cpu().numpy())
        # df.to_csv(f"./debugging/siqi_node_encoder_output.csv", index=False) # for debugging
        # df = pd.DataFrame(edge_attr.detach().cpu().numpy())
        # df.to_csv(f"./debugging/siqi_edge_encoder_output.csv", index=False) # for debugging

        # if node_atten is not None:
        #     assert edge_atten is not None
        #     x = node_atten * x
        #     edge_attr = edge_atten * edge_attr

        if self.bn_input:
            # print(f"self.bn_input: {self.bn_input}")
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)
            # df = pd.DataFrame(x.detach().cpu().numpy())
            # df.to_csv(f"./debugging/siqi_node_encoder_norm_output.csv", index=False) # for debugging
            # df = pd.DataFrame(edge_attr.detach().cpu().numpy())
            # df.to_csv(f"./debugging/siqi_edge_encoder_norm_output.csv", index=False) # for debugging

        

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.mlps[i](x, batch)

            # if self.virtual_node:
            #     if i == 0:
            #         hx, cx = identity[v_idx], torch.zeros_like(identity[v_idx])
            #     if self.readout == 'lstm':
            #         hx, cx = self.lstm(x[v_idx], (hx, cx))
            #     elif self.readout == 'jknet':
            #         v_emb.append(x[v_idx])

            
            x += identity
            # print(f"layer {i} output: {x}")
            # df = pd.DataFrame(x.detach().cpu().numpy())
            # df.to_csv(f"./debugging/siqi_layer{i}_residual_output.csv", index=False) 

        if self.virtual_node:
            if self.readout == 'lstm':
                pool_out = hx
            elif self.readout == 'jknet':
                pool_out = self.downsample(torch.cat(v_emb, dim=1))
            elif self.readout == 'vn':
                pool_out = x[v_idx]
            elif self.readout == 'pool':
                pool_out = self.pool(x, batch)
        else:
            pool_out = self.pool(x, batch)
        out = self.fc_out(pool_out)
        return out.sigmoid()

    def get_emb(self, x, edge_index, edge_attr, batch, data):
        v_idx, v_emb = (data.ptr[1:] - 1, []) if self.virtual_node else (None, None)
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
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (GraphNorm, InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, norm_type, dropout, bias=True):

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

        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(nn.LeakyReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)