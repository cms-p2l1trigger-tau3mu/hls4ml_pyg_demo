import torch
import torch.nn as nn
from torch_geometric.nn import InstanceNorm, LayerNorm, GraphNorm, global_mean_pool, global_add_pool, global_max_pool

from .gen_conv import GENConv
import pandas as pd
import torch.nn.utils.prune as prune
import torch.nn.functional as F


class PruneModel(nn.Module):
    def __init__(self, masks, x_dim, edge_attr_dim, virtual_node, model_config):
        super(PruneModel, self).__init__()
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
        # self.m_node_encoder = masks['node_encoder']
        # self.m_edge_encoder = masks['edge_encoder']
        self.m_node_encoder_weight = masks["weight"]['node_encoder']
        self.m_edge_encoder_weight = masks["weight"]['edge_encoder']
        self.m_node_encoder_bias = masks["bias"]['node_encoder']
        self.m_edge_encoder_bias = masks["bias"]['edge_encoder']
        if self.bn_input:
            self.bn_node_feature = nn.BatchNorm1d(self.out_channels)
            self.bn_edge_feature = nn.BatchNorm1d(self.out_channels)

        for idx in range(self.n_layers):
            self.convs.append(GENConv(self.out_channels, self.out_channels, aggr=self.deepgcn_aggr, learn_t=True, learn_p=True, id = idx))
            self.mlps.append(MLP(channels, masks, norm_type=self.norm_type, dropout=self.dropout_p))
            # self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))

        if self.virtual_node:
            if self.readout == 'pool':
                self.pool = global_mean_pool
            else:
                raise NotImplementedError
        else:
            assert self.readout == 'pool'
            self.pool = global_mean_pool

        self.fc_out = nn.Linear(self.out_channels, 1)
        # self.m_fc_out = masks['fc_out']
        self.m_fc_out_weight = masks["weight"]['fc_out']
        self.m_fc_out_bias = masks["bias"]['fc_out']

    def update_masks(self, masks):
        # self.m_node_encoder = masks['node_encoder']
        # self.m_edge_encoder = masks['edge_encoder']
        self.m_node_encoder_weight = masks["weight"]['node_encoder']
        self.m_edge_encoder_weight = masks["weight"]['edge_encoder']
        self.m_node_encoder_bias = masks["bias"]['node_encoder']
        self.m_edge_encoder_bias = masks["bias"]['edge_encoder']
        for idx in range(len(self.mlps)):
            self.mlps[idx].update_masks(masks)
        # self.m_fc_out = masks['fc_out']
        self.m_fc_out_weight = masks["weight"]['fc_out']
        self.m_fc_out_bias = masks["bias"]['fc_out']

    def mask_to_device(self, device):
        # self.m_node_encoder = self.m_node_encoder.to(device)
        # self.m_edge_encoder = self.m_edge_encoder.to(device)
        self.m_node_encoder_weight = self.m_node_encoder_weight.to(device)
        self.m_edge_encoder_weight = self.m_edge_encoder_weight.to(device)
        self.m_node_encoder_bias = self.m_node_encoder_bias.to(device)
        self.m_edge_encoder_bias = self.m_edge_encoder_bias.to(device)
        for idx in range(len(self.mlps)):
            self.mlps[idx].mask_to_device(device)
        # self.m_fc_out = self.m_fc_out.to(device)
        self.m_fc_out_weight = self.m_fc_out_weight.to(device)
        self.m_fc_out_bias = self.m_fc_out_bias.to(device)

    def force_mask_apply(self):
        # self.node_encoder.weight.data.mul_(self.m_node_encoder)
        # self.edge_encoder.weight.data.mul_(self.m_edge_encoder)
        self.node_encoder.weight.data.mul_(self.m_node_encoder_weight)
        self.edge_encoder.weight.data.mul_(self.m_edge_encoder_weight)
        self.node_encoder.bias.data.mul_(self.m_node_encoder_bias)
        self.edge_encoder.bias.data.mul_(self.m_edge_encoder_bias)
        for idx in range(len(self.mlps)):
            self.mlps[idx].force_mask_apply()
        # self.fc_out.weight.data.mul_(self.m_fc_out)
        self.fc_out.weight.data.mul_(self.m_fc_out_weight)
        self.fc_out.bias.data.mul_(self.m_fc_out_bias)


    def forward(self, x, edge_index, edge_attr, batch, data, edge_atten=None, node_atten=None):
        x = self.node_encoder(x)
        # self.node_encoder.weight.data.mul_(self.m_node_encoder)
        self.node_encoder.weight.data.mul_(self.m_node_encoder_weight)
        self.node_encoder.bias.data.mul_(self.m_node_encoder_bias)
        edge_attr = self.edge_encoder(edge_attr)
        # self.edge_encoder.weight.data.mul_(self.m_edge_encoder)
        self.edge_encoder.weight.data.mul_(self.m_edge_encoder_weight)
        self.edge_encoder.bias.data.mul_(self.m_edge_encoder_bias)
        
        

        if self.bn_input:
            # print(f"self.bn_input: {self.bn_input}")
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)

        

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            # x = self.mlps[i](x, batch)
            x = self.mlps[i](x)


            
            x = x + identity
            # x += identity
            """
            NOTE:
            x += identity here doesn't work for some reason and give error:
            RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [6242, 128]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
            This is interesting because the inplace operation works fine in both Model and BV_Model
            I haved looked for what this PruneModel does so differently, 
            and the only difference I could find is that I use nn.Moudule as MLP class here
            except I use nn.Sequential for both Model and BV_Model
            Edit: I reverted my MLP class to the original nn.Sequential one, and I don't have the
            error statement, so that was the issue
            """

        if self.virtual_node:
            if self.readout == 'pool':
                pool_out = self.pool(x, batch)
            else: 
                raise NotImplementedError
        else:
            raise NotImplementedError
        out = self.fc_out(pool_out)
        # self.fc_out.weight.data.mul_(self.m_fc_out)
        self.fc_out.weight.data.mul_(self.m_fc_out_weight)
        self.fc_out.bias.data.mul_(self.m_fc_out_bias)
        return out.sigmoid()






class MLP(nn.Module):
    """
    hard coded for len(channels) == 3, and norm_type == batch
    """
    def __init__(self, channels, masks, norm_type, dropout, bias=True, idx=0):
        super(MLP, self).__init__()
        if norm_type == 'batch':
            norm = nn.BatchNorm1d
        else:
            raise NotImplementedError
        self.fc1 = nn.Linear(channels[0], channels[1], bias= bias)
        self.bn1 = norm(channels[1])
        self.fc2 = nn.Linear(channels[1], channels[2], bias= bias)
        self.bn2 = norm(channels[2])
        self.idx = idx
        # self.m1 = masks[f'mlps.{self.idx}.fc1']
        # self.m2 = masks[f'mlps.{self.idx}.fc2']
        self.m1_weight = masks["weight"][f'mlps.{self.idx}.fc1']
        self.m2_weight = masks["weight"][f'mlps.{self.idx}.fc2']
        self.m1_bias = masks["bias"][f'mlps.{self.idx}.fc1']
        self.m2_bias = masks["bias"][f'mlps.{self.idx}.fc2']
        
    def update_masks(self, masks):
        # self.m1 = masks[f'mlps.{self.idx}.fc1']
        # self.m2 = masks[f'mlps.{self.idx}.fc2']
        self.m1_weight = masks["weight"][f'mlps.{self.idx}.fc1']
        self.m2_weight = masks["weight"][f'mlps.{self.idx}.fc2']
        self.m1_bias = masks["bias"][f'mlps.{self.idx}.fc1']
        self.m2_bias = masks["bias"][f'mlps.{self.idx}.fc2']

    def mask_to_device(self, device):
        # self.m1 = self.m1.to(device)
        # self.m2 = self.m2.to(device)
        self.m1_weight = self.m1_weight.to(device)
        self.m2_weight = self.m2_weight.to(device)
        self.m1_bias = self.m1_bias.to(device)
        self.m2_bias = self.m2_bias.to(device)

    def force_mask_apply(self):
        # self.fc1.weight.data.mul_(self.m1)
        # self.fc2.weight.data.mul_(self.m2)
        self.fc1.weight.data.mul_(self.m1_weight)
        self.fc2.weight.data.mul_(self.m2_weight)
        self.fc1.bias.data.mul_(self.m1_bias)
        self.fc2.bias.data.mul_(self.m2_bias)

    def forward(self, x):
        test = self.fc1(x)
        x = F.relu(self.bn1(test))
        # self.fc1.weight.data.mul_(self.m1) # wonder if the masking is done after forwarding to zero the
        # # the changed weights from backprop ??
        # out = F.relu(self.bn2(self.fc2(x)))
        # self.fc2.weight.data.mul_(self.m2)
        self.fc1.weight.data.mul_(self.m1_weight) # wonder if the masking is done after forwarding to zero the
        # the changed weights from backprop ??
        self.fc1.bias.data.mul_(self.m1_bias)
        out = F.relu(self.bn2(self.fc2(x)))
        self.fc2.weight.data.mul_(self.m2_weight)
        self.fc2.bias.data.mul_(self.m2_bias)

        return out

# class BatchSequential(nn.Sequential):
#     def forward(self, inputs, batch):
#         for module in self._modules.values():
#             if isinstance(module, (GraphNorm, InstanceNorm)):
#                 inputs = module(inputs, batch)
#             else:
#                 inputs = module(inputs)
#         return inputs


# class MLP(BatchSequential):
#     def __init__(self, channels, norm_type, dropout, bias=True):

#         if norm_type == 'batch':
#             norm = nn.BatchNorm1d
#         elif norm_type == 'layer':
#             norm = LayerNorm
#         elif norm_type == 'instance':
#             norm = InstanceNorm
#         elif norm_type == 'graph':
#             norm = GraphNorm
#         else:
#             raise NotImplementedError

#         m = []
#         for i in range(1, len(channels)):
#             m.append(nn.Linear(channels[i - 1], channels[i], bias))

#             if i < len(channels) - 1:
#                 m.append(norm(channels[i]))
#                 m.append(nn.LeakyReLU())
#                 m.append(nn.Dropout(dropout))

#         super(MLP, self).__init__(*m)