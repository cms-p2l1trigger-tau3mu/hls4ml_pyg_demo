import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import InstanceNorm, LayerNorm, GraphNorm, global_mean_pool, global_add_pool, global_max_pool

from .bv_gen_conv import GENConv
import pandas as pd
import copy

# custom bv classes
import brevitas.nn as qnn
from brevitas.nn.utils import mul_add_from_bn
from .custom_bv import BatchNorm1dToQuantScaleBias, getCustomQuantizer


class BV_Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, virtual_node, model_config, quantization=False):
        print("bv model is being used")
        super(BV_Model, self).__init__()
        self.out_channels = model_config['out_channels']
        self.n_layers = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.readout = model_config['readout']
        self.norm_type = model_config['norm_type']
        self.deepgcn_aggr = model_config['deepgcn_aggr']
        self.bn_input = model_config['bn_input']
        self.virtual_node = virtual_node
        # self.bn_quantization = model_config['bn_quantization']
        # ap_fixed setup for use in brevitas
        self.ap_fixed_dict = {}
        self.ap_fixed_dict["linear"] = getCustomQuantizer(
                            model_config['linear_ap_fixed_int'],
                            model_config['linear_ap_fixed_fract']
        )
        self.ap_fixed_dict["norm"] = getCustomQuantizer(
                            model_config['norm_ap_fixed_int'],
                            model_config['norm_ap_fixed_fract']
        )
        lin_weight_quantizer = self.ap_fixed_dict["linear"]["weight"]
        lin_bias_quantizer = self.ap_fixed_dict["linear"]["bias"]
        lin_act_quantizer = self.ap_fixed_dict["linear"]["act"]
        norm_weight_quantizer = self.ap_fixed_dict["norm"]["weight"]
        norm_bias_quantizer = self.ap_fixed_dict["norm"]["bias"]
        norm_act_quantizer = self.ap_fixed_dict["norm"]["act"]

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        channels = [self.out_channels, self.out_channels*2, self.out_channels]

        # initialize the encoders
        self.node_encoder = qnn.QuantLinear(
            x_dim, 
            self.out_channels,
            bias= True,
            weight_quant= norm_weight_quantizer,
            bias_quant= norm_bias_quantizer,
            input_quant= norm_act_quantizer,
            output_quant= norm_act_quantizer,
            return_quant_tensor= True
        )
        self.edge_encoder = qnn.QuantLinear(
            edge_attr_dim, 
            self.out_channels,
            bias= True,
            weight_quant= norm_weight_quantizer,
            bias_quant= norm_bias_quantizer,
            input_quant= norm_act_quantizer,
            output_quant= norm_act_quantizer,
            return_quant_tensor= True
        )
        self.quant_identity = qnn.QuantIdentity(
            # input_quant= norm_act_quantizer,
            act_quant= norm_act_quantizer,
            return_quant_tensor=True
        )
        # save model_config for GENConv
        self.model_config = model_config

        # intialize norms
        if self.bn_input:
            # self.bn_node_feature = BatchNorm1dToQuantScaleBias(
            #     num_features= self.out_channels,
            #     weight_quant= norm_weight_quantizer,
            #     bias_quant= norm_bias_quantizer,
            #     input_quant= norm_act_quantizer,
            #     output_quant= norm_act_quantizer,
            #     # return_quant_tensor= True
            # )
            # self.bn_edge_feature = BatchNorm1dToQuantScaleBias(
            #     num_features= self.out_channels,
            #     weight_quant= norm_weight_quantizer,
            #     bias_quant= norm_bias_quantizer,
            #     input_quant= norm_act_quantizer,
            #     output_quant= norm_act_quantizer,
            #     # return_quant_tensor= True
            # )
            self.bn_node_feature = nn.BatchNorm1d(self.out_channels)
            self.bn_edge_feature = nn.BatchNorm1d(self.out_channels)

        for idx in range(self.n_layers):
            self.convs.append(GENConv(self.out_channels, self.out_channels, self.model_config, aggr=self.deepgcn_aggr, learn_t=True, learn_p=True, id = idx))
            self.mlps.append(BV_MLP(channels, quantizer_dict=self.ap_fixed_dict["linear"], norm_type=self.norm_type, dropout=self.dropout_p))


        if self.virtual_node:
            if self.readout == 'lstm':
                self.lstm = nn.LSTMCell(self.out_channels, self.out_channels)
            elif self.readout == 'jknet':
                self.downsample = nn.Linear(self.out_channels * self.n_layers, self.out_channels)
            elif self.readout == 'vn':
                pass
            elif self.readout == 'pool':
                self.pool = global_mean_pool
            elif self.readout == 'sum':
                self.pool = global_add_pool
            else:
                raise NotImplementedError
        else:
            assert self.readout == 'pool'
            self.pool = global_mean_pool

        # self.fc_out = nn.Linear(self.out_channels, 1)
        self.fc_out = qnn.QuantLinear(
                    self.out_channels, 
                    1, 
                    bias=True,
                    weight_quant= lin_weight_quantizer,
                    bias_quant= lin_bias_quantizer,
                    input_quant= lin_act_quantizer,
                    output_quant= lin_act_quantizer,
                    # return_quant_tensor= True
        )

    def forward(self, x, edge_index, edge_attr, batch, data, edge_atten=None, node_atten=None):
        # v_idx, v_emb = (data.ptr[1:] - 1, []) if self.virtual_node else (None, None)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        if self.bn_input:
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)

        # # transform into quantensors
        # x = self.quant_identity(x)
        # edge_attr = self.quant_identity(edge_attr)

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.mlps[i](x, batch)
            x += identity

        if self.virtual_node:
            if self.readout == 'lstm':
                pool_out = hx
            elif self.readout == 'jknet':
                pool_out = self.downsample(torch.cat(v_emb, dim=1))
            elif self.readout == 'vn':
                pool_out = x[v_idx]
            elif self.readout == 'pool':
                pool_out = self.pool(x, batch)
            elif self.readout == 'sum':
                pool_out = self.pool(x, batch)
        else:
            pool_out = self.pool(x, batch)
        out = self.fc_out(pool_out)
        out = qnn.QuantSigmoid(act_quant=self.ap_fixed_dict["linear"]["act"])(out)
        return out

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




class BV_MLP(BatchSequential):
    def __init__(self, channels, norm_type, dropout, quantizer_dict, bias=True):

        if norm_type == 'batch':
            # norm = BatchNorm1dToQuantScaleBias
            norm = nn.BatchNorm1d
        else:
            raise NotImplementedError

        weight_quantizer = quantizer_dict["weight"]
        bias_quantizer = quantizer_dict["bias"]
        act_quantizer = quantizer_dict["act"]

        m = []
        for i in range(1, len(channels)):
            m.append(
                qnn.QuantLinear(
                    channels[i - 1], 
                    channels[i], 
                    bias=bias,
                    weight_quant= weight_quantizer,
                    bias_quant= bias_quantizer,
                    input_quant= act_quantizer,
                    output_quant= act_quantizer,
                    # return_quant_tensor= True
                )
            )

            if i < len(channels) - 1:
                # m.append(
                #     norm(
                #         num_features= channels[i],
                #         weight_quant= weight_quantizer,
                #         bias_quant= bias_quantizer,
                #         input_quant= act_quantizer,
                #         output_quant= act_quantizer,
                #         return_quant_tensor= True
                #     )
                # )
                m.append(norm(channels[i]))
                m.append(
                    qnn.QuantReLU(
                                act_quant= act_quantizer,
                                return_quant_tensor= True
                    )
                )

                # m.append(
                #     qnn.QuantDropout(p=dropout)
                # )

        super(BV_MLP, self).__init__(*m)

def _convertBnToBvbn(bn: BatchNorm1d, quantizer_dict: dict) -> BatchNorm1dToQuantScaleBias:
    """
    takes torch.nn batchnorm layer and returns an
    equivaldent BatchNorm1dToQuantScaleBias layer
    """
    out = mul_add_from_bn(
        bn_mean=bn.running_mean,
        bn_var=bn.running_var,
        bn_eps=bn.eps,
        bn_weight=bn.weight.data.clone(),
        bn_bias=bn.bias.data.clone())
    mul_factor, add_factor = out
    quant_bn = BatchNorm1dToQuantScaleBias(
                    num_features= bn.num_features,
                    weight_quant= quantizer_dict["weight"],
                    bias_quant= quantizer_dict["bias"],
                    input_quant= quantizer_dict["act"],
                    output_quant= quantizer_dict["act"],
                    return_quant_tensor= False
    )
    quant_bn.weight.data = mul_factor
    quant_bn.bias.data = add_factor
    return quant_bn

def convertBnToBvbn(bv_model):
    """
    returns the bv_model defined above, but with torch batchnorm converted
    to layers, and then the batchnorm values deleted
    """
    quantizer_dict = bv_model.ap_fixed_dict["norm"]
    new_model = copy.deepcopy(bv_model)
    new_model.bn_node_feature = _convertBnToBvbn(new_model.bn_node_feature, quantizer_dict)
    new_model.bn_edge_feature = _convertBnToBvbn(new_model.bn_edge_feature, quantizer_dict)
    for i in range(new_model.n_layers):
        if type(new_model.mlps[i]) == nn.BatchNorm1d:
            new_model.mlps[i] = _convertBnToBvbn(new_model.mlps[i], quantizer_dict)

    return new_model