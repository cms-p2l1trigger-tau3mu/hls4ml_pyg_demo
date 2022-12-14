
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
import brevitas.nn as qnn
import copy
from typing import Type

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, quantizer_dict, bias = True):
        super(ObjectModel, self).__init__()
        self.quant_identity = qnn.QuantIdentity(
                act_quant = quantizer_dict["act"],
                return_quant_tensor= False
        )
        self.layers = nn.Sequential(
            # qnn.QuantIdentity(
            #     act_quant = quantizer_dict["act"],
            #     return_quant_tensor= False
            # ),
            qnn.QuantLinear(
                input_size, hidden_size,
                bias=bias,
                weight_quant= quantizer_dict["weight"],
                bias_quant= quantizer_dict["bias"],
                input_quant= quantizer_dict["act"],
                output_quant= quantizer_dict["act"],
            ),
            # qnn.QuantIdentity(
            #     act_quant = quantizer_dict["act"],
            #     return_quant_tensor= False
            # ),
            BatchNorm1dToQuantScaleBias(
                hidden_size,
                weight_quant= quantizer_dict["weight"],
                bias_quant= quantizer_dict["bias"],
                input_quant= quantizer_dict["act"],
                output_quant= quantizer_dict["act"],
                return_quant_tensor= False
            ),
            qnn.QuantReLU(
                act_quant = quantizer_dict["act"],
                return_quant_tensor = False
            ),
            qnn.QuantLinear(
                hidden_size, output_size,
                bias=bias,
                weight_quant= quantizer_dict["weight"],
                bias_quant= quantizer_dict["bias"],
                input_quant= quantizer_dict["act"],
                output_quant= quantizer_dict["act"],
            ),
        )

    def forward(self, C):
        return self.quant_identity(self.layers(C))



class BvNodeEncoderBatchNorm1d(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, quantizer_dict):
        super(BvNodeEncoderBatchNorm1d, self).__init__()
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



class BvEdgeEncoderBatchNorm1d(nn.Module):
    """
    output shape should be same as the input shape
    """
    def __init__(self, input_size, quantizer_dict):
        super(BvEdgeEncoderBatchNorm1d, self).__init__()
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
            quantizer_dict,
            flow='source_to_target', 
            out_channels=128, 
            nodeblock = None,
            id = 0,
            debugging = False
        ):
        super(BvGENConvSmall, self).__init__(flow=flow)
        self.out_channels = out_channels
        self.hidden_size = 2*self.out_channels
        self.quantizer_dict = quantizer_dict
        self.quant_identity = qnn.QuantIdentity(
            act_quant = quantizer_dict["act"],
            return_quant_tensor= False
        )
        # self.res_block = ResidualBlock(self.out_channels)
        
        if nodeblock == None:
            self.O = ObjectModel(
                self.out_channels, 
                self.out_channels, 
                self.hidden_size, 
                self.quantizer_dict
            )
        else:
            self.O = nodeblock

        self.id = id
        self.debugging = debugging
        

        # print("nodeblock state dict: ", self.O.state_dict)
        self.beta = 0.01 # inverse temperature for softmax aggr. Has to match with hls version
        # self.eps = 1e-07 # for message passing
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


        msg = self.quant_identity(x_j) + self.quant_identity(edge_attr)
        # print(f"x_j + edge_attr: {msg}")
        # msg = F.relu(msg) + self.eps
        msg = qnn.QuantReLU(act_quant= self.quantizer_dict["act"])(msg)
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


        output = scatter(inputs, index, dim=self.node_dim,
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
            # df = pd.DataFrame(aggr_out.cpu().numpy())
            # df.to_csv(f"./debugging/{self.id}_update_aggregate_output.csv", index=False) # for debugging
            df = pd.DataFrame(x.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_update_x.csv", index=False) # for debugging
        # print(f"update x: {x}")

        # c = x + aggr_out
        c = aggr_out
        if self.debugging:
            df = pd.DataFrame(c.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_graphconv_output.csv", index=False) # for debugging
        
        # print(f"x: {x}")

        
        output = c
        output = self.quant_identity(output)
        idx = 0
        # for layer in self.O.layers:
        #     output_old = output
        #     output = layer(output)
        #     if self.debugging:
        #         df = pd.DataFrame(output.cpu().numpy())
        #         df.to_csv(f"./debugging/{self.id}_update_mlp{idx}.csv", index=False) # for debugging
        #         idx += 1
            # if layer.__class__.__name__ == 'BatchNorm1d':
                # print(f"BatchNorm1d input: {output_old}")
                # print(f"BatchNorm1d output: {output}")
            
            # print(f"layer {layer} output: {output}")

        # print(f"O output mean: {torch.mean(output)}, std: {torch.std(output)}")
        # print(f"O output {output}")
        
        # print(f"node update output shape: {output.shape}")
        # print("updating")
        output = self.O(output)
        if self.debugging:
            df = pd.DataFrame(output.cpu().numpy())
            df.to_csv(f"./debugging/{self.id}_mlp_output.csv", index=False) # for debugging
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
        self.node_encoder = qnn.QuantLinear(
            3, self.out_channels,
            bias= True,
            weight_quant= self.quantizer_dict["weight"],
            bias_quant= self.quantizer_dict["bias"],
            input_quant= self.quantizer_dict["act"],
            output_quant= self.quantizer_dict["act"],
        )
        self.edge_encoder = qnn.QuantLinear(
            4, self.out_channels,
            bias= True,
            weight_quant= self.quantizer_dict["weight"],
            bias_quant= self.quantizer_dict["bias"],
            input_quant= self.quantizer_dict["act"],
            output_quant= self.quantizer_dict["act"],
        )
        
        self.quant_identity = qnn.QuantIdentity(
                act_quant = self.quantizer_dict["act"],
                return_quant_tensor= False
        )

        self.node_encoder_norm = BvNodeEncoderBatchNorm1d(self.out_channels, self.quantizer_dict)
        self.edge_encoder_norm = BvEdgeEncoderBatchNorm1d(self.out_channels, self.quantizer_dict)

        self.gnns = nn.ModuleList() # this is where we keep our GNN layers
        # automate nodeblock creation
        for idx in range(self.n_layers):
            # this assigns a nodeblock variable to class variable with varying names
            layer_name = f'O_{idx}'
            # set attribute a certain name to make is easy for pyg_to_hls converter
            setattr(self, layer_name, 
                ObjectModel(
                    self.out_channels, 
                    self.out_channels, 
                    self.hidden_size,
                    self.quantizer_dict
                )
            )
            gnn = BvGENConvSmall(
                flow = self.flow, 
                quantizer_dict = self.quantizer_dict,
                out_channels = out_channels, 
                nodeblock = getattr(self, layer_name),
                id = idx,
                debugging = self.debugging
            )
            self.gnns.append(gnn)
        assert(len(self.gnns) == self.n_layers)

        self.pool = global_mean_pool
        self.fc_out =  qnn.QuantLinear(
            self.out_channels, 1,
            bias= True,
            weight_quant= self.quantizer_dict["weight"],
            bias_quant= self.quantizer_dict["bias"],
            input_quant= self.quantizer_dict["act"],
            output_quant= self.quantizer_dict["act"],
        )
        


        

    def forward(self, data):
        with torch.no_grad():
            x = data.x
            # print(f"Node Encoder input {x}")
            x = self.node_encoder(self.quant_identity(x))
            x = self.quant_identity(x)
            if self.debugging:
                df = pd.DataFrame(x.cpu().numpy())
                df.to_csv(f"./debugging/node_encoder_output.csv", index=False) 
            
            # print(f"Node Encoder output max: {torch.max(x)}")
            # print(f"Node Encoder output abs means: {torch.mean(torch.abs(x))}")
            # print(f"Node Encoder output mean: {torch.mean(x)}. std: {torch.std(x)}")
            # print(f"Node Encoder output {x}")


            
            edge_index, edge_attr = data.edge_index, data.edge_attr
            # print(f"Edge Encoder input {edge_attr}")
            edge_attr = self.edge_encoder(self.quant_identity(edge_attr))
            edge_attr = self.quant_identity(edge_attr)
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
                # residual = self.quant_identity(residual)
                if self.debugging:
                    df = pd.DataFrame(residual.cpu().numpy())
                    df.to_csv(f"./debugging/{counter}_residual_output.csv", index=False) 
                    counter += 1# for debugging

            
            batch = None
            output = self.pool(residual, batch)
            if self.debugging:
                df = pd.DataFrame(output.cpu().numpy())
                df.to_csv(f"./debugging/pool_output.csv", index=False)
            output = self.quant_identity(output)
            output = self.fc_out(output)
            # output = self.quant_identity(output)
            output =  qnn.QuantSigmoid(act_quant=self.quantizer_dict["act"])(output) # final activation
            # output = torch.sigmoid(output.flatten()) # final activation
            if self.debugging:
                df = pd.DataFrame(output.cpu().numpy())
                df.to_csv(f"./debugging/final_output.csv", index=False)
            return output.flatten()


    def SetDebugMode(self, debug_mode : bool):
        self.debugging = debug_mode
        for gnn in self.gnns:
            gnn.debugging = debug_mode


def _convertBvToTorch(module):
    """
    NOTE: not so sure converting quant activations matter,
    Since I think pyg_to_hls only copies the weights and biases
    """
    new_module = None
    if (module.__class__.__name__ == 'QuantLinear'):
        new_module = nn.Linear(
            module.in_features,
            module.out_features,
        )
        # load weights and biases
        new_module.weight = nn.Parameter(
            module.quant_weight().value
        )

        new_module.bias = nn.Parameter(
        module.quant_bias().value
        )
    elif (module.__class__.__name__ == 'QuantReLU'):
        new_module = nn.ReLU()
    elif (module.__class__.__name__ == 'QuantSigmoid'):
        new_module = nn.Sigmoid()
    elif (module.__class__.__name__ == 'BatchNorm1dToQuantScaleBias'):
        # keep the module as it is, but keep the weights and bias
        # as quantized
        new_module = copy.deepcopy(module)
        new_module.weight = nn.Parameter(
            module.quant_weight().value
        )
        new_module.bias = nn.Parameter(
            module.quant_bias().value
        )
    else:
        print(f"ERROR: supported module: {module.__class__.__name__}")
    
    
    return new_module


def convertBvToTorch(model: BvGENConvBig):
    new_model = copy.deepcopy(model)
    new_model.node_encoder = _convertBvToTorch(model.node_encoder)
    new_model.edge_encoder = _convertBvToTorch(model.edge_encoder)
    for idx in range(new_model.n_layers):
        gnn = new_model.gnns[idx]
        gnn_mlp = gnn.O
        for jdx in range(len(gnn_mlp.layers)):
            gnn_mlp.layers[jdx] = _convertBvToTorch(gnn_mlp.layers[jdx])

    new_model.fc_out = _convertBvToTorch(model.fc_out)

    return new_model


