from typing import Optional, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn.conv import MessagePassing


class GENConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = 'softmax', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False, eps: float = 1e-7, **kwargs):

        kwargs.setdefault('aggr', None)
        super(GENConv, self).__init__(**kwargs)
        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.eps = eps

        assert aggr in ['softmax', 'softmax_sg', 'power', 'mean', 'max', 'sum']

        self.initial_t = t
        self.initial_p = p

        if learn_t and aggr == 'softmax':
            self.t = Parameter(torch.Tensor([t]), requires_grad=True)
        else:
            self.t = t

        if learn_p:
            self.p = Parameter(torch.Tensor([p]), requires_grad=True)
        else:
            self.p = p

    def reset_parameters(self):
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.initial_t)
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.initial_p)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        msg = x_j if edge_attr is None else x_j + edge_attr
        msg = F.relu(msg) + self.eps
 
        return msg

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggr == 'softmax':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        elif self.aggr == 'max':
            return scatter(inputs, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='max')

        elif self.aggr == 'mean':
            return scatter(inputs, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='mean')

        elif self.aggr == 'sum':
            return scatter(inputs, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        elif self.aggr == 'softmax_sg':
            out = scatter_softmax(inputs * self.t, index,
                                  dim=self.node_dim).detach()
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        else:
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1 / self.p)

    def __repr__(self):
        return '{}({}, {}, aggr={})'.format(self.__class__.__name__,
                                            self.in_channels,
                                            self.out_channels, self.aggr)
