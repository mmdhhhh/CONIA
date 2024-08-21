import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric import utils
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
def gcn_norm(adj_t, order=-0.5, add_self_loops=True):

    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

class RobustGCN(nn.Module):
    r"""
    Description
    -----------
    Robust Graph Convolutional Networks (`RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__)
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    dropout : bool, optional
        Whether to dropout during training. Default: ``True``.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,device):
        super(RobustGCN, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels

        self.act0 = F.elu
        self.act1 = F.relu
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(in_channels, hidden_channels, act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(num_layers - 2):
            self.layers.append(RobustGCNConv(hidden_channels, hidden_channels,
                                             act0=self.act0, act1=self.act1, dropout=dropout))
        self.layers.append(RobustGCNConv(hidden_channels, out_channels, act0=self.act0, act1=self.act1))
        self.dropout = dropout
        self.gaussian = None

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        r"""
        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : list of torch.SparseTensor
            List of sparse tensor of adjacency matrix.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).
        """

        row = edge_index[0]
        col = edge_index[1]
        adj  = SparseTensor(row=row, col=col,value=torch.ones(row.size(0),device=self.device))
        adj0, adj1 = gcn_norm(adj), gcn_norm(adj, order=-1.0)
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
        sample = torch.randn(var.shape).to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output.log_softmax(dim=-1)


class RobustGCNConv(nn.Module):
    r"""
    Description
    -----------
    RobustGCN convolutional layer.
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    act0 : func of torch.nn.functional, optional
        Activation function. Default: ``F.elu``.
    act1 : func of torch.nn.functional, optional
        Activation function. Default: ``F.relu``.
    initial : bool, optional
        Whether to initialize variance.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.
    """

    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        self.dropout = dropout

    def reset_parameters(self):
        self.mean_conv.reset_parameters()
        self.var_conv.reset_parameters()

    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""
        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        """
        if self.initial:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= mean
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)
        else:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= F.dropout(var, p=self.dropout, training=self.training)
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var ) +1e-6
            attention = torch.exp(-var)

            mean = mean * attention
            var = var * attention * attention
            mean = adj0 @ mean
            var = adj1 @ var

        return mean, var
