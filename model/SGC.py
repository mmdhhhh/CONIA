from torch_geometric.nn import SGConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGC(torch.nn.Module):

    def __init__(self, feature, hidden, classes,dropout,num_layers):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sgc = SGConv(feature, classes,K = num_layers)

    def forward(self,x, edge_index):

        x = self.sgc(x, edge_index)

        return x