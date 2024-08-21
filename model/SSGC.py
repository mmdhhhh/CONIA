from torch_geometric.nn import SSGConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSGC(torch.nn.Module):

    def __init__(self, feature, hidden, classes,dropout,num_layers):
        super(SSGC, self).__init__()
        self.dropout = dropout
        self.ssgc = SSGConv(feature, classes,K = num_layers,alpha=0.05)

    def forward(self,x, edge_index):

        x = self.ssgc(x, edge_index)

        return x