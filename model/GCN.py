from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):

    def __init__(self, nfeat, hidden, classes, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(nfeat, hidden)
        self.conv2 = GCNConv(hidden, classes)


    def forward(self, x, edge_index,edge_weight=None):

        if edge_weight != None:
            x = self.conv1(x, edge_index,edge_weight)
            x = F.relu(x)
            x = F.dropout(x,self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        return x
