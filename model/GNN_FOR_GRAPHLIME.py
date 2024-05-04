from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv

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

        return F.log_softmax(x, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, nfeat, hidden, classes,dropout,att_dropout, heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(nfeat, hidden//heads, heads=heads,dropout=att_dropout)
        self.gat2 = GATConv(hidden, classes,dropout=att_dropout)

    def forward(self,x, edge_index):

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)