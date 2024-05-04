from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        return x