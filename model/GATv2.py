from torch_geometric.nn import GATv2Conv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATv2(torch.nn.Module):

    def __init__(self, nfeat, hidden, classes,dropout,att_dropout, heads):
        super(GATv2, self).__init__()
        self.dropout = dropout
        self.gatv2_1 = GATv2Conv(nfeat, hidden//heads, heads=heads,dropout=att_dropout)
        self.gatv2_2 = GATv2Conv(hidden, classes,dropout=att_dropout)

    def forward(self,x, edge_index):
        x = self.gatv2_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,self.dropout, training=self.training)
        x = self.gatv2_2(x, edge_index)

        return x