from torch_geometric.nn import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):

    def __init__(self, feature, hidden, classes,dropout):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self,x, edge_index):

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,self.dropout, training=self.training)
        x = self.sage2(x, edge_index)

        return x