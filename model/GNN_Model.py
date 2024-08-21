import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GATv2Conv,GCNConv, SAGEConv,SGConv,SSGConv

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
class SGC(torch.nn.Module):

    def __init__(self, feature, hidden, classes,dropout,num_layers):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sgc = SGConv(feature, classes,K = num_layers)

    def forward(self,x, edge_index):

        x = self.sgc(x, edge_index)

        return x
class SSGC(torch.nn.Module):

    def __init__(self, feature, hidden, classes,dropout,num_layers):
        super(SSGC, self).__init__()
        self.dropout = dropout
        self.ssgc = SSGConv(feature, classes,K = num_layers,alpha=0.05)

    def forward(self,x, edge_index):

        x = self.ssgc(x, edge_index)

        return x