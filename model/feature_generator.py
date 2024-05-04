import torch
from torch_geometric.nn import GCNConv
from utils import *
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import copy
import math
class Feature_Generator(torch.nn.Module):
    def __init__(self, in_feats, h_feats, discrete_feat,feat_lim_min,feat_lim_max):
        super(Feature_Generator, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, h_feats)
        self.discrete_feat = discrete_feat
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        if discrete_feat:
            self.generator = torch.nn.Linear(3 * h_feats, in_feats)
        else:
            self.generator = torch.nn.Linear(3 * h_feats,2 * in_feats)
        self.activation = torch.nn.LeakyReLU()
        self.in_feats = in_feats

    def forward(self, features_tensor, all_edge_index,all_subset,remains, all_mapping,inj_num):
        ru_all = []
        for idx in range(inj_num):
            edge_index_inj = all_edge_index[idx]
            subset = all_subset[idx]
            features_subset = features_tensor[subset]
            inj_node = all_mapping[idx]
            target_nodes = remains[idx]
            ru_all.append(features_subset[target_nodes].sum(0)/target_nodes.shape[0])
            h = self.activation(self.conv1(features_subset, edge_index_inj))
            h = self.activation(self.conv2(h, edge_index_inj))
            target_nodes_h = h[target_nodes]
            inj_node_h = h[inj_node].squeeze()
            h = torch.cat((h.mean(0), inj_node_h, target_nodes_h.mean(0)))
            if idx == 0:
                all_hidden = h.unsqueeze(0)
            else:
                all_hidden = torch.cat((all_hidden, h.unsqueeze(0)), dim=0)
        feature_dist = self.generator(all_hidden)
        feature_dist = self.activation(feature_dist)
        ru_inj = torch.cat(ru_all).reshape(inj_num,features_tensor.shape[1])
        if self.discrete_feat:
            feature_dist = torch.sigmoid(feature_dist)
            dist = BernoulliStraightThrough(probs=feature_dist)
            feat = dist.rsample()

            homophily = F.cosine_similarity(ru_inj, feat)

            return feat, feat.mean(1),homophily.mean()
        else:
            mu = feature_dist[:,:self.in_feats]
            sigma = torch.abs(feature_dist[:,self.in_feats:]) +1e-9
            dist = torch.distributions.Normal(mu, sigma)
            feat = dist.rsample()
            feat = torch.clamp(feat,self.feat_lim_min,self.feat_lim_max)
            homophily = F.cosine_similarity(ru_inj, feat)
            return feat, [mu, sigma],homophily.mean()

class BernoulliStraightThrough(torch.distributions.Bernoulli):
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al, 2013)
    """
    has_rsample = True

    def rsample(self, sample_shape=torch.Size()):
        samples = self.sample(sample_shape)
        probs = self._param
        return samples + (probs - probs.detach())