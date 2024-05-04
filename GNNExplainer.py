import os.path as osp
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer
import argparse
from torch_geometric.nn import models
import numpy as np
import torch.optim
from utils import *
from model.GNN_FOR_GRAPHLIME import *
from torch_geometric import utils
from torch_geometric.data import Data
import time
import torch.nn.functional as F
import math
from torch_geometric.datasets import Planetoid,Amazon

setup_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="gcn", help='Train model')
parser.add_argument('--dataset', default='pubmed', help='Dataset')
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--nepochs', type=int, default=1000,help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--hidden_gat', type=int, default=64, help='Number of gat hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
parser.add_argument('--feature_type', type=bool, default=True, help='True: discrete')
parser.add_argument('--device', type=str, default="cuda:0", help='Gpu or cpu.')
parser.add_argument('--train', action='store_true', help='Train or test')
parser.add_argument('--random_select', type=bool, default=True, help='Random select')


args = parser.parse_args()

if args.dataset in ['cora','citeseer']:
    args.feature_type = True
else:
    args.feature_type = False

base_dir = 'dataset'
adj_sp, features_np, labels_np, train_index_np, val_index_np, test_index_np = load_graph_data(args.dataset, base_dir)
print('train, val, test:', train_index_np.shape, val_index_np.shape, test_index_np.shape)
train_val_idx = np.concatenate([train_index_np, val_index_np])
train_val_idx = np.sort(train_val_idx)
nodes_number = features_np.shape[0]
degree_nodes=np.array(adj_sp.sum(axis=0))[0]
mean_degree = adj_sp.sum() / nodes_number
n_edge_max = math.ceil(mean_degree)

nfeat = features_np.shape[1]
nodes_number = features_np.shape[0]
class_number = labels_np.max() + 1

if args.random_select:
    select_num =  class_number
save_model_path = 'GNNExplainer_model/'

att_dropout = 0.6
heads = 8

if args.model == 'gcn':
    model = GCN(nfeat, args.hidden, class_number, args.dropout).to(args.device)
elif args.model == 'gat':
    model = GAT(nfeat, args.hidden_gat, class_number, args.dropout,att_dropout,heads).to(args.device)
if args.train:
    train_model(model, save_model_path, adj_sp, features_np, labels_np, train_index_np, val_index_np,
                test_index_np,args.lr, args.weight_decay, args.feature_type, args.device, args.model,
                args.dataset,epochs=args.nepochs)
    print("model train out!")

if args.device == 'cpu':
    map_location = torch.device('cpu')
    model.load_state_dict(
        torch.load(os.path.join(save_model_path, args.model +'_'+args.dataset+ '_checkpoint.pkl'), map_location=map_location))
else:
    model.load_state_dict(torch.load(os.path.join(save_model_path, args.model +'_'+args.dataset+ '_checkpoint.pkl')))
model.train()

edge_index, dege_weight = utils.from_scipy_sparse_matrix(adj_sp)

if args.feature_type:
    features_tensor = torch.from_numpy(features_np.todense().astype('double')).float()
else:
    features_tensor = torch.from_numpy(features_np.astype('double')).float()

labels_tensor = torch.LongTensor(labels_np).to(args.device)
data = Data(x=features_tensor, y=labels_tensor, edge_index=edge_index, num_nodes=features_tensor.shape[0],
            num_features=features_tensor.shape[1], n_class=labels_tensor.max().item() + 1).to(args.device)

logp = model(data.x, data.edge_index)

pred_labels = logp.max(1)[1]
pred_labels_np =  pred_labels.cpu().numpy()

n_class_nodes = []
n_nodes_index = np.arange(nodes_number)

if args.random_select == False:
    epo = class_number
    for class_ in range(class_number):
        indx_i_class = np.where(pred_labels_np == class_)[0]
        n_class_nodes.append(indx_i_class)
else:
    epo = select_num
    n_nodes_index = np.arange(nodes_number,dtype = np.int64)
    remain_nodes = n_nodes_index
    remain_num = nodes_number
    for i in range(epo):
        target_num = int(nodes_number / epo)
        if i == epo - 1:
            target_num = remain_num
        select_indx = np.random.choice(remain_num, target_num, replace=False)
        target_nodes = remain_nodes[select_indx]
        n_class_nodes.append(target_nodes)

        remain_nodes = np.setdiff1d(remain_nodes, target_nodes)
        remain_num = remain_num - target_num


explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

all_coefs = []
for i in range(epo):
    explain_class = i
    class_coefs = []
    print("class {} number: {}".format(i,n_class_nodes[explain_class].shape[0]))
    node_indexs = torch.from_numpy(n_class_nodes[explain_class])
    class_coefs = []
    print("class {} ...".format(i))
    for node_index in tqdm(node_indexs):
        explanation = explainer(data.x, data.edge_index, index=node_index)
        node_mask = explanation.get('node_mask')
        feat_labels = range(node_mask.size(1))
        score = node_mask.sum(dim=0)
        class_coefs.append(score.unsqueeze(0))
    class_coefs = torch.cat(class_coefs, dim=0)
    all_coefs.append(class_coefs)
    topk_values, topk_indices = torch.topk(all_coefs[i], k=10)
    unique_values, value_counts = torch.unique(topk_indices, return_counts=True)

    counts_zip = zip(unique_values.cpu(), value_counts.cpu())
    sort_counts = sorted(counts_zip, key=lambda nodes_margin_zip: nodes_margin_zip[1], reverse=True)
    sort_counts_list = list(zip(*sort_counts))
    dim_rank = np.array(sort_counts_list[0])
    dim_count_rank = np.array(sort_counts_list[1])
    print("dim_rank: ", dim_rank[0:20])
    print("dim_count_rank: ", dim_count_rank[0:20])
