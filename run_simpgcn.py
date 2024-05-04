import torch
import scipy.sparse as sp
import numpy as np
import os
import time
from copy import deepcopy
import math
from model import *
import torch.nn.functional as F
import argparse
from utils import *
import deeprobust.graph.utils as utils

from torch_geometric.data import Data
from simpgcn.simpgcn import SimPGCN

random_state = 123
setup_seed(random_state)
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda:0", help='gpu or cpu.')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--feature_type', action='store_false', help='True: discrete,False: continuous')
parser.add_argument('--lr_vict', default=0.01, type=float, help='Learning rate of victim model')
parser.add_argument('--weight_decay_vict', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--nepochs_vict', type=int, default=200, help='Number of victim model epochs')
parser.add_argument('--hidden_vict', type=int, default=64, help='Number of victim model hidden units.')
parser.add_argument('--hidden_vict_gat', type=int, default=64, help='Number of gat hidden units.')
parser.add_argument('--dropout_vict', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_vict', action='store_false', help='Train victim model')
parser.add_argument('--data_split', action='store_true', help='Split data')

args = parser.parse_args()

if args.dataset in ['cora','citeseer']:
    args.feature_type = True
else:
    args.feature_type = False

base_dir = 'dataset'
save_split_dir = "data_split/"
if not os.path.exists(save_split_dir):
    os.mkdir(save_split_dir)
file_train_index_np = save_split_dir + args.dataset + "_train_index_np.npy"
file_val_index_np = save_split_dir + args.dataset + "_val_index_np.npy"
file_test_index_np = save_split_dir + args.dataset + "_test_index_np.npy"

adj_sp, features_np, labels_np, train_index_np, val_index_np, test_index_np = load_graph_data(args.dataset, base_dir,
                                                                                       random_state=random_state)
labels_tensor = torch.LongTensor(labels_np).to(args.device)
if not args.data_split:
    train_index_np = np.load(file_train_index_np, allow_pickle=True)
    val_index_np = np.load(file_val_index_np, allow_pickle=True)
    test_index_np = np.load(file_test_index_np, allow_pickle=True)
print('train, val, test shape:', train_index_np.shape, val_index_np.shape, test_index_np.shape)
n_class = np.max(labels_np) + 1
nodes_number = features_np.shape[0]
nfeat = features_np.shape[1]
nodes_number = features_np.shape[0]
class_number = labels_np.max() + 1
nfeat = features_np.shape[1]
mod = "simpGCN"
map_location = torch.device(args.device)
victim_model = SimPGCN(nodes_number,nfeat,args.hidden_vict, class_number, device = args.device).to(args.device)
if args.train_vict:
    victim_model.fit(features_np, adj_sp,labels_np,train_index_np, val_index_np)
    print("victim {} model train out!\n".format(mod))
victim_model.eval()
for p in victim_model.parameters():
    p.requires_grad = False

features_tensor, adj_tensor = utils.to_tensor(features_np, adj_sp, device=args.device)

if utils.is_sparse_tensor(adj_tensor):
    adj_norm = utils.normalize_adj_tensor(adj_tensor, sparse=True)
else:
    adj_norm = utils.normalize_adj_tensor(adj_tensor)
logits = victim_model(features_tensor, adj_norm)

val_acc = accuracy(logits[val_index_np], labels_tensor[val_index_np])
test_acc = accuracy(logits[test_index_np], labels_tensor[test_index_np])
train_acc = accuracy(logits[train_index_np], labels_tensor[train_index_np])
acc = accuracy(logits, labels_tensor)
print("Test accuracy {:.4%}".format(test_acc))
print("ALL accuracy {:.4%}\n".format(acc))
