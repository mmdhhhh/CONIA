import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
import os
import time
from model.RobustGCN import RobustGCN
from model.GNN_Model import GCN as torch_GCN
import argparse

from simpgcn.simpgcn import SimPGCN
from utils import *
from torch_geometric import utils
from torch_geometric.data import Data
from deeprobust.graph.defense import GCN
import deeprobust.graph.utils as routils


random_state = 123
setup_seed(random_state)
parser = argparse.ArgumentParser(description='CONIA')
parser.add_argument('--device', type=str, default="cpu", help='gpu or cpu.')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--attack', type=str, default='conia',  help='Attack method')
parser.add_argument('--feature_type', action='store_false', help='True: discrete,False: continuous')
parser.add_argument('--defense_model', type=str, default='robustgcn',choices=['robustgcn', 'simpgcn'],
                    help='defense model')
parser.add_argument('--injection_ratio', default=0.03, type=float, help='injection ratio')
parser.add_argument('--lr_defense', default=0.01, type=float, help='Learning rate of defense model')
parser.add_argument('--weight_decay_defense', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--nepochs_defense', type=int, default=200, help='Number of defense model epochs')
parser.add_argument('--hidden_defense', type=int, default=64, help='Number of defense model hidden units.')
parser.add_argument('--dropout_defense', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_defense', action='store_true', help='Train defense model')
parser.add_argument('--train_type', action='store_false', help='Data type: True: clean data')
parser.add_argument('--data_split', action='store_true', help='Split data')


args = parser.parse_args()

if args.dataset in ['cora','citeseer']:
    args.feature_type = True
else:
    args.feature_type = False

if args.train_type:
    print("train on clean {} dataset.".format(args.dataset))
else:
    print("train on attacked {} dataset.".format(args.dataset))
save_defense_path = 'defense_model/'
if not os.path.exists(save_defense_path):
    os.mkdir(save_defense_path)

save_attack_dir = "Graph_attacked/"
save_split_dir = "Graph_attacked/"

file_adj_attack = save_attack_dir + args.dataset + "_adj_attacked.npz"
if args.feature_type:
    file_features_attack = save_attack_dir + args.dataset + "_features_attacked.npz"
else:
    file_features_attack = save_attack_dir + args.dataset + "_features_attacked.npy"
file_train_index_np = save_split_dir + args.dataset + "_train_index_np.npy"
file_val_index_np = save_split_dir + args.dataset + "_val_index_np.npy"
file_test_index_np = save_split_dir + args.dataset + "_test_index_np.npy"

base_dir = 'dataset'
adj_sp, features_np, labels_np, train_index_np, val_index_np, test_index_np = \
    load_graph_data(args.dataset, base_dir,random_state=random_state)
print("args.data_split: ",args.data_split)
if not args.data_split:
    train_index_np = np.load(file_train_index_np, allow_pickle=True)
    val_index_np = np.load(file_val_index_np, allow_pickle=True)
    test_index_np = np.load(file_test_index_np, allow_pickle=True)
labels_tensor = torch.LongTensor(labels_np)


class_num = labels_np.max() + 1
clean_nodes_number = features_np.shape[0]
nfeat = features_np.shape[1]

adj_attacked = sp.load_npz(file_adj_attack)
if args.feature_type:
    features_attacked = sp.load_npz(file_features_attack)
else:
    features_attacked = np.load(file_features_attack, allow_pickle=True)

attacked_nodes_number = features_attacked.shape[0]

inj_nodes_idx = np.arange(clean_nodes_number,attacked_nodes_number)


all_labels_np = np.concatenate((labels_np, np.ones(attacked_nodes_number-clean_nodes_number,dtype=int)))
all_labels_tensor = torch.LongTensor(all_labels_np)

surro_model = torch_GCN(nfeat, args.hidden_defense, class_num, args.dropout_defense).to(args.device)

if args.defense_model == 'robustgcn':
    victim_model = RobustGCN(nfeat, args.hidden_defense, class_num, 2, args.dropout_defense, args.device).to(
        args.device)
else:
    if args.train_type:
        victim_model = SimPGCN(clean_nodes_number, nfeat, args.hidden_defense, class_num, device=args.device).to(args.device)
    else:
        victim_model = SimPGCN(attacked_nodes_number, nfeat, args.hidden_defense, class_num, device=args.device).to(
            args.device)

if args.train_type:
    save_type = "clean"
    print("save_type: ", save_type)
    use_adj_np = adj_sp
    use_features_np = features_np
else:
    save_type = "attacked"
    print("save_type: ",save_type)
    use_adj_np = adj_attacked
    use_features_np = features_attacked

map_location = torch.device(args.device)
if args.defense_model == 'robustgcn':
    if args.train_defense:
        train_model(victim_model, save_defense_path, use_adj_np, use_features_np, labels_np, train_index_np, val_index_np,
                        test_index_np,args.lr_defense, args.weight_decay_defense, args.feature_type, args.device,
                        save_type+"_"+args.defense_model,
                        args.dataset, epochs=args.nepochs_defense)
        print("defense {} model train out!\n".format(args.defense_model))
    victim_model.load_state_dict(torch.load(
        os.path.join(save_defense_path, save_type + "_" + args.defense_model + '_' + args.dataset + '_checkpoint.pkl'),
        map_location=map_location))
    victim_model.eval()
    for p in victim_model.parameters():
        p.requires_grad = False
elif args.defense_model == 'simpgcn':
    if args.train_defense:
        victim_model.fit(use_features_np, use_adj_np, labels_np, train_index_np, val_index_np)
        victim_model.eval()
        for p in victim_model.parameters():
            p.requires_grad = False
        best_weights = deepcopy(victim_model.state_dict())
        torch.save(best_weights, os.path.join(save_defense_path, save_type + "_" + args.defense_model + '_' + args.dataset + '_checkpoint.pkl'),
                   _use_new_zipfile_serialization=False)

if args.defense_model in ['robustgcn']:
    edge_index_attacked, _ = utils.from_scipy_sparse_matrix(adj_attacked)
    if args.feature_type:
        features_tensor_attacked = torch.from_numpy(features_attacked.todense().astype('double')).float()
    else:
        features_tensor_attacked = torch.from_numpy(features_attacked.astype('double')).float()


    data_attacked = Data(x=features_tensor_attacked, y=labels_tensor, edge_index=edge_index_attacked,
                         num_nodes=features_tensor_attacked.shape[0],
                         num_features=features_tensor_attacked.shape[1], n_class=labels_tensor.max().item() + 1)
    data_attacked = data_attacked.to(args.device)
    if args.train_type:
        edge_index_clean, _ = utils.from_scipy_sparse_matrix(adj_sp)
        if args.feature_type:
            features_tensor_clean = torch.from_numpy(features_np.todense().astype('double')).float()
        else:
            features_tensor_clean = torch.from_numpy(features_np.astype('double')).float()

        data_clean = Data(x=features_tensor_clean, y=labels_tensor, edge_index=edge_index_clean, num_nodes=features_tensor_clean.shape[0],
                    num_features=features_tensor_clean.shape[1], n_class=labels_tensor.max().item() + 1)
        data_clean = data_clean.to(args.device)
        logits_attack = victim_model(data_attacked.x, data_attacked.edge_index)
        logits_clean = victim_model(data_clean.x, data_clean.edge_index)
        logp_attack = F.log_softmax(logits_attack, dim=1)
        logp_clean = F.log_softmax(logits_clean, dim=1)

        acc_class_attacked = accuracy(logp_attack[test_index_np], data_attacked.y[test_index_np])
        acc_class_clean = accuracy(logp_clean[test_index_np], data_clean.y[test_index_np])
        print("before attack acc: {:.4%}".format(acc_class_clean))
        print("after attack acc: {:.4%}".format(acc_class_attacked))
        print("defense model test accuracy down {:.4%}\n".format(acc_class_clean - acc_class_attacked))
    else:
        logits_attack = victim_model(data_attacked.x, data_attacked.edge_index)
        logp_attack = F.log_softmax(logits_attack, dim=1)
        acc_class_attacked = accuracy(logp_attack[test_index_np], data_attacked.y[test_index_np])
        print("after attack acc: {:.4%}".format(acc_class_attacked))
else:
    if args.train_type:
        features_tensor, adj_tensor = routils.to_tensor(features_np, adj_sp, device=args.device)

        if routils.is_sparse_tensor(adj_tensor):
            adj_norm = routils.normalize_adj_tensor(adj_tensor, sparse=True)
        else:
            adj_norm = routils.normalize_adj_tensor(adj_tensor)
        logits = victim_model(features_tensor, adj_norm)
        val_acc = accuracy(logits[val_index_np], labels_tensor[val_index_np])
        test_acc = accuracy(logits[test_index_np], labels_tensor[test_index_np])
        train_acc = accuracy(logits[train_index_np], labels_tensor[train_index_np])
        print("before attack Test accuracy {:.4%}".format(test_acc))
    else:

        features_tensor_attacked, adj_tensor_attacked = routils.to_tensor(features_attacked, adj_attacked, device=args.device)

        if routils.is_sparse_tensor(adj_tensor_attacked):
            adj_norm_attacked = routils.normalize_adj_tensor(adj_tensor_attacked, sparse=True)
        else:
            adj_norm_attacked = routils.normalize_adj_tensor(adj_tensor_attacked)

        logits = victim_model(features_tensor_attacked, adj_norm_attacked)
        val_acc = accuracy(logits[val_index_np], labels_tensor[val_index_np])
        test_acc = accuracy(logits[test_index_np], labels_tensor[test_index_np])
        train_acc = accuracy(logits[train_index_np], labels_tensor[train_index_np])
        print("attacked Test accuracy {:.4%}".format(test_acc))