import torch
import scipy.sparse as sp
import numpy as np
import os
import time
from copy import deepcopy
import math
from model.GNN_Model import *
from model.RobustGCN import RobustGCN
import torch.nn.functional as F
import argparse
from utils import *
from torch_geometric import utils
from torch_geometric.data import Data

from attack import CONIA


random_state = 123
setup_seed(random_state)
parser = argparse.ArgumentParser(description='CONIA')
parser.add_argument('--device', type=str, default="cuda:0", help='gpu or cpu.')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--feature_type', action='store_false', help='True: discrete,False: continuous')
parser.add_argument('--surro_type', type=str, default='gcn', help='Surrogate gnn model')
parser.add_argument('--injection_ratio', default=0.03, type=float, help='Injection ratio')
parser.add_argument('--lr_surr', default=0.01, type=float, help='Learning rate of surrogate model')
parser.add_argument('--lr_vict', default=0.01, type=float, help='Learning rate of victim model')
parser.add_argument('--weight_decay_surr', type=float, default=5e-4, help='Weight decay of surrogate model')
parser.add_argument('--weight_decay_vict', type=float, default=5e-4, help='Weight decay of victim model')
parser.add_argument('--nepochs_surr', type=int, default=2000, help='Number of surrogate model epochs')
parser.add_argument('--nepochs_vict', type=int, default=2000, help='Number of victim model epochs')
parser.add_argument('--hidden_surro', type=int, default=64, help='Number of surrogate model hidden units.')
parser.add_argument('--hidden_vict', type=int, default=64, help='Number of victim model hidden units.')
parser.add_argument('--hidden_vict_gat', type=int, default=64, help='Number of gat hidden units.')
parser.add_argument('--dropout_surro', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_vict', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attack', action='store_false', help='Whether to execute the attack')
parser.add_argument('--ceta', type=float, default=1, help='Homophily Constraint Coefficient')
parser.add_argument('--train_surr', action='store_false', help='Whether to train surrogate model')
parser.add_argument('--train_vict', action='store_false', help='Whether to train victim model')
parser.add_argument('--data_split', action='store_false', help='Split data')
parser.add_argument('--m', type=int, default=5,help='Gradient Computation Times')
parser.add_argument('--r', type=int, default=3,help='Control Factor')
parser.add_argument('--epoch_sec', type=int, default=901,help='Feature generator train epoch')
parser.add_argument('--use_pre', action='store_false', help='Predicted label or true label')

args = parser.parse_args()
if args.dataset in ['cora','citeseer']:
    args.feature_type = True
    args.ceta = 1
    # args.ceta = args.ceta
else:
    args.feature_type = False
    args.ceta = 0.1
    # args.ceta = args.ceta
victim_model_='gcn'

base_dir = 'dataset'
save_surrmodel_path = 'surrogate_model/'
save_victmodel_path = 'victim_model/'
save_attack_dir = "Graph_attacked/"
file_adj_attack = save_attack_dir + args.dataset + "_adj_attacked.npz"
if args.feature_type:
    file_features_attack = save_attack_dir + args.dataset + "_features_attacked.npz"
else:
    file_features_attack = save_attack_dir + args.dataset + "_features_attacked.npy"
file_train_index_np = save_attack_dir + args.dataset + "_train_index_np.npy"
file_val_index_np = save_attack_dir + args.dataset + "_val_index_np.npy"
file_test_index_np = save_attack_dir + args.dataset + "_test_index_np.npy"

adj_sp, features_np, labels_np, train_index_np, val_index_np, test_index_np = \
    load_graph_data(args.dataset, base_dir,random_state=random_state)
if not args.data_split:
    train_index_np = np.load(file_train_index_np, allow_pickle=True)
    val_index_np = np.load(file_val_index_np, allow_pickle=True)
    test_index_np = np.load(file_test_index_np, allow_pickle=True)
else:
    args.train_surr = True
    args.train_vict = True
    args.attack = True
print('dataset nodes num: {}, train num: {} val num: {}, test num: {}'.format(features_np.shape[0],
      train_index_np.shape[0], val_index_np.shape[0], test_index_np.shape[0]))

n_class = np.max(labels_np) + 1
nfeat = features_np.shape[1]
nodes_number = features_np.shape[0]
class_number = labels_np.max() + 1

att_dropout = 0.6
heads = 8
injection_ratio = args.injection_ratio
mean_degree = adj_sp.sum() / nodes_number

n_edge_max = 4

epoch_update_features = 600
epoch_select_edges = 300


lbth = args.r
ceta = args.ceta
epoch_sec = args.epoch_sec
if injection_ratio <= 0.01:
    lbth += 1

add_homophily = True
lr_sec = 1e-2

if args.dataset.lower() in ['pubmed','dblp']:
    args.nepochs_surr = 3000
    args.nepochs_vict = 3000


if args.surro_type == 'gcn':
    surro_model = GCN(nfeat, args.hidden_surro, class_number, args.dropout_surro).to(args.device)
else:
    raise Exception("surrogate {} model not supported\n".format(args.surro_type))

if args.train_surr:
    train_model(surro_model, save_surrmodel_path, adj_sp, features_np, labels_np, train_index_np, val_index_np,
                test_index_np, args.lr_surr,
                args.weight_decay_surr, args.feature_type, args.device, 'surr_'+args.surro_type,args.dataset,epochs=args.nepochs_surr)
    print("surrogate model train out!\n")

map_location = torch.device(args.device)

if victim_model_ == 'gcn':
    victim_model = GCN(nfeat, args.hidden_vict, class_number, args.dropout_vict).to(args.device)
elif victim_model_ == 'gat':
    victim_model = GAT(nfeat, args.hidden_vict_gat, class_number, args.dropout_vict,att_dropout,heads).to(args.device)
elif victim_model_ == 'graphsage':
    victim_model = GraphSAGE(nfeat, args.hidden_vict, class_number, args.dropout_vict).to(args.device)
elif victim_model_ == 'gatv2':
    victim_model = GATv2(nfeat, args.hidden_vict_gat, class_number, args.dropout_vict, att_dropout, heads).to(
        args.device)
elif victim_model_ == 'sgc':
    victim_model = SGC(nfeat, args.hidden_vict, class_number, args.dropout_vict,num_layers=2).to(
        args.device)
elif victim_model_ == 'ssgc':
    victim_model = SSGC(nfeat, args.hidden_vict, class_number, args.dropout_vict,num_layers=2).to(
        args.device)
elif victim_model_ == 'robustgcn':
    victim_model = RobustGCN(nfeat, args.hidden_vict, class_number, 2, args.dropout_vict,args.device).to(
        args.device)
if args.train_vict:
    train_model(victim_model, save_victmodel_path, adj_sp, features_np, labels_np, train_index_np, val_index_np,
                test_index_np,args.lr_vict, args.weight_decay_vict, args.feature_type, args.device,
                'victim_' + victim_model_, args.dataset,
                epochs=args.nepochs_vict)
victim_model.load_state_dict(torch.load(os.path.join(save_victmodel_path, 'victim_' + victim_model_ + '_' + args.dataset + '_checkpoint.pkl'),
        map_location=map_location))
victim_model.eval()
for p in victim_model.parameters():
    p.requires_grad = False
np.save(file_train_index_np, train_index_np)
np.save(file_val_index_np, val_index_np)
np.save(file_test_index_np, test_index_np)

surro_model.load_state_dict(
    torch.load(os.path.join(save_surrmodel_path, 'surr_'+args.surro_type +'_'+args.dataset+ '_checkpoint.pkl'), map_location=map_location))

surro_model.eval()

for p in surro_model.parameters():
    p.requires_grad = False

edge_index_clean, _ = utils.from_scipy_sparse_matrix(adj_sp)
if args.feature_type:
    features_tensor_clean = torch.from_numpy(features_np.todense().astype('double')).float()
else:
    features_tensor_clean = torch.from_numpy(features_np.astype('double')).float()
labels_tensor = torch.LongTensor(labels_np)

data_clean = Data(x=features_tensor_clean, y=labels_tensor, edge_index=edge_index_clean, num_nodes=features_tensor_clean.shape[0],
            num_features=features_tensor_clean.shape[1], n_class=labels_tensor.max().item() + 1)
data_clean = data_clean.to(args.device)

if not args.train_surr:
    print("surrogate model: {}".format(args.surro_type))
    logits = surro_model(data_clean.x, data_clean.edge_index)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_index_np], data_clean.y[val_index_np])
    test_acc = accuracy(logp[test_index_np], data_clean.y[test_index_np])
    train_acc = accuracy(logp[train_index_np], data_clean.y[train_index_np])
    acc = accuracy(logp, data_clean.y)
    print("Test accuracy {:.4%}".format(test_acc))
    print("ALL accuracy {:.4%}\n".format(acc))
if not args.train_vict:
    print("victim model: {}".format(victim_model_))
    logits = victim_model(data_clean.x, data_clean.edge_index)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_index_np], data_clean.y[val_index_np])
    test_acc = accuracy(logp[test_index_np], data_clean.y[test_index_np])
    train_acc = accuracy(logp[train_index_np], data_clean.y[train_index_np])
    acc = accuracy(logp, data_clean.y)
    print("Test accuracy {:.4%}".format(test_acc))
    print("ALL accuracy {:.4%}\n".format(acc))

if args.attack:
    attack_model = CONIA(features_np, adj_sp, labels_np, surro_model, args.feature_type,
                                    args.device, use_pre=args.use_pre)
    adj_attack,features_attack = attack_model.attack(test_index_np, injection_ratio,n_edge_max, epoch_update_features,
                                                     epoch_select_edges,add_homophily = add_homophily,
                                                     lr_sec=lr_sec,
                                                     epoch_sec=epoch_sec,ceta=ceta,lbth=lbth,m = args.m)
    print("attack finish !")
    if (args.feature_type):
        features_attack = sp.csr_matrix(features_attack)

    sp.save_npz(file_adj_attack, adj_attack)
    if args.feature_type:
        sp.save_npz(file_features_attack, features_attack)
    else:
        np.save(file_features_attack, features_attack)
    adj_attacked = sp.load_npz(file_adj_attack)
    if args.feature_type:
        features_attacked = sp.load_npz(file_features_attack)

    else:
        features_attacked = np.load(file_features_attack, allow_pickle=True)

    edge_index_attacked, _ = utils.from_scipy_sparse_matrix(adj_attacked)
    if args.feature_type:
        features_tensor_attacked = torch.from_numpy(features_attacked.todense().astype('double')).float()
    else:
        features_tensor_attacked = torch.from_numpy(features_attacked.astype('double')).float()

    data_attacked = Data(x=features_tensor_attacked, y=labels_tensor, edge_index=edge_index_attacked, num_nodes=features_tensor_attacked.shape[0],
                num_features=features_tensor_attacked.shape[1], n_class=labels_tensor.max().item() + 1)

    data_attacked = data_attacked.to(args.device)

    print("victim model: {}".format(victim_model_))
    logits_attack = victim_model(data_attacked.x, data_attacked.edge_index)
    logits_clean = victim_model(data_clean.x, data_clean.edge_index)
    logp_attack = F.log_softmax(logits_attack, dim=1)
    logp_clean = F.log_softmax(logits_clean, dim=1)
    acc_class_attacked = accuracy(logp_attack[test_index_np], data_attacked.y[test_index_np])
    acc_class_clean = accuracy(logp_clean[test_index_np], data_clean.y[test_index_np])
    print("before attack acc: {:.4%}".format(acc_class_clean))
    print("after attack acc: {:.4%}".format(acc_class_attacked))
    print("victim model test accuracy down {:.4%}".format(acc_class_clean - acc_class_attacked))
    print("\n")

else:

    adj_attacked = sp.load_npz(file_adj_attack)
    if args.feature_type:
        features_attacked = sp.load_npz(file_features_attack)
    else:
        features_attacked = np.load(file_features_attack, allow_pickle=True)

    edge_index_attacked, _ = utils.from_scipy_sparse_matrix(adj_attacked)
    if args.feature_type:
        features_tensor_attacked = torch.from_numpy(features_attacked.todense().astype('double')).float()
    else:
        features_tensor_attacked = torch.from_numpy(features_attacked.astype('double')).float()
    print("injection features.mean: ", features_tensor_attacked[data_clean.num_nodes:].sum(1).mean())
    data_attacked = Data(x=features_tensor_attacked, y=labels_tensor, edge_index=edge_index_attacked,
                         num_nodes=features_tensor_attacked.shape[0],
                         num_features=features_tensor_attacked.shape[1], n_class=labels_tensor.max().item() + 1)
    data_attacked = data_attacked.to(args.device)

    print("victim model: {}".format(victim_model_))
    logits_attack = victim_model(data_attacked.x, data_attacked.edge_index)
    logits_clean = victim_model(data_clean.x, data_clean.edge_index)
    logp_attack = F.log_softmax(logits_attack, dim=1)
    logp_clean = F.log_softmax(logits_clean, dim=1)

    acc_class_attacked = accuracy(logp_attack[test_index_np], data_attacked.y[test_index_np])
    acc_class_clean = accuracy(logp_clean[test_index_np], data_clean.y[test_index_np])

    print("before attack acc: {:.4%}".format(acc_class_clean))
    print("after attack acc: {:.4%}".format(acc_class_attacked))
    print("victim model test accuracy down {:.4%}\n".format(acc_class_clean - acc_class_attacked))

