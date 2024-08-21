import torch
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
import numpy as np
import os
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
import torch.nn.functional as F
import os
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid

def gcn_norm(edge_index, edge_weight=None, num_nodes=None,
             add_self_loops=False,flow="source_to_target", dtype=torch.float):

    fill_value = 1.
    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight

def load_graph_data(dataset_name,base_dir='dataset',random_state=123):
    file_name=os.path.join(base_dir,dataset_name.lower())
    train_size = 0.1
    val_size = 0.1
    test_size = 0.8
    if dataset_name.lower() in ['dblp','cora']:
        if not file_name.endswith('.npz'):
            file_name += '.npz'
        with np.load(file_name,allow_pickle=True) as loader:
            loader = dict(loader)
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                                  loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                       loader['attr_indptr']), shape=loader['attr_shape'])
                if dataset_name.lower() in ['reddit','products','cora_ml','dblp']:
                    features = features.toarray()
            else:
                features = None

            labels = loader.get('labels')
    elif dataset_name.lower() in ['pubmed']:
        adj = sp.load_npz(file_name+'/adj.npz')
        features = np.load(file_name+'/features.npy')
        labels = np.load(file_name+'/labels.npy')
    elif dataset_name.lower() in ['citeseer']:
        adj=sp.load_npz(file_name+'/adj.npz')
        features=sp.load_npz(file_name+'/features.npz')
        labels=np.load(file_name+'/labels.npy')
    else:
        raise Exception("dataset not supported")

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj[adj > 1] = 1
    if dataset_name.lower() in ['citeseer']:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
    mask = np.arange(labels.shape[0])
    train_index, val_index, test_index = train_val_test_split_tabular(mask, train_size=train_size, val_size=val_size,
                                                                      test_size=test_size,
                                                                      random_state=random_state)
    train_index=np.sort(train_index)
    val_index=np.sort(val_index)
    test_index=np.sort(test_index)
    return adj, features, labels, train_index, val_index, test_index

def train_val_test_split(labels_np,train_size, val_size,test_size):
    n_class = np.max(labels_np) + 1
    train_index=[]
    val_index=[]
    test_index=[]
    for label in range(n_class):
        num = np.count_nonzero(labels_np == label)
        index=np.where(labels_np == label)[0]
        class_split_train_num=int(num*train_size)
        class_split_val_num=int(num*val_size)
        random_index = np.array(random.sample(range(0, num), num))
        select_nodes_train=index[random_index[0:class_split_train_num]]
        train_index.append(select_nodes_train)
        select_nodes_val=index[random_index[class_split_train_num:class_split_train_num+class_split_val_num]]
        val_index.append(select_nodes_val)
        select_nodes_test=index[random_index[class_split_train_num+class_split_val_num:]]
        test_index.append(select_nodes_test)
    train_index = np.sort(np.concatenate(train_index))
    val_index = np.sort(np.concatenate(val_index))
    test_index = np.sort(np.concatenate(test_index))

    return train_index,val_index,test_index

def normalize_tensor(sp_adj_tensor, edges=None, sub_graph_nodes=None, sp_degree=None):
    edge_index = sp_adj_tensor.coalesce().indices()
    edge_weight = sp_adj_tensor.coalesce().values()
    shape = sp_adj_tensor.shape
    num_nodes = sp_adj_tensor.size(0)

    row, col = edge_index
    if sp_degree is None:
        deg = torch.sparse.sum(sp_adj_tensor, 1).to_dense().flatten()
    else:
        deg = sp_degree
        for i in range(len(edges)):
            idx = sub_graph_nodes[0, i]
            deg[idx] = deg[idx] + edges[i]
        last_deg = torch.sparse.sum(sp_adj_tensor[-1]).unsqueeze(0).data
        deg = torch.cat((deg, last_deg))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    nor_adj_tensor = torch.sparse.FloatTensor(edge_index, values, shape)
    del edge_index, edge_weight, values, deg_inv_sqrt
    return nor_adj_tensor

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)

    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape)).coalesce()


def generate_percent_split(dataset, seed=0, train_percent=10, val_percent=10):
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = torch.nonzero(data.y == c,as_tuple=True)[0].flatten()
        num_c = all_c_idx.size(0)
        train_num_per_c = num_c * train_percent // 100
        val_num_per_c = num_c * val_percent // 100
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask

def train_val_test_split_tabular(arrays, train_size=0.1, val_size=0.1, test_size=0.8, stratify=None, random_state=123):
    idx = arrays
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def worst_case_class(logits_pro, labels_np):
    logits_pro=deepcopy(logits_pro)
    logits_np = logits_pro.cpu().numpy()
    max_indx = logits_np.argmax(1)
    for i, indx in enumerate(max_indx):
        logits_np[i][labels_np[i]] = np.nan
    second_max_indx = np.nanargmax(logits_np, axis=1)

    return second_max_indx


def worst_class(logits_pro,use_pre=True):
    logits_pro=deepcopy(logits_pro)
    logits_np = logits_pro.cpu().numpy()
    max_indx = logits_np.argmax(1)

    logits_np[np.arange(len(max_indx)),max_indx] = np.nan
    second_max_indx = np.nanargmax(logits_np, axis=1)

    return second_max_indx


def get_neighbor(adj,target):
    one_order_nei = adj[target].nonzero()[1]
    return one_order_nei


def get_neighbor_self(adj,nodes):
    one_order_nei = np.unique(adj[nodes].nonzero()[1])
    neighbor_self=np.concatenate([nodes,one_order_nei])
    neighbor_self=np.sort(neighbor_self)
    return neighbor_self


def get_neighbors(adj,nodes):
    one_order_nei = np.unique(adj[nodes].nonzero()[1])
    neighbor_self=np.sort(one_order_nei)
    return neighbor_self


def train_model(model,save_model_path,adj_np,features_np,labels_np,train_index, val_index, test_index,lr,
                weight_decay,feature_type,device,model_name,dataset,epochs=500):
    edge_index, dege_weight = utils.from_scipy_sparse_matrix(adj_np)
    if feature_type:
        features_tensor = torch.from_numpy(features_np.todense().astype('double')).float()
    else:
        features_tensor = torch.from_numpy(features_np.astype('double')).float()
    labels = torch.LongTensor(labels_np).to(device)

    data = Data(x=features_tensor, y=labels, edge_index=edge_index, num_nodes=features_tensor.shape[0],
                num_features=features_tensor.shape[1], n_class=labels.max().item() + 1)
    data = data.to(device)

    model = model.to(device)

    if not os.path.exists(save_model_path):

        os.makedirs(save_model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("lr %.4f, weight_decay %.4f" % (lr, weight_decay))
    best_val = 0
    best_weights = None
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logp[train_index], labels[train_index])
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(data.x, data.edge_index)
        logp = F.log_softmax(logits, dim=1)
        acc_val = accuracy(logp[val_index], labels[val_index])
        if acc_val > best_val:
            best_val = acc_val
            best_weights = deepcopy(model.state_dict())
        del loss, logits
    torch.save(best_weights, os.path.join(save_model_path, model_name +'_'+dataset+ '_checkpoint.pkl'),
               _use_new_zipfile_serialization=False)
    model.load_state_dict(torch.load(os.path.join(save_model_path, model_name +'_'+dataset + '_checkpoint.pkl')))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logits = model(data.x, data.edge_index)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_index], labels[val_index])
    test_acc = accuracy(logp[test_index], labels[test_index])
    train_acc = accuracy(logp[train_index], labels[train_index])

    print("Train accuracy {:.4%}".format(train_acc))
    print("Validate accuracy {:.4%}".format(val_acc))
    print("Test accuracy {:.4%}".format(test_acc))


def get_subgraph(subset_nodes, edge_index,ori_num):
    subset_nodes_np = subset_nodes.cpu().numpy()
    adj_sp = utils.to_scipy_sparse_matrix(edge_index,num_nodes=ori_num)

    adj_sp = adj_sp.tocsr()
    adj_sp = adj_sp[subset_nodes_np][:, subset_nodes_np]
    subset_edge_index = utils.from_scipy_sparse_matrix(adj_sp)[0]

    return subset_edge_index

def get_inj_edge_index(subset_nodes, edge_index,ori_num,inj_sum):

    inj_num = subset_nodes.shape[0] - ori_num
    subset_nodes_np = subset_nodes.cpu().numpy()
    adj_sp = utils.to_scipy_sparse_matrix(edge_index, num_nodes=ori_num+inj_sum)

    adj_sp = adj_sp.tocsr()
    adj_sp = adj_sp[subset_nodes_np][:, subset_nodes_np]
    subset_edge_index = utils.from_scipy_sparse_matrix(adj_sp)[0]

    mask_ = torch.zeros(subset_edge_index.shape[1]).bool()
    mask_[subset_edge_index[0] >= ori_num] = True
    mask_[subset_edge_index[1] >= ori_num] = True
    indices0 = subset_edge_index[0][mask_].unsqueeze(0)
    indices1 = subset_edge_index[1][mask_].unsqueeze(0)
    indices = torch.cat((indices0, indices1), dim=0)
    inj_edge_index = indices
    return inj_edge_index



def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in zip(edge_index[0],edge_index[1]):
        sims.append(F.cosine_similarity(features[u].unsqueeze(0),features[v].unsqueeze(0)).item())
    sims = np.array(sims)
    print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")
    return sims

def inj_edge_sim_analysis(new_data,orig_num_nodes):
    neg_idx = torch.where(torch.logical_or(new_data.edge_index[0]>=orig_num_nodes,new_data.edge_index[1]>=orig_num_nodes))
    pos_mask = torch.ones(new_data.edge_index.size(1)).bool()
    pos_mask[neg_idx] = 0
    neg_mask = torch.zeros(new_data.edge_index.size(1)).bool()
    neg_mask[neg_idx] = 1

    pos_edge_index = new_data.edge_index[:,pos_mask]
    neg_edge_index = new_data.edge_index[:,neg_mask]
    sims = []
    for (u,v) in zip(neg_edge_index[0],neg_edge_index[1]):
        sims.append(F.cosine_similarity(new_data.x[u].unsqueeze(0),new_data.x[v].unsqueeze(0)))
    sims = np.array(sims)
    print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")
    return sims

def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature : bool
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        if sp.issparse(features):
            features = torch.FloatTensor(np.array(features.todense()))
        else:
            features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
