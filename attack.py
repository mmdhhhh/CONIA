import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import time
import random
from torch_sparse import SparseTensor
from torch_geometric import utils
from torch_geometric.data import Data
import torch_sparse
from utils import *
import math
from tqdm import tqdm
from model.feature_generator import Feature_Generator


class CONIA:
    def __init__(self, features_np, adj_np, groundtruth_labels_np, surrogate_model, feature_type,
                 device, use_pre=True):
        self.surrogate_model = surrogate_model
        self.features = deepcopy(features_np)
        self.adj = deepcopy(adj_np)
        self.groundtruth_labels_np = groundtruth_labels_np
        self.groundtruth_labels_tensor = torch.LongTensor(groundtruth_labels_np)
        self.adj_attack = deepcopy(adj_np)
        self.n_class = np.max(groundtruth_labels_np) + 1
        self.features_dim = features_np.shape[1]
        self.nodes_num = features_np.shape[0]
        self.nodes_num_attack = features_np.shape[0]
        self.n_degree = np.array(adj_np.sum(0))[0]
        self.mean_degree = float(adj_np.sum() / self.nodes_num)
        self.feature_type = feature_type
        self.device = device
        self.use_pre = use_pre

        edge_index, _ = utils.from_scipy_sparse_matrix(adj_np)
        self.edge_index = edge_index.to(self.device)
        if feature_type:
            self.feature_budget = int(np.mean(features_np.sum(1)))
            self.features_tensor = torch.from_numpy(features_np.todense().astype('double')).float().to(self.device)
            self.features_attack_tensor = torch.from_numpy(features_np.todense().astype('double')).float().to(self.device)
        else:
            self.feature_budget = float(np.mean(features_np.sum(1)))
            self.features_tensor = torch.from_numpy(features_np.astype('double')).float().to(self.device)
            self.features_attack_tensor = torch.from_numpy(features_np.astype('double')).float().to(self.device)
            self.mu = torch.from_numpy(features_np.mean(0).astype('double')).float().to(self.device)
            self.sigma = torch.from_numpy(features_np.std(0).astype('double')).float().to(self.device)
        self.logits = self.surrogate_model(self.features_tensor, self.edge_index)
        self.logp = F.log_softmax(self.logits, dim=1).to(self.device)
        self.logits_pro = F.softmax(self.logits, dim=1).to(self.device)
        if use_pre:
            preds = self.logp.max(1)[1]
            self.labels_tensor = preds
            self.labels_np = preds.cpu().numpy()
        else:
            self.labels_np = groundtruth_labels_np
            self.labels_tensor = torch.LongTensor(groundtruth_labels_np).to(self.device)


    def generate_feature(self, num,surr_model,edge_index,features_tensor,
                            target_nodes,  feature_type=True):
        features_generate = np.random.randn(num, self.features_dim)
        return features_generate

    def generate_fulladj_attack(self, adj, injection_num, target_nodes, weight=1):

        adj = adj.tocoo()
        cur_num = self.nodes_num_attack
        injection_nodes = np.arange(cur_num, cur_num + injection_num)
        injection_nodes_repeat = np.repeat(injection_nodes, target_nodes.shape[0])
        target_nodes_repeat = np.tile(target_nodes, injection_num)
        newadjx = np.concatenate([injection_nodes_repeat, target_nodes_repeat])
        newadjy = np.concatenate([target_nodes_repeat, injection_nodes_repeat])
        newdata = np.repeat(weight, newadjy.shape[0])

        newrow = np.hstack([adj.row, newadjx])
        newcol = np.hstack([adj.col, newadjy])
        newdata = np.hstack([adj.data, newdata])

        newadj = sp.csr_matrix((newdata, (newrow, newcol)),
                               shape=(cur_num + injection_num, cur_num + injection_num))

        return newadj


    def attack(self, test_nodes,injection_ratio, n_edge_max, epoch_update_features, epoch_select_edges,
               add_homophily = False,lr_sec=1e-3, epoch_sec=1000,ceta=1, lbth=1.5,m=5):
        self.injection_ratio = injection_ratio

        if self.feature_type:
            feat_lim_min = 0
            feat_lim_max = 1
            feat_lim_min_ = 0 / 5
            feat_lim_max_ = 1 / 5
        else:
            feat_lim_min = torch.from_numpy(self.features.min(0).astype('double')).float().to(self.device)
            feat_lim_max = torch.from_numpy(self.features.max(0).astype('double')).float().to(self.device)
            feat_lim_min_ = feat_lim_min / 5
            feat_lim_max_ = feat_lim_max / 5

        injection_num = int(test_nodes.shape[0] * injection_ratio)
        print("injection_ratio: {},injection_num: {}, n_edge_max: {}\n".format(injection_ratio,
                                                                             injection_num,n_edge_max))
        has_inj_num =0
        pre_logits_pro = self.logits_pro
        spendtime = 0
        for attack_class in range(self.n_class):
            attack_class_indx = np.where(self.labels_np[test_nodes] == attack_class)[0]
            true_class_indx = np.where(self.groundtruth_labels_np[test_nodes] == attack_class)[0]
            target_nodes = test_nodes[attack_class_indx]
            target_nodes_tensor = torch.LongTensor(target_nodes).to(self.device)
            class_nodes = test_nodes[true_class_indx]
            target_num = target_nodes.shape[0]
            class_num = class_nodes.shape[0]

            if attack_class == self.n_class - 1:
                cur_injection_num = injection_num - has_inj_num
            else:
                cur_injection_num = round(target_num * injection_ratio)

            if cur_injection_num <= 0 or target_num<=0:
                continue
            print("attack class: {}. target nodes num: {},class nodes num: {},injection number: {}".
                  format(attack_class,target_num,class_nodes.shape[0],cur_injection_num))
            acc_target = accuracy(self.logp[target_nodes], self.labels_tensor[target_nodes])

            acc_true_target = accuracy(self.logp[target_nodes], self.groundtruth_labels_tensor[target_nodes])
            acc_true_class = accuracy(self.logp[class_nodes], self.groundtruth_labels_tensor[class_nodes])
            print("target nodes number: {},surr acc: {:.4%}, true acc: {:.4%}".format(target_num, acc_target,
                                                                                      acc_true_target))
            print("class nodes number: {}, true acc: {:.4%}".format(class_num, acc_true_class))
            first_target = target_nodes

            fulladj_attack = self.generate_fulladj_attack(self.adj_attack, 1, first_target)

            edge_index_full, _ = utils.from_scipy_sparse_matrix(fulladj_attack)
            edge_index_full = edge_index_full.to(self.device)
            inj_features = self.generate_feature(1,self.surrogate_model, edge_index_full,
                                                 self.features_attack_tensor,target_nodes_tensor,
                                                 feature_type=self.feature_type)

            inj_features_tensor = torch.from_numpy(inj_features.astype('double')).float().to(self.device)

            inj_features_tensor = self.update_features(self.surrogate_model, edge_index_full, self.features_attack_tensor,
                                                       inj_features_tensor, target_nodes_tensor,
                                                       n_epoch=epoch_update_features,
                                                       feat_lim_min=feat_lim_min_, feat_lim_max=feat_lim_max_)

            inj_features_tensor = inj_features_tensor.squeeze()
            inj_features_tensor =inj_features_tensor.repeat(cur_injection_num).reshape(cur_injection_num, self.features_dim)
            first_target_tensor = torch.LongTensor(first_target).to(self.device)
            adj_attack_tensor = sparse_mx_to_torch_sparse_tensor(self.adj_attack)

            features_attack_tensor = torch.cat((self.features_attack_tensor, inj_features_tensor), dim=0)
            t0 = time.time()

            adj_attack_tensor,pre_logits_pro = self.generate_adj_attack_grad(self.surrogate_model, adj_attack_tensor,
                                                              features_attack_tensor, cur_injection_num, first_target_tensor,
                                                              target_nodes_tensor, pre_logits_pro, n_edge_max,
                                                              m=m,aerfa=10, epsilon=0.02,
                                                              epoch=epoch_select_edges)
            spendtime += (time.time() - t0)
            edge_index, _ = utils.to_edge_index(adj_attack_tensor)

            self.adj_attack = utils.to_scipy_sparse_matrix(edge_index, num_nodes=self.nodes_num_attack).tocsr()
            logits = self.surrogate_model(features_attack_tensor, edge_index)
            logp = F.log_softmax(logits, dim=1)
            target_acc = accuracy(logp[target_nodes], self.labels_tensor[target_nodes])
            acc_true_target_ = accuracy(logp[target_nodes], self.groundtruth_labels_tensor[target_nodes])
            acc_true_class_ = accuracy(logp[class_nodes], self.groundtruth_labels_tensor[class_nodes])
            print("After update edges...")
            print("All injection done adj_attack_tensor.shape: ", adj_attack_tensor.shape)
            print("Injection done surrogate target nodes surr acc: {:.4%},"
                  "target nodes true acc: {:.4%},class nodes true acc: {:.4%}".format(target_acc,acc_true_target_,acc_true_class_))
            print("Injection done surrogate target nodes true acc down: {:.4%},"
                  "class nodes true acc down: {:.4%}".format(acc_true_target - acc_true_target_,acc_true_class - acc_true_class_))


            adj_attack_de = utils.to_scipy_sparse_matrix(edge_index, num_nodes=self.nodes_num_attack).tocsr()
            select_nodes = []
            for i in range(cur_injection_num):
                inj_node = self.nodes_num_attack-cur_injection_num + i
                select_nodes.append(get_neighbor(adj_attack_de, inj_node))
            select_nodes = np.concatenate(select_nodes)

            select_nodes = np.unique(select_nodes)
            degree_select = self.adj[select_nodes].sum(1).reshape(1, -1)
            cur_num = self.nodes_num_attack-cur_injection_num


            inj_features_tensor = self.sec_update_features(self.surrogate_model, edge_index, self.features_attack_tensor,
                                                           inj_features_tensor,
                                                           target_nodes_tensor, feat_lim_min,
                                                           feat_lim_max,cur_num,lbth=lbth,
                                                           n_epoch=epoch_sec, lr=lr_sec,ceta=ceta,
                                                           add_homophily=add_homophily)

            features_attack_tensor = torch.cat((self.features_attack_tensor, inj_features_tensor), dim=0)
            self.features_attack_tensor = features_attack_tensor

            has_inj_num += cur_injection_num
            logits = self.surrogate_model(self.features_attack_tensor, edge_index)
            logp = F.log_softmax(logits, dim=1)
            target_acc = accuracy(logp[target_nodes], self.labels_tensor[target_nodes])
            acc_true_target_ = accuracy(logp[target_nodes], self.groundtruth_labels_tensor[target_nodes])
            acc_true_class_ = accuracy(logp[class_nodes], self.groundtruth_labels_tensor[class_nodes])

            print("After second update features...")
            print("All injection done adj_attack_tensor.shape: ", adj_attack_tensor.shape)
            print("Injection done surrogate target nodes surr acc: {:.4%},"
                  "target nodes true acc: {:.4%}, class nodes true acc: "
                  "{:.4%}".format(target_acc,acc_true_target_,acc_true_class_))
            print("Injection done surrogate target nodes true acc down: {:.4%},"
                  "class nodes true acc down: {:.4%}".format(
                acc_true_target - acc_true_target_,acc_true_class - acc_true_class_))

            print("class {} has been inject {:04d} nodes, "
                  "total has been inject {} nodes!\n".format(attack_class, cur_injection_num,has_inj_num))

        print("Spend time: {}".format(spendtime))
        target_nodes = test_nodes
        target_nodes_tensor = torch.LongTensor(test_nodes).to(self.device)
        test_num = target_nodes.shape[0]


        acc_target = accuracy(self.logp[target_nodes], self.labels_tensor[target_nodes])
        acc_true_target = accuracy(self.logp[target_nodes], self.groundtruth_labels_tensor[target_nodes])
        print("test nodes number: {},surr acc: {:.4%}, true acc: {:.4%}".format(test_num, acc_target,
                                                                                  acc_true_target))
        logits = self.surrogate_model(self.features_attack_tensor, edge_index)
        logp = F.log_softmax(logits, dim=1)
        target_acc = accuracy(logp[target_nodes], self.labels_tensor[target_nodes])
        acc_true_target_ = accuracy(logp[target_nodes], self.groundtruth_labels_tensor[target_nodes])
        print("After sec update features...")
        print("All injection done adj_attack_tensor.shape: ", adj_attack_tensor.shape)
        print("All injection done features_attack_tensor.shape: ", self.features_attack_tensor.shape)
        print("Injection done surrogate test nodes surr acc: {:.4%}".format(target_acc))
        print("Injection done surrogate test nodes true acc: {:.4%}".format(acc_true_target_))
        print("Injection done surrogate test nodes true acc down: {:.4%}".format(
            acc_true_target - acc_true_target_))

        features_attack = self.features_attack_tensor.cpu().numpy()
        return self.adj_attack, features_attack

    def random_generate_features(self,features_np,num,feature_type):
        if feature_type:
            features_dim = features_np.shape[1]
            features_generate = np.zeros((num, features_dim), dtype=int)
            select_num = int(features_dim * 0.15)
            for i in range(num):
                nonzero_dim = np.random.choice(features_dim, select_num, replace=False)
                features_generate[i, nonzero_dim] = 1
            features_generate = torch.from_numpy(features_generate.astype('double')).float()
        else:
            features_dim = features_np.shape[1]
            features_generate = np.zeros((num, features_dim), dtype=float)
            mu_noise = np.zeros(features_dim)
            sigma_noise = np.ones(features_dim)
            mu = features_np.mean(0)
            for i in range(num):
                features_generate[i] = mu + np.random.normal(mu_noise, sigma_noise)
            features_generate = torch.from_numpy(features_generate.astype('double')).float()
        return features_generate

    def generate_adj_attack_grad(self, surro_model, adj_attack_tensor, features_attack_tensor,
                                 injection_num, select_target, target_nodes, pre_logits_pro,
                                 n_edge_max,m=5, aerfa=10,epsilon=0.01,epoch=600):
        cur_num = self.nodes_num_attack
        target_nodes_num = target_nodes.shape[0]
        select_target_num = select_target.shape[0]
        edge_index, _ = utils.to_edge_index(adj_attack_tensor)
        edge_index = edge_index.to(self.device)
        cur_features_attack_tensor = features_attack_tensor[0:cur_num]
        logits = surro_model(cur_features_attack_tensor, edge_index)
        pred = F.softmax(logits, dim=1)
        pred_labels_tensor = pred.max(1)[1].type_as(self.labels_tensor)

        addscore = torch.zeros(select_target_num).to(self.device)

        deg = torch.tensor(self.n_degree).to(self.device) + 1.0

        for i in range(select_target_num):
            it = select_target[i]

            label = pred_labels_tensor[it]
            score = pred[it][label] * self.mean_degree
            addscore_ = score / deg[it]
            addscore[i] = addscore_
        addscore = torch.sigmoid(addscore)
        cannot_link = np.array([], dtype=np.int64)
        addscore_ = addscore
        select_target_ = select_target
        select_target_num_ = select_target_num
        for i in tqdm(range(injection_num)):
            idx_remain = np.arange(select_target_num_)
            injection_nodes = torch.arange(cur_num, cur_num + 1)
            new_injx = torch.repeat_interleave(injection_nodes, select_target_num_).to(self.device)

            new_injy = select_target_.to(self.device)

            assert new_injx.size() == new_injy.size()

            indices_cur = adj_attack_tensor.indices().to(self.device)
            values_cur = adj_attack_tensor.values().to(self.device)

            new_row = torch.cat((indices_cur[0], new_injx, new_injy), dim=0)
            new_col = torch.cat((indices_cur[1], new_injy, new_injx), dim=0)
            k = m
            injvals = torch.zeros(new_injx.shape[0], dtype=torch.float64).to(self.device)
            injvals.requires_grad_(True)
            scores = torch.zeros(new_injx.shape[0]).to(self.device)
            injvals_ = torch.ones(new_injx.shape[0], dtype=torch.float64).to(self.device)
            for j in range(1, k + 1):
                with torch.no_grad():
                    injvals -= injvals
                    injvals += (injvals_ * j / k)
                new_adj_attack = SparseTensor(row=new_row, col=new_col,
                                              value=torch.cat((values_cur, injvals, injvals), dim=0),
                                              sparse_sizes=(cur_num + 1, cur_num + 1))

                logits = surro_model(features_attack_tensor, new_adj_attack)
                logp = F.log_softmax(logits, dim=1)
                pred_loss = F.nll_loss(logp[:self.nodes_num][target_nodes], self.labels_tensor[target_nodes]).to(
                    self.device)
                adj_meta_grad = torch.autograd.grad(pred_loss, injvals, retain_graph=True)[0]
                grad_score = adj_meta_grad
                scores += grad_score
            scores = scores / k
            tmp_vals = torch.sigmoid(scores)
            tmp_vals[scores <= 0] = -1

            tmp_vals = -tmp_vals
            tmp_vals = tmp_vals - addscore_
            sel_idx = tmp_vals.argsort(dim=-1)[:n_edge_max]
            sel_mask = torch.zeros(tmp_vals.size()).bool()
            sel_mask[sel_idx] = True
            sel_idx = torch.nonzero(sel_mask.view(-1)).squeeze()
            select_node = select_target_[sel_idx]
            cannot_link = np.append(cannot_link, select_node.cpu().numpy())
            new_injx = new_injx[sel_idx]
            new_injy = new_injy[sel_idx]

            new_row = torch.cat((indices_cur[0], new_injx, new_injy), dim=0)
            new_col = torch.cat((indices_cur[1], new_injy, new_injx), dim=0)
            new_adj_attack = SparseTensor(row=new_row, col=new_col,
                                          value=torch.ones(new_row.size(0), device=self.device),
                                          sparse_sizes=(cur_num + 1, cur_num + 1))
            edge_index, _ = utils.to_edge_index(new_adj_attack)
            adj_attack_tensor = utils.to_torch_coo_tensor(edge_index,
                                                     size=(cur_num + 1, cur_num + 1))
            cur_num +=1

            pre_select_np = select_target_.cpu().numpy()
            pre_select_np = np.setdiff1d(pre_select_np,cannot_link)
            if pre_select_np.size >= n_edge_max:
                select_target_ = torch.LongTensor(pre_select_np).to(self.device)
                select_target_num_ = select_target_.shape[0]
                idx_remain= np.setdiff1d(idx_remain,sel_idx.cpu().numpy())
                addscore_ = addscore_[idx_remain]
            else:
                select_target_ = select_target
                select_target_num_ = select_target.shape[0]
                addscore_ = addscore
                cannot_link = np.array([], dtype=np.int64)

        self.nodes_num_attack = cur_num

        return adj_attack_tensor,pre_logits_pro

    def update_features(self, surr_model, edge_index, features_tensor, inj_features_tensor, target_nodes,
                        feat_lim_min, feat_lim_max, n_epoch=600, epsilon=1):
        surr_model.eval()
        injection_num = inj_features_tensor.shape[0]

        feature_add = Variable(inj_features_tensor, requires_grad=True)

        optimizer = torch.optim.Adam([{'params': [feature_add]}], lr=epsilon)
        for i in tqdm(range(n_epoch)):
            feature_add_ = torch.sin(feature_add) * (feat_lim_max - feat_lim_min) / 2 + (feat_lim_max + feat_lim_min) / 2
            features_concat = torch.cat((features_tensor, feature_add_), dim=0)
            pred = surr_model(features_concat, edge_index)
            pred = F.log_softmax(pred, dim=1)
            pred_loss = F.nll_loss(pred[:self.nodes_num][target_nodes], self.labels_tensor[target_nodes])
            pred_loss = -pred_loss
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

        inj_features_tensor_ = features_concat[self.nodes_num_attack:].detach()
        return inj_features_tensor_

    def sec_update_features(self, surr_model, edge_index, features_tensor, inj_features_tensor, target_nodes,
                            feat_lim_min, feat_lim_max, cur_num,
                            lbth=1.5, n_epoch=1500, lr=1e-3,ceta=1, add_homophily=False):
        if self.feature_type:
            ceta = ceta
        else:
            ceta = ceta
        target_num = target_nodes.shape[0]
        original_num = cur_num
        inj_num = inj_features_tensor.shape[0]
        inj_nodes = torch.arange(original_num, original_num + inj_num)
        random_idx = torch.randperm(inj_num)
        ori_nodes = torch.arange(original_num)
        ori_edge_index = self.edge_index

        batch_size = min(inj_num, round(target_num * 0.03))
        if not self.feature_type:
            cur_mu = self.mu.repeat(batch_size).reshape(batch_size, -1)
            cur_sigma = self.sigma.repeat(batch_size).reshape(batch_size, -1)
        class_inj_num = 0
        target_nodes_ = target_nodes
        update_sum = math.ceil(inj_num / batch_size)


        dim = torch.tensor(self.features_dim, dtype=torch.float).to(self.device)
        batch_edge_index = []
        batch_features_attack = []
        batch_edge_index_norm = []
        batch_edge_weight_norm = []
        batch_all_edge_index = []
        batch_all_subset = []
        batch_remains = []
        batch_all_mapping = []

        while class_inj_num < inj_num:

            will_inj_num = min(inj_num, class_inj_num + batch_size)
            curr_inj = will_inj_num - class_inj_num
            if curr_inj >= batch_size:
                select_idx = random_idx[class_inj_num:class_inj_num + curr_inj]
            else:
                curr_inj = batch_size
                select_idx = random_idx[inj_num - batch_size:inj_num]
            select_idx = torch.sort(select_idx)[0]
            cur_inj_nodes = inj_nodes[select_idx]
            cur_inj_features_tensor = inj_features_tensor[select_idx]

            batch_features_attack.append(cur_inj_features_tensor)

            subset_nodes = torch.cat((ori_nodes, cur_inj_nodes), dim=0)
            inj_edge_index = get_inj_edge_index(subset_nodes, edge_index, original_num,inj_num).to(self.device)

            batch_edge_index.append(inj_edge_index)

            inj_nodes_relocation = torch.arange(original_num, curr_inj + original_num)
            cur_edge_index = torch.cat((ori_edge_index, inj_edge_index), dim=1)
            if add_homophily:
                edge_index_norm, edge_weight_norm = gcn_norm(cur_edge_index, num_nodes=curr_inj + original_num,
                                                             add_self_loops=False)
                batch_edge_index_norm.append(edge_index_norm)
                batch_edge_weight_norm.append(edge_weight_norm)

            all_subset = []
            all_edge_index = []
            all_mapping = []
            for i in range(curr_inj):
                subset, edge_index_, mapping, edge_mask = utils.k_hop_subgraph(inj_nodes_relocation[i].item(), 1, cur_edge_index,
                                                                               relabel_nodes=True,
                                                                               num_nodes=curr_inj + original_num,
                                                                               directed=False)
                all_subset.append(subset)
                all_edge_index.append(edge_index_)
                all_mapping.append(mapping)


            remains = []
            for idx in range(curr_inj):
                subset = all_subset[idx]
                map_inj = all_mapping[idx]
                remain = torch.arange(subset.shape[0]).to(self.device)
                inj = torch.nonzero(remain == map_inj).squeeze()
                remains.append(torch.cat((remain[:inj], remain[inj + 1:])))
            batch_all_edge_index.append(all_edge_index)
            batch_all_subset.append(all_subset)
            batch_remains.append(remains)
            batch_all_mapping.append(all_mapping)
            class_inj_num = will_inj_num

        surr_model.eval()
        feature_generator = Feature_Generator(self.features_dim, 256, self.feature_type, feat_lim_min,
                                              feat_lim_max)
        feature_generator = feature_generator.to(self.device)
        optimizer = torch.optim.Adam(list(feature_generator.parameters()), lr=lr,
                                     weight_decay=1e-3)
        dur = []
        m = 1
        m_h = 0
        threshold = torch.tensor([0.1, 0.3, 0.6, 1.0]).to(self.device)

        lbth_ = lbth
        max_loss = 1e7
        best_weights = None
        for e in range(n_epoch):
            t0 = time.time()
            select_sub_idx = e % update_sum
            inj_edge_index = batch_edge_index[select_sub_idx].to(self.device)
            cur_inj_features_tensor = batch_features_attack[select_sub_idx].to(self.device)
            all_edge_index = batch_all_edge_index[select_sub_idx]
            all_subset = batch_all_subset[select_sub_idx]
            remains = batch_remains[select_sub_idx]
            all_mapping = batch_all_mapping[select_sub_idx]

            old_features_tensor_attacked = torch.cat((features_tensor, cur_inj_features_tensor), dim=0)
            cur_edge_index = torch.cat((ori_edge_index, inj_edge_index), dim=1)
            if (e % 50 == 0):

                m = threshold[e // 50] if e < 200 else threshold[-1]
                m_h = 0 if e < 200 else 1

            feature_buffer, num_feat,homophily = feature_generator(old_features_tensor_attacked, all_edge_index,
                                                         all_subset, remains,
                                                         all_mapping, batch_size)
            if self.feature_type:

                bud = torch.full(size=(batch_size,), fill_value=self.feature_budget / self.features_dim,
                                 dtype=torch.float).to(self.device)
                feature_loss = dim/2 * F.mse_loss(num_feat, bud, reduction='mean')

            else:
                feature_loss = (F.mse_loss(cur_mu, num_feat[0], reduction='mean') +
                                      F.mse_loss(cur_sigma, num_feat[1], reduction='mean'))
            features_attack_tensor = torch.cat((features_tensor, feature_buffer), dim=0)
            if add_homophily:
                homophily_loss = m_h * ceta * homophily
            else:
                homophily_loss = 0
            pred = surr_model(features_attack_tensor, cur_edge_index)
            pred = F.log_softmax(pred, dim=1)
            pred_loss = F.nll_loss(pred[target_nodes_], self.labels_tensor[target_nodes_],
                                   reduction='none').to(self.device)
            if self.feature_type:
                pred_loss = F.relu(-pred_loss + lbth_) ** 2
            else:
                pred_loss = F.relu(-pred_loss + lbth_) ** 2
            feature_loss = m * feature_loss
            loss_sum = feature_loss + pred_loss.mean() - homophily_loss
            acc = accuracy(pred[target_nodes], self.labels_tensor[target_nodes])
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            dur.append(time.time() - t0)
            if e >= 300 and (e % 50 == 0):
                if max_loss > loss_sum:
                    max_loss = loss_sum
                    best_weights = deepcopy(feature_generator.state_dict())

            if e % 100 == 0:
                if add_homophily:
                    print(
                        "epoch {:4d}, Loss: {:.4f}, feature loss: {:.4f},pred_loss: {:.4f},homophily: {:.4f},"
                        "acc: {:.4%}, Time(s) {:.4f}".format(e, loss_sum, feature_loss, pred_loss.mean(), homophily_loss,
                                                             acc, np.mean(dur)))
                else:
                    print("epoch {:4d}, Loss: {:.4f}, feature_loss: {:.4f}, pred_loss: {:.4f}, "
                          "acc: {:.4%}, Time(s) {:.4f}".format(e, loss_sum,
                                                               feature_loss, pred_loss.mean(), acc,
                                                               np.mean(dur)))
        feature_generator.load_state_dict(best_weights)

        feature_generator.eval()
        old_features_tensor_attacked = torch.cat((features_tensor, inj_features_tensor), dim=0)
        all_subset = []
        all_edge_index = []
        all_mapping = []

        for i in range(inj_num):
            subset, edge_index_, mapping, edge_mask = utils.k_hop_subgraph(inj_nodes[i].item(), 1, edge_index,
                                                                           relabel_nodes=True,
                                                                           num_nodes=inj_num + original_num,
                                                                           directed=False)
            all_subset.append(subset)
            all_edge_index.append(edge_index_)
            all_mapping.append(mapping)
        remains = []
        for idx in range(inj_num):
            subset = all_subset[idx]
            map_inj = all_mapping[idx]
            remain = torch.arange(subset.shape[0]).to(self.device)
            inj = torch.nonzero(remain == map_inj).squeeze()
            remains.append(torch.cat((remain[:inj], remain[inj + 1:])))
        feature_buffer, num_feat,_ = feature_generator(old_features_tensor_attacked, all_edge_index, all_subset,
                                                     remains,
                                                     all_mapping, inj_num)
        new_inj_attack_tensor = feature_buffer.detach()
        if self.feature_type:
            new_inj_attack_tensor[new_inj_attack_tensor>=0.5] = 1
            new_inj_attack_tensor[new_inj_attack_tensor<0.5] = 0

        return new_inj_attack_tensor


