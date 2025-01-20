import numpy as np
import argparse
from model import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from SGNN_models import SGNN
import os
import time
from collections import OrderedDict
from utils import *
from utils_data import *
from dgl.nn import GATConv
from sklearn.feature_extraction.text import TfidfTransformer
import torch
from torch.autograd import Variable
import torch.utils.data as utils
import numpy as np
import copy
import time

parser = argparse.ArgumentParser()
parser.add_argument('--scity', type=str, default='NY')
parser.add_argument('--tcity', type=str, default='DC')
parser.add_argument('--dataname', type=str, default='Taxi', help='Within [Bike, Taxi]')
parser.add_argument('--datatype', type=str, default='pickup', help='Within [pickup, dropoff]')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument("--model", type=str, default='STNet_nobn', help='Within [STResNet, STNet, STNet_nobn]')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of source training epochs')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
parser.add_argument('--data_amount', type=int, default=3, help='0: full data, 30/7/3 correspond to days of data')
parser.add_argument('--sinneriter', type=int, default=3, help='Number of inner iterations (source) for meta learning')
parser.add_argument('--tinneriter', type=int, default=1, help='Number of inner iterations (target) for meta learning')
parser.add_argument('--innerlr', type=float, default=5e-5, help='Learning rate for inner loop of meta-learning')
parser.add_argument('--outeriter', type=int, default=20, help='Number of outer iterations for meta-learning')
parser.add_argument('--outerlr', type=float, default=1e-4, help='Learning rate for the outer loop of meta-learning')
parser.add_argument('--topk', type=int, default=15)
parser.add_argument('--mmd_w', type=float, default=2, help='mmd weight')
parser.add_argument('--et_w', type=float, default=2, help='edge classifier weight')
parser.add_argument("--ma_coef", type=float, default=0.6, help='Moving average parameter for source domain weights')
parser.add_argument("--weight_reg", type=float, default=1e-3, help="Regularizer for the source domain weights.")
parser.add_argument("--pretrain_iter", type=int, default=-1, help='Pre-training iterations per pre-training epoch. ')
parser.add_argument('-d', default='PEMS-BAY', type=str, help="specify dataset")
parser.add_argument('-m', default=0.4, type=float, help="specify missing rate")
parser.add_argument('-o', default='Adam', type=str, help="specify training optimizer")
parser.add_argument('-l', default=0.001, type=float, help="specify initial learning rate")
parser.add_argument('-g', default=0.9, type=float, help="specify gamma for GMN")
parser.add_argument('-t', default='random', type=str, help="specify masking type")
parser.add_argument('-r', default=1024, type=int, help="specify random seed")
parser.add_argument('-s', default=1, type=int, help="specify whether save model")
args = parser.parse_args()


if args.seed != -1:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



dataname = args.dataname
scity = args.scity
tcity = args.tcity
datatype = args.datatype
num_epochs = args.num_epochs
start_time = time.time()
dataset = args.d
missing_rate = args.m
optm = args.o
learning_rate = args.l
gamma = args.g
random_seed = args.r
masking_type = args.t
save_model = args.s
data_amount = args.data_amount




def loadDataset(dataset=None):
    if dataset == 'PEMS-BAY':
        speed_matrix = pd.read_hdf('../data/PEMS-BAY/pems-bay.h5')
        A = pd.read_pickle('../data/PEMS-BAY/adj_mx_bay.pkl')
        A = A[2]
        A[np.where(A != 0)] = 1
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[0]):
                if A[i, j] == 1:
                    A[j, i] = 1
    elif dataset == 'METR-LA':
        speed_matrix = pd.read_hdf('../data/METR-LA/metr-la.h5')
        A = pd.read_pickle('../data/METR-LA/adj_mx.pkl')
        A = A[2]
        A[np.where(A != 0)] = 1
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[0]):
                if A[i, j] == 1:
                    A[j, i] = 1
    elif dataset == 'LOOP-SEA':
        speed_matrix = pd.read_pickle('../data/LOOP-SEA/speed_matrix_2015_1mph')
        A = np.load('../data/LOOP-SEA/Loop_Seattle_2015_A.npy')
    elif dataset == 'INRIX-SEA':
        speed_matrix = pd.read_pickle('../data/INRIX-SEA/INRIX_Seattle_Speed_Matrix__2012_v2.pkl')
        A = np.load('../data/INRIX-SEA/INRIX_Seattle_Adjacency_matrix_2012_v2.npy')
    else:
        print('Dataset not found.')
        return None, None
    print('Dataset loaded.')
    return speed_matrix, A


speed_matrix_pems, A_pems = loadDataset(dataset=dataset)
speed_matrix_la, A_la = loadDataset(dataset='METR-LA')


mask_ones_proportion = 1 - missing_rate
from utils_data import PrepareDataset

train_dataloader_pems, valid_dataloader_pems, test_dataloader_pems, max_speed_pems, X_mean_pems, pems_train_data, pems_train_label = PrepareDataset(
    speed_matrix_pems, BATCH_SIZE=64, seq_len=10, pred_len=1, \
    train_propotion=0.6, valid_propotion=0.2, \
    mask_ones_proportion=mask_ones_proportion, \
    masking=True, masking_type=masking_type, delta_last_obsv=False, \
    shuffle=True, random_seed=random_seed, data_amount=0)  # data_amount用于控制时间缺失，预训练不缺失
train_dataloader_la, valid_dataloader_la, test_dataloader_la, max_speed_la, X_mean_la, la_train_data, la_train_label = PrepareDataset(
    speed_matrix_la, BATCH_SIZE=64, seq_len=10, pred_len=1, \
    train_propotion=0.6, valid_propotion=0.2, \
    mask_ones_proportion=mask_ones_proportion, \
    masking=True, masking_type=masking_type, delta_last_obsv=False, \
    shuffle=True, random_seed=random_seed, data_amount=data_amount)  # data_amount用于控制时间缺失，微调缺失
inputs, labels = next(iter(train_dataloader_pems))
[batch_size, type_size, step_size, fea_size] = inputs.size()

def graphs_to_edge_labels(graphs):
    edge_label_dict = {}
    for i, graph in enumerate(graphs):
        src, dst = graph.edges()
        for s, d in zip(src, dst):
            s = s.item()
            d = d.item()
            if (s, d) not in edge_label_dict:
                edge_label_dict[(s, d)] = np.zeros(len(graphs))
            edge_label_dict[(s, d)][i] = 1
    edges = []
    edge_labels = []
    for k in edge_label_dict.keys():
        edges.append(k)
        edge_labels.append(edge_label_dict[k])
    edges = np.array(edges)
    edge_labels = np.array(edge_labels)
    return edges, edge_labels




pems_road_adj = A_pems
la_road_adj = A_la
transform = TfidfTransformer()


def build_sgnn_prox_graph():
    adj_prox = np.zeros((207, 207))
    for i in range(207):
        for j in [i - 1, i, i + 1]:
            if j is not None and j != 207:
                adj_prox[i, j] = 1
                adj_prox[j, i] = 1
    adj_prox[0, 206] = 0
    adj_prox[206, 0] = 0
    return adj_prox


pems_prox_adj = add_self_loop(build_sgnn_prox_graph())
la_prox_adj = add_self_loop(build_sgnn_prox_graph())
pems_feat = pd.read_pickle('../data/PEMS-BAY/adj_mx_bay.pkl')
pems_feat = pems_feat[2]
pems_feat = pems_feat[:cut_sensors, :cut_sensors]
la_feat = pd.read_pickle('../data/METR-LA/adj_mx.pkl')
la_feat = la_feat[2]
pems_norm_feat = np.array(transform.fit_transform(pems_feat).todense())
la_norm_feat = np.array(transform.fit_transform(la_feat).todense())
pems_feat_adj, pems_feat_cos = build_poi_graph(pems_norm_feat, args.topk)
la_feat_adj, la_feat_cos = build_poi_graph(la_norm_feat, args.topk)
pems_feat_adj = add_self_loop(pems_feat_adj)
la_feat_adj = add_self_loop(la_feat_adj)
pems_graphs = adjs_to_graphs([pems_road_adj, pems_prox_adj, pems_feat_adj])
la_graphs = adjs_to_graphs([la_road_adj, la_prox_adj, la_feat_adj])
for i in range(len(pems_graphs)):
    pems_graphs[i] = pems_graphs[i].to(device)
    la_graphs[i] = la_graphs[i].to(device)
pems_edges, pems_edge_labels = graphs_to_edge_labels(pems_graphs)
la_edges, la_edge_labels = graphs_to_edge_labels(la_graphs)




class MVGAT(nn.Module):
    def __init__(self, num_graphs=3, num_gat_layer=2, in_dim=14, hidden_dim=64, emb_dim=32, num_heads=2, residual=True):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_gat_layer = num_gat_layer
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.residual = residual

        self.multi_gats = nn.ModuleList()
        for j in range(self.num_gat_layer):
            gats = nn.ModuleList()
            for i in range(self.num_graphs):
                if j == 0:
                    gats.append(GATConv(self.in_dim,
                                        self.hidden_dim,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
                elif j == self.num_gat_layer - 1:
                    gats.append(GATConv(self.hidden_dim * self.num_heads,
                                        self.emb_dim // self.num_heads,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
                else:
                    gats.append(GATConv(self.hidden_dim * self.num_heads,
                                        self.hidden_dim,
                                        self.num_heads,
                                        residual=self.residual,
                                        allow_zero_in_degree=True))
            self.multi_gats.append(gats)

    def forward(self, graphs, feat):
        views = []
        for i in range(self.num_graphs):
            for j in range(self.num_gat_layer):
                if j == 0:
                    z = self.multi_gats[j][i](graphs[i], feat)
                else:
                    z = self.multi_gats[j][i](graphs[i], z)
                if j != self.num_gat_layer - 1:
                    z = F.relu(z)
                z = z.flatten(1)
            views.append(z)
        return views


class FusionModule(nn.Module):
    def __init__(self, num_graphs, emb_dim, alpha):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.alpha = alpha

        self.fusion_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.self_q = nn.ModuleList()
        self.self_k = nn.ModuleList()
        for i in range(self.num_graphs):
            self.self_q.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.self_k.append(nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, views):
        # run fusion by self attention
        cat_views = torch.stack(views, dim=0)
        self_attentions = []
        for i in range(self.num_graphs):
            Q = self.self_q[i](cat_views)
            K = self.self_k[i](cat_views)
            # (3, num_nodes, 64)
            attn = F.softmax(torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim), dim=-1)
            # (3, num_nodes, num_nodes)
            output = torch.matmul(attn, cat_views)
            self_attentions.append(output)
        self_attentions = sum(self_attentions) / self.num_graphs
        # (3, num_nodes, 64 * 2)
        for i in range(self.num_graphs):
            views[i] = self.alpha * self_attentions[i] + (1 - self.alpha) * views[i]

        # further run multi-view fusion
        mv_outputs = []
        for i in range(self.num_graphs):
            mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])

        fused_outputs = sum(mv_outputs)
        # next_in = [(view + fused_outputs) / 2 for view in views]
        return fused_outputs, [(views[i] + fused_outputs) / 2 for i in range(self.num_graphs)]


class Scoring(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.score = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.emb_dim // 2, self.emb_dim // 2))

    def forward(self, source_emb, target_emb):
        target_context = torch.tanh(self.score(target_emb).mean(0))
        source_trans_emb = self.score(source_emb)
        source_score = (source_trans_emb * target_context).sum(1)
        # the following lines modify inner product similarity to cosine similarity
        # target_norm = target_context.pow(2).sum().pow(1/2)
        # source_norm = source_trans_emb.pow(2).sum(1).pow(1/2)
        # source_score /= source_norm
        # source_score /= target_norm
        # print(source_score)
        return F.relu(torch.tanh(source_score))


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


mmd = MMD_loss()


class EdgeTypeDiscriminator(nn.Module):
    def __init__(self, num_graphs, emb_dim):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.edge_network = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.emb_dim, self.num_graphs))

    def forward(self, src_embs, dst_embs):
        edge_vec = torch.cat([src_embs, dst_embs], dim=1)
        return self.edge_network(edge_vec)



num_gat_layers = 2
in_dim = 207
hidden_dim = 64
emb_dim = 64
num_heads = 2
mmd_w = args.mmd_w
et_w = args.et_w
ma_param = args.ma_coef


mvgat = MVGAT(len(pems_graphs), num_gat_layers, in_dim, hidden_dim, emb_dim, num_heads, True).to(device)
fusion = FusionModule(len(pems_graphs), emb_dim, 0.8).to(device)
scoring = Scoring(emb_dim).to(device)
edge_disc = EdgeTypeDiscriminator(len(pems_graphs), emb_dim).to(device)
mmd = MMD_loss()
model = args.model



emb_param_list = list(mvgat.parameters()) + list(fusion.parameters()) + list(edge_disc.parameters())
emb_optimizer = optim.Adam(emb_param_list, lr=args.learning_rate, weight_decay=args.weight_decay)
meta_optimizer = optim.Adam(scoring.parameters(), lr=args.outerlr, weight_decay=args.weight_decay)
best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999



def evaluate(net_, loader, spatial_mask):
    net_.eval()
    with torch.no_grad():
        se = 0
        ae = 0
        valid_points = 0
        for it_ in loader:
            if len(it_) == 2:
                (x, y) = it_
            elif len(it_) == 4:
                _, _, x, y = it_
            x = x.to(device)
            y = y.to(device)
            lng = x.shape[2]
            lat = x.shape[3]
            out = net_(x, spatial_mask=spatial_mask.bool())
            valid_points += x.shape[0] * spatial_mask.sum().item()
            if len(out.shape) == 4:  # STResNet
                se += (((out - y) ** 2) * (spatial_mask.view(1, 1, lng, lat))).sum().item()
                ae += ((out - y).abs() * (spatial_mask.view(1, 1, lng, lat))).sum().item()
            elif len(out.shape) == 3:  # STNet
                batch_size = y.shape[0]
                lag = y.shape[1]
                y = y.view(batch_size, lag, -1)[:, :, spatial_mask.view(-1).bool()]
                # print("out", out.shape)
                # print("y", y.shape)
                se += ((out - y) ** 2).sum().item()
                ae += (out - y).abs().sum().item()
    return np.sqrt(se / valid_points), ae / valid_points


def batch_sampler(tensor_list, batch_size):
    num_samples = tensor_list[0].size(0)
    idx = np.random.permutation(num_samples)[:batch_size]
    return (x[idx] for x in tensor_list)


def get_weights_bn_vars(module):
    fast_weights = OrderedDict(module.named_parameters())
    bn_vars = OrderedDict()
    for k in module.state_dict():
        if k not in fast_weights.keys():
            bn_vars[k] = module.state_dict()[k]
    return fast_weights, bn_vars


def train_epoch(net_, loader_, optimizer_, weights=None, mask=None, num_iters=None):
    net_.train()
    epoch_loss = []
    for i, (x, y) in enumerate(loader_):
        x = x.to(device)
        y = y.to(device)
        out = net_(x, spatial_mask=mask.bool())
        if len(out.shape) == 4:  # STResNet
            eff_batch_size = y.shape[0]
            loss = ((out - y) ** 2).view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
            # print("loss", loss.shape)
            if weights is not None:
                loss = (loss * weights)
                # print("weights", weights.shape)
                # print("loss * weights", loss.shape)
                loss = loss.mean(0).sum()
            else:
                loss = loss.mean(0).sum()
        elif len(out.shape) == 3:  # STNet
            eff_batch_size = y.shape[0]
            y = y.view(eff_batch_size, 1, -1)[:, :, mask.view(-1).bool()]
            loss = ((out - y) ** 2)
            if weights is not None:
                # print(loss.shape)
                # print(weights.shape)
                loss = (loss * weights.view(1, 1, -1)).mean(0).sum()
            else:
                loss = loss.mean(0).sum()
        optimizer_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=2)
        optimizer_.step()
        epoch_loss.append(loss.item())
        if num_iters is not None and num_iters == i:
            break
    return epoch_loss


def forward_emb(graphs_, in_feat_, od_adj_, poi_cos_):
    views = mvgat(graphs_, torch.Tensor(in_feat_).to(device))
    fused_emb, embs = fusion(views)
    s_emb = embs[-2]
    d_emb = embs[-1]
    poi_emb = embs[-3]
    recons_sd = torch.matmul(s_emb, d_emb.transpose(0, 1))
    pred_d = torch.log(torch.softmax(recons_sd, dim=1) + 1e-5)
    loss_d = (torch.Tensor(od_adj_).to(device) * pred_d).mean()
    pred_s = torch.log(torch.softmax(recons_sd, dim=0) + 1e-5)
    loss_s = (torch.Tensor(od_adj_).to(device) * pred_s).mean()
    poi_sim = torch.matmul(poi_emb, poi_emb.transpose(0, 1))
    loss_poi = ((poi_sim - torch.Tensor(poi_cos_).to(device)) ** 2).mean()
    loss = -loss_s - loss_d + loss_poi

    return loss, fused_emb, embs


def forward_sgnn_emb(graphs_, in_feat_, poi_cos_):
    views = mvgat(graphs_, torch.Tensor(in_feat_).to(device))
    fused_emb, embs = fusion(views)
    poi_emb = embs[-1]
    poi_sim = torch.matmul(poi_emb, poi_emb.transpose(0, 1))
    loss_poi = ((poi_sim - torch.Tensor(poi_cos_).to(device)) ** 2).mean()
    loss = loss_poi
    return loss, fused_emb, embs


def meta_sgnn_train_epoch(s_embs, t_embs):
    model.cuda()
    meta_query_losses = []
    for meta_ep in range(args.outeriter):
        fast_weights, bn_vars = get_weights_bn_vars(model)
        source_weights = scoring(s_embs, t_embs)
        # inner loop on source, pre-train with weights
        for meta_it in range(args.sinneriter):
            s_x, s_y = batch_sampler((torch.Tensor(pems_train_data), torch.Tensor(pems_train_label)), args.batch_size)
            s_x = s_x.to(device)
            s_y = s_y.to(device)
            pred_pems = model.functional_forward(s_x, A_pems, weights=fast_weights)
            if len(s_y[s_y == 0]):
                label_mask = torch.ones_like(s_y).cuda()
                label_mask = label_mask * s_y
                label_mask[label_mask != 0] = 1
                loss_pems = (((pred_pems * label_mask - torch.squeeze(s_y)) ** 2) * source_weights)
            else:
                loss_pems = (((pred_pems - torch.squeeze(s_y)) ** 2) * source_weights)
            loss_pems = loss_pems.mean(0).sum()
            fast_loss = loss_pems
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True, allow_unused=True)
            for name, grad in zip(fast_weights.keys(), grads):
                if grad is None:
                    pass
                else:
                    fast_weights[name] = fast_weights[name] - args.innerlr * grad
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        # inner loop on target, simulate fine-tune
        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(la_train_data), torch.Tensor(la_train_label)), args.batch_size)
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            pred_la = model.functional_forward(t_x, A_la, weights=fast_weights)
            if len(t_y[t_y == 0]):
                label_mask = torch.ones_like(t_y).cuda()
                label_mask = label_mask * t_y
                label_mask[label_mask != 0] = 1
                loss_la = (((pred_la * label_mask - torch.squeeze(t_y)) ** 2) * source_weights)
            else:
                loss_la = (((pred_la - torch.squeeze(t_y)) ** 2) * source_weights)
            loss_la = loss_la.mean(0).sum()
            fast_loss = loss_la
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True, allow_unused=True)
            for name, grad in zip(fast_weights.keys(), grads):
                if grad is None:
                    pass
                else:
                    fast_weights[name] = fast_weights[name] - args.innerlr * grad
                # fast_weights[name].add_(grad, alpha = -args.innerlr)

        q_losses = []
        for k in range(3):
            # query loss
            x_q, y_q = batch_sampler((torch.Tensor(la_train_data), torch.Tensor(la_train_label)), args.batch_size)
            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_q = model.functional_forward(x_q, A_la, weights=fast_weights)
            if len(y_q[y_q == 0]):
                label_mask = torch.ones_like(y_q).cuda()
                label_mask = label_mask * y_q
                label_mask[label_mask != 0] = 1
                loss_la = (((pred_q * label_mask - torch.squeeze(t_y)) ** 2) * source_weights)
            else:
                loss_la = (((pred_q - torch.squeeze(t_y)) ** 2) * source_weights)
            loss_la = loss_la.mean(0).sum()
            q_losses.append(loss_la)
        q_loss = torch.stack(q_losses).mean()
        weights_mean = (source_weights ** 2).mean()
        meta_loss = q_loss + weights_mean * args.weight_reg
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()))
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()
        meta_query_losses.append(q_loss.item())
    return np.mean(meta_query_losses)


def calculate_cw(la_train_data, la_train_label, num_sensors, t_steps):
    # 计算每个传感器在历史时间步中的原始观察比例 R_t
    R_t = torch.zeros(num_sensors).to(device)
    for n in range(num_sensors):
        observed = torch.sum(la_train_data[n, :, :].isnan() == False, dim=0)  # 计算每个传感器在每个时间步的非缺失观测值
        R_t[n] = observed.sum().float() / t_steps  # 计算每个传感器的观测比例
    return R_t


def meta_sgnn_train_epoch_cw(s_embs, t_embs):
    model.cuda()
    meta_query_losses = []

    # 计算源城市和目标城市的 Credibility Weights
    source_weights = scoring(s_embs, t_embs)

    for meta_ep in range(args.outeriter):
        fast_weights, bn_vars = get_weights_bn_vars(model)

        # 计算目标城市的可信度权重
        R_t = calculate_cw(la_train_data, la_train_label, num_sensors=la_train_data.shape[0],
                           t_steps=la_train_data.shape[1])
        W_t = torch.sigmoid(R_t).to(device)  # 使用sigmoid函数转化为可信度权重

        # 内循环：源城市数据预训练
        for meta_it in range(args.sinneriter):
            s_x, s_y = batch_sampler((torch.Tensor(pems_train_data), torch.Tensor(pems_train_label)), args.batch_size)
            s_x = s_x.to(device)
            s_y = s_y.to(device)
            pred_pems = model.functional_forward(s_x, A_pems, weights=fast_weights)

            if len(s_y[s_y == 0]):
                label_mask = torch.ones_like(s_y).cuda()
                label_mask = label_mask * s_y
                label_mask[label_mask != 0] = 1
                loss_pems = (((pred_pems * label_mask - torch.squeeze(s_y)) ** 2) * source_weights)
            else:
                loss_pems = (((pred_pems - torch.squeeze(s_y)) ** 2) * source_weights)

            loss_pems = loss_pems.mean(0).sum()
            fast_loss = loss_pems
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True, allow_unused=True)
            for name, grad in zip(fast_weights.keys(), grads):
                if grad is None:
                    pass
                else:
                    fast_weights[name] = fast_weights[name] - args.innerlr * grad

        # 内循环：目标城市数据微调
        for meta_it in range(args.tinneriter):
            t_x, t_y = batch_sampler((torch.Tensor(la_train_data), torch.Tensor(la_train_label)), args.batch_size)
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            pred_la = model.functional_forward(t_x, A_la, weights=fast_weights)

            if len(t_y[t_y == 0]):
                label_mask = torch.ones_like(t_y).cuda()
                label_mask = label_mask * t_y
                label_mask[label_mask != 0] = 1
                loss_la = (((pred_la * label_mask - torch.squeeze(t_y)) ** 2) * source_weights)
            else:
                loss_la = (((pred_la - torch.squeeze(t_y)) ** 2) * source_weights)

            loss_la = loss_la.mean(0).sum()
            fast_loss = loss_la
            grads = torch.autograd.grad(fast_loss, fast_weights.values(), create_graph=True, allow_unused=True)
            for name, grad in zip(fast_weights.keys(), grads):
                if grad is None:
                    pass
                else:
                    fast_weights[name] = fast_weights[name] - args.innerlr * grad

        # 查询损失计算：将计算出的W_t与查询损失结合
        q_losses = []
        for k in range(3):
            # 查询损失
            x_q, y_q = batch_sampler((torch.Tensor(la_train_data), torch.Tensor(la_train_label)), args.batch_size)
            x_q = x_q.to(device)
            y_q = y_q.to(device)
            pred_q = model.functional_forward(x_q, A_la, weights=fast_weights)

            if len(y_q[y_q == 0]):
                label_mask = torch.ones_like(y_q).cuda()
                label_mask = label_mask * y_q
                label_mask[label_mask != 0] = 1
                loss_q = (((pred_q * label_mask - torch.squeeze(y_q)) ** 2) * source_weights)
            else:
                loss_q = (((pred_q - torch.squeeze(y_q)) ** 2) * source_weights)

            # 使用目标城市的可信度权重 W_t 加权损失
            weighted_loss = loss_q * W_t
            q_losses.append(weighted_loss.mean())

        # 计算加权损失
        q_loss = torch.stack(q_losses).mean()
        weights_mean = (source_weights ** 2).mean()
        meta_loss = q_loss + weights_mean * args.weight_reg

        # 更新参数
        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=list(scoring.parameters()))
        torch.nn.utils.clip_grad_norm_(scoring.parameters(), max_norm=2)
        meta_optimizer.step()

        meta_query_losses.append(q_loss.item())

    return np.mean(meta_query_losses)


def TrainMETAModel(model, train_dataloader, valid_dataloader, A, learning_rate=1e-5, optm='Adam', num_epochs=500,
                   patience=10, min_delta=0.00001):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()
    model.cuda()
    learning_rate = learning_rate
    if optm == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optm == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optm == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    use_gpu = torch.cuda.is_available()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    train_time_list = []
    valid_time_list = []
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    losses_epochs_valid_steps_l1 = []
    sub_epoch = 1
    for epoch in range(0, num_epochs):
        model.train()
        mvgat.train()
        fusion.train()
        scoring.train()
        # train embeddings
        emb_losses = []
        mmd_losses = []
        edge_losses = []
        for emb_ep in range(5):
            loss_emb_, loss_mmd_, loss_et_ = train_sgnn_emb_epoch()
            emb_losses.append(loss_emb_)
            mmd_losses.append(loss_mmd_)
            edge_losses.append(loss_et_)
        # evaluate embeddings
        with torch.no_grad():
            views = mvgat(pems_graphs, torch.Tensor(pems_norm_feat).to(device))
            fused_emb_s, _ = fusion(views)
            views = mvgat(la_graphs, torch.Tensor(la_norm_feat).to(device))
            fused_emb_t, _ = fusion(views)
        # meta train scorings
        avg_q_loss = meta_sgnn_train_epoch(fused_emb_s, fused_emb_t)
        with torch.no_grad():
            source_weights = scoring(fused_emb_s, fused_emb_t)
        # implement a moving average
        if epoch == 0:
            source_weights_ma = torch.ones_like(source_weights, device=device, requires_grad=False)
        source_weights_ma = ma_param * source_weights_ma + (1 - ma_param) * source_weights

        losses_epoch_train = []
        losses_epoch_valid = []
        train_start = time.time()
        for data in train_dataloader:
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            model.zero_grad()
            outputs = model(inputs, A)

            if len(labels[labels == 0]):
                label_mask = torch.ones_like(labels).cuda()
                label_mask = label_mask * labels
                label_mask[label_mask != 0] = 1
                loss_train = (((outputs * label_mask - torch.squeeze(labels)) ** 2) * source_weights_ma)  #
            else:
                loss_train = (((outputs - torch.squeeze(labels)) ** 2) * source_weights_ma)  #
            loss_train = loss_train.mean(0).sum()
            losses_epoch_train.append(loss_train.item())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        train_end = time.time()
        # validation
        valid_start = time.time()
        for data in valid_dataloader:
            inputs_val, labels_val = data
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
            outputs_val = model(inputs_val, A)
            if len(labels_val[labels_val == 0]):
                labels_val_mask = torch.ones_like(labels_val).cuda()
                labels_val_mask = labels_val_mask * labels_val
                labels_val_mask[labels_val_mask != 0] = 1
                loss_valid = (((outputs_val * labels_val_mask - torch.squeeze(labels_val)) ** 2))
            else:
                loss_valid = (((outputs_val - torch.squeeze(labels_val)) ** 2))
            loss_valid = loss_valid.mean(0).sum()
            losses_epoch_valid.append(loss_valid.item())
        valid_end = time.time()
        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = copy.deepcopy(model)
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        #             sub_epoch += 1
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        if (epoch >= 5 and (patient_epoch == 4 or sub_epoch % 50 == 0)) and learning_rate > 1e-5:
            learning_rate = learning_rate / 10
            if optm == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optm == 'Adadelta':
                optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
            elif optm == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            sub_epoch = 1
        else:
            sub_epoch += 1
        # Print training parameters
        cur_time = time.time()
        train_time = np.around([train_end - train_start], decimals=2)
        train_time_list.append(train_time)
        valid_time = np.around([valid_end - valid_start], decimals=2)
        valid_time_list.append(valid_time)
        print("avg_q_loss:{}".format(avg_q_loss))
        print(
            'Epoch: {}, train_loss: {}, valid_loss: {}, lr: {}, train_time: {}, valid_time: {}, best model: {}'.format( \
                epoch, \
                np.around(avg_losses_epoch_train, decimals=8), \
                np.around(avg_losses_epoch_valid, decimals=8), \
                learning_rate, \
                np.around([train_end - train_start], decimals=2), \
                np.around([valid_end - valid_start], decimals=2), \
                is_best_model))
    train_time_avg = np.mean(np.array(train_time_list))
    valid_time_avg = np.mean(np.array(valid_time_list))
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid,
                        losses_epochs_valid_steps_l1, train_time_avg, valid_time_avg]


def train_sgnn_emb_epoch():
    loss_source, fused_emb_s, embs_s = forward_sgnn_emb(pems_graphs, pems_norm_feat, pems_feat_cos)
    loss_target, fused_emb_t, embs_t = forward_sgnn_emb(la_graphs, la_norm_feat, la_feat_cos)
    loss_emb = loss_source + loss_target
    # 计算域适应损失
    source_ids = np.random.randint(0, len(fused_emb_s), size=(128,))
    target_ids = np.random.randint(0, len(fused_emb_t), size=(128,))
    mmd_loss = mmd(fused_emb_s[source_ids, :], fused_emb_t[target_ids, :])

    source_batch_edges = np.random.randint(0, len(pems_edges), size=(256,))
    target_batch_edges = np.random.randint(0, len(la_edges), size=(256,))
    source_batch_src = torch.Tensor(pems_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(pems_edges[source_batch_edges, 1]).long()
    source_emb_src = fused_emb_s[source_batch_src, :]
    source_emb_dst = fused_emb_s[source_batch_dst, :]
    target_batch_src = torch.Tensor(la_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(la_edges[target_batch_edges, 1]).long()
    target_emb_src = fused_emb_t[target_batch_src, :]
    target_emb_dst = fused_emb_t[target_batch_dst, :]

    pred_source = edge_disc.forward(source_emb_src, source_emb_dst)
    pred_target = edge_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(pems_edge_labels[source_batch_edges]).to(device)
    target_batch_labels = torch.Tensor(la_edge_labels[target_batch_edges]).to(device)
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (
                1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (
                1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target

    emb_optimizer.zero_grad()
    loss = loss_emb + mmd_w * mmd_loss + et_w * loss_et
    loss.backward()
    emb_optimizer.step()
    return loss_emb.item(), mmd_loss.item(), loss_et.item()


from SGNN_utils import TestModel
from SGNN_utils import TrainModel



emb_losses = []
mmd_losses = []
edge_losses = []
pretrain_emb_epoch = 80
for emb_ep in range(pretrain_emb_epoch):
    loss_emb_, loss_mmd_, loss_et_ = train_sgnn_emb_epoch()
    emb_losses.append(loss_emb_)
    mmd_losses.append(loss_mmd_)
    edge_losses.append(loss_et_)
with torch.no_grad():
    views = mvgat(pems_graphs, torch.Tensor(pems_norm_feat).to(device))
    fused_emb_s, _ = fusion(views)
    views = mvgat(la_graphs, torch.Tensor(la_norm_feat).to(device))
    fused_emb_t, _ = fusion(views)


#stage1
model, train_result = TrainMETAModel(model, train_dataloader_pems, valid_dataloader_pems, A_pems, optm=optm,
                                     learning_rate=learning_rate, patience=10)

#stage2
model, train_result = TrainModel(model, train_dataloader_la, valid_dataloader_la, A_la, optm=optm,
                                 learning_rate=learning_rate, patience=10)
#stage3
model, train_result = TrainModel(model, train_dataloader_la, valid_dataloader_la, A_la, optm=optm,
                                 learning_rate=learning_rate, patience=10)

test_result = TestModel(model, test_dataloader_la, A_la, max_speed_la)


