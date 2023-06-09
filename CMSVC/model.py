import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import os
import logging
import datetime
import copy
from dgl import function as fn
import dgl
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from collections import OrderedDict
# TODO: replace the VF.lstm in this file 
# try implement LSTM and see which is better
from torch._VF import lstm


def functional_lstm(input_, w_ih, w_hh, b_ih, b_hh, hidden_size_):
    # This function implements a functional LSTM forwarder. 
    time_steps = input_.shape[0]
    bsize = input_.shape[1]
    # initialize hidden 
    h = torch.zeros(1, bsize, hidden_size_, device=w_ih.device)
    c = torch.zeros(1, bsize, hidden_size_, device=w_ih.device)
    outputs = []
    states = []
    for i in range(time_steps):
        from_i = F.linear(input_[i].squeeze(), w_ih, b_ih)
        from_h = F.linear(h.squeeze(), w_hh, b_hh)
        chunks = from_i + from_h
        it, ft, gt, ot = chunks.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        c = ft * c + it * gt
        h = ot * torch.tanh(c)
        outputs.append(h)
    return torch.cat(outputs, 0), (h, c)


def functional_linear(weight, bias, inputs):
    res = torch.mm(inputs, weight.t()) + bias
    return res


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x

    def functional_forward(self, x, weights=None, bn_vars=None, bn_training=True):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
            bn_vars = OrderedDict()
            for k in self.state_dict():
                if 'running_mean' in k or 'running_var' in k:
                    bn_vars[k] = self.state_dict()[k]
        z = F.batch_norm(x, bn_vars['bn1.running_mean'], bn_vars['bn1.running_var'], weights['bn1.weight'],
                         weights['bn1.bias'], training=bn_training)
        z = F.relu(z)
        z = F.conv2d(z, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
        z = F.batch_norm(z, bn_vars['bn2.running_mean'], bn_vars['bn2.running_var'], weights['bn2.weight'],
                         weights['bn2.bias'], training=bn_training)
        z = F.relu(z)
        z = F.conv2d(z, weights['conv2.weight'], weights['conv2.bias'], stride=1, padding=1)
        return z + x


class ResUnit_nobn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit_nobn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = F.relu(x)
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x

    def functional_forward(self, x, weights=None, bn_vars=None, bn_training=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        z = F.relu(x)
        z = F.conv2d(z, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
        z = F.relu(z)
        z = F.conv2d(z, weights['conv2.weight'], weights['conv2.bias'], stride=1, padding=1)
        return z + x


class BatchGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(BatchGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, verbose=False):
        graph = graph.local_var()
        num_node = feat.shape[0]
        num_batch = feat.shape[1]

        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(num_node, num_batch, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(num_node, num_batch, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                num_node, num_batch, self._num_heads, self._out_feats)

        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        if verbose:
            print(graph.edata['a'])
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(num_node, num_batch, -1, self._out_feats)
            rst = rst + resval
        if self.activation:
            rst = self.activation(rst)
        return rst


class STResNet_nobn(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, sigmoid_act=False):
        super(STResNet_nobn, self).__init__()
        self.sigmoid_act = sigmoid_act
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.layers = []
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.layers.append(ResUnit_nobn(in_channels=64, out_channels=64))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X, spatial_mask=None):
        X = self.conv1(X)
        for layer in self.layers:
            X = layer(X)
        X = self.conv2(X)
        if self.sigmoid_act:
            return torch.sigmoid(X)
        else:
            return X

    def functional_forward(self, X, spatial_mask=None, weights=None, bn_vars=None, bn_training=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        X = F.conv2d(X, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
        for i in range(self.num_blocks):
            sub_weights = OrderedDict()
            for k in weights.keys():
                if k.startswith("layers.%d" % i):
                    stripped_key = k.lstrip("layers.%d." % i)
                    sub_weights[stripped_key] = weights[k]
            X = self.layers[i].functional_forward(X, sub_weights)
        X = F.conv2d(X, weights['conv2.weight'], weights['conv2.bias'], stride=1, padding=1)
        if self.sigmoid_act:
            return torch.sigmoid(X)
        else:
            return X


class STNet_nobn(nn.Module):
    def __init__(self, num_channels, num_convs, spatial_mask, sigmoid_out=False):
        super(STNet_nobn, self).__init__()
        self.num_channels = num_channels
        self.spatial_mask = spatial_mask.bool()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        self.bns = []
        self.num_convs = num_convs
        for i in range(num_convs):
            self.layers.append(ResUnit_nobn(64, 64))
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid_out = sigmoid_out

    def forward(self, X, spatial_mask=None, return_feat=False):
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        num_lag = (X.shape[1] // self.num_channels)
        batch_size = X.shape[0]
        outs = []
        for i in range(num_lag):
            input = X[:, i * self.num_channels:(i + 1) * self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              64).contiguous()  # [:, spatial_mask.view(-1), :].contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim=0)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :]
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim=-1)
        temporal_valid = temporal[:, spatial_mask.view(-1), :]
        hid = F.relu(self.linear1(temporal_valid))
        output = self.linear2(hid).permute(0, 2, 1)
        if self.sigmoid_out:
            output = torch.sigmoid(output)
        if return_feat:
            return temporal, output
        else:
            return output

    def functional_forward(self, X, spatial_mask=None, weights=None, bn_vars=None, bn_training=None, return_feat=False):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        num_lag = (X.shape[1] // self.num_channels)
        batch_size = X.shape[0]
        outs = []
        for i in range(num_lag):
            input = X[:, i * self.num_channels:(i + 1) * self.num_channels, :, :]
            z = F.conv2d(input, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
            for j in range(self.num_convs):
                sub_weights = OrderedDict()
                for k in weights.keys():
                    if k.startswith("layers.%d" % j):
                        stripped_key = k.lstrip("layers.%d." % j)
                        sub_weights[stripped_key] = weights[k]
                z = self.layers[j].functional_forward(z, sub_weights)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim=0)
        flat_weights = [weights['lstm.weight_ih_l0'], weights['lstm.weight_hh_l0'], weights['lstm.bias_ih_l0'],
                        weights['lstm.bias_hh_l0']]
        result = functional_lstm(z, flat_weights[0], flat_weights[1], flat_weights[2], flat_weights[3], 128)
        temporal_out = result[0]
        temporal_hid, _ = result[1]
        temporal_out = temporal_out[-1:, :]
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim=-1)
        temporal_valid = temporal[:, spatial_mask.view(-1), :]
        hid = F.relu(F.linear(temporal_valid, weights['linear1.weight'], weights['linear1.bias']))
        output = F.linear(hid, weights['linear2.weight'], weights['linear2.bias']).permute(0, 2, 1)
        if self.sigmoid_out:
            output = torch.sigmoid(output)
        if return_feat:
            return temporal, output
        else:
            return output


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


import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    def grad(x, constant):
        return Grad.apply(x, constant)


class Domain_classifier_DG(nn.Module):

    def __init__(self, num_class, encode_dim):
        super(Domain_classifier_DG, self).__init__()

        self.num_class = num_class
        self.encode_dim = encode_dim

        self.fc1 = nn.Linear(self.encode_dim, 16)
        self.fc2 = nn.Linear(16, num_class)

    def forward(self, input, constant, Reverse):
        if Reverse:
            input = GradReverse.grad_reverse(input, constant)
        else:
            input = Grad.grad(input, constant)
        logits = torch.tanh(self.fc1(input))
        logits = self.fc2(logits)
        logits = F.log_softmax(logits, 1)

        return logits


class VGRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(VGRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape[0], inputs.shape[1]
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.reshape((-1, self._num_gru_units + 1))
        outputs = concatenation @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def functional_forward(self, inputs, hidden_state, weight, bias):
        batch_size, num_nodes = inputs.shape[0], inputs.shape[1]
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.reshape((-1, self._num_gru_units + 1))
        outputs = concatenation @ weight + bias
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class VGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, adj_encodedim):
        super(VGRUCell, self).__init__()
        self._encode_dim = adj_encodedim
        self._hidden_dim = hidden_dim
        self.weights = nn.Parameter(
            torch.FloatTensor(self._encode_dim, self._encode_dim)
        )
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.linear = nn.Linear(self._encode_dim + self._hidden_dim, self._hidden_dim)
        self.linear1 = VGRULinear(self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.linear2 = VGRULinear(self._hidden_dim, self._hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs, hidden_state, feat):
        batch_size, num_nodes = inputs.shape[0], inputs.shape[1]
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.linear2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1 - u) * c

        new_hidden_state = new_hidden_state.reshape((batch_size * num_nodes, self._hidden_dim))
        feat = feat.reshape((batch_size * num_nodes, feat.shape[-1]))
        feat = feat @ self.weights + self.bias
        new_hidden_state = torch.cat((new_hidden_state, feat), 1)
        new_hidden_state = self.linear(new_hidden_state)
        new_hidden_state = new_hidden_state.reshape((batch_size, num_nodes * self._hidden_dim))

        return new_hidden_state, new_hidden_state

    def functional_forward(self, inputs, hidden_state, feat, w1, b1, w2, b2, wg, bg, wl, bl):
        batch_size, num_nodes = inputs.shape[0], inputs.shape[1]
        concatenation = torch.sigmoid(self.linear1.functional_forward(inputs, hidden_state, w1, b1))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.linear2.functional_forward(inputs, r * hidden_state, w2, b2))
        new_hidden_state = u * hidden_state + (1 - u) * c

        new_hidden_state = new_hidden_state.reshape((batch_size * num_nodes, self._hidden_dim))
        feat = feat.reshape((batch_size * num_nodes, feat.shape[-1]))
        feat = feat @ wg + bg
        new_hidden_state = torch.cat((new_hidden_state, feat), 1)
        new_hidden_state = functional_linear(wl, bl, new_hidden_state)
        new_hidden_state = new_hidden_state.reshape((batch_size, num_nodes * self._hidden_dim))

        return new_hidden_state, new_hidden_state


class VGRU_FEAT(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, encode_dim: int):
        super(VGRU_FEAT, self).__init__()
        self._encode_dim = encode_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self.vgru_cell = VGRUCell(self._hidden_dim, self._encode_dim)

    def forward(self, inputs, feat):
        batch_size, seq_len, num_nodes = inputs.shape
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        for i in range(seq_len):
            output, hidden_state = self.vgru_cell(inputs[:, i, :], hidden_state, feat)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        last_output = last_output.reshape((-1, last_output.size(2)))

        return last_output

    def functional_forward(self, inputs, feat, w1, b1, w2, b2, wg, bg, wl, bl):
        batch_size, seq_len, num_nodes = inputs.shape
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        for i in range(seq_len):
            output, hidden_state = self.vgru_cell.functional_forward(inputs[:, i, :], hidden_state, feat,
                                                                     w1, b1, w2, b2, wg, bg, wl, bl)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        last_output = last_output.reshape((-1, last_output.size(2)))

        return last_output


class Extractor_N2V(nn.Module):

    def __init__(self, input_dim, hidden_dim: int, encode_dim, device, batch_size, etype):
        super(Extractor_N2V, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.etype = etype
        self._input_dim = input_dim
        self._encode_dim = encode_dim
        self._hidden_dim = hidden_dim
        self.adj_encoderlayer1 = nn.Linear(input_dim, hidden_dim)
        self.adj_encoderlayer2 = nn.Linear(hidden_dim, encode_dim)
        self.batch_norm = nn.BatchNorm1d(encode_dim)
        self.eps1 = nn.Parameter(torch.tensor([1.0]))

    def forward(self, h, adj):
        h = self.adj_encoderlayer1(h.float())
        if self.etype == "gin":
            pooled = torch.mm(adj.float(), h.float())
            degree = torch.spmm(adj.float(), torch.ones((adj.shape[0], 1)).float().to(self.device)).to(
                self.device)
            degree = torch.where(degree < torch.tensor(1e-6, dtype=degree.dtype, device=degree.device),
                                 torch.tensor(1.0, dtype=degree.dtype, device=degree.device), degree)
            pooled = pooled / degree
            h = pooled + self.eps1 * h

        h = self.batch_norm(h.float())
        h = self.adj_encoderlayer2(h.float())

        return h

    def functional_forward(self, h, adj, ew1, eb1, ew2, eb2, bnw, bnb, rm, rv, bn_training):
        h = functional_linear(ew1, eb1, h.float())
        if self.etype == "gin":
            pooled = torch.mm(adj.float(), h.float())
            degree = torch.spmm(adj.float(), torch.ones((adj.shape[0], 1)).float().to(self.device)).to(
                self.device)
            degree = torch.where(degree < torch.tensor(1e-6, dtype=degree.dtype, device=degree.device),
                                 torch.tensor(1.0, dtype=degree.dtype, device=degree.device), degree)
            pooled = pooled / degree
            h = pooled + self.eps1 * h
        h = torch.nn.functional.batch_norm(h.float(), running_mean=rm, running_var=rv, bias=bnb, weight=bnw,
                                           training=bn_training)
        h = functional_linear(ew2, eb2, h.float())

        return h


class DASTNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, device, batch_size, etype, pre_len, dataset, ft_dataset,
                 adj_pems04, adj_pems07, adj_pems08):
        super(DASTNet, self).__init__()
        self.dataset = dataset
        self.finetune_dataset = ft_dataset
        self.pems04_adj = adj_pems04
        self.pems07_adj = adj_pems07
        self.pems08_adj = adj_pems08
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.device = device

        self.pems04_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.pems07_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.pems08_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size, etype).to(
            device)
        self.shared_pems04_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)
        self.shared_pems07_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)
        self.shared_pems08_featExtractor = Extractor_N2V(input_dim, hidden_dim, encode_dim, device, batch_size,
                                                         etype).to(device)

        self.speed_predictor = VGRU_FEAT(hidden_dim=hidden_dim, output_dim=pre_len, encode_dim=encode_dim).to(device)
        self.pems04_linear = nn.Linear(hidden_dim, pre_len, )
        self.pems07_linear = nn.Linear(hidden_dim, pre_len, )
        self.pems08_linear = nn.Linear(hidden_dim, pre_len, )

        self.weight_feat_private = nn.Parameter(torch.tensor([1.0]).to(self.device))
        self.weight_feat_shared = nn.Parameter(torch.tensor([0.0]).to(self.device))
        self.private_pems04_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.private_pems07_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.private_pems08_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.shared_pems04_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.shared_pems07_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.shared_pems08_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems04_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems07_linear = nn.Linear(hidden_dim, hidden_dim, )
        self.combine_pems08_linear = nn.Linear(hidden_dim, hidden_dim, )

    def forward(self, vec_pems04, vec_pems07, vec_pems08, feat, eval, need_road=True):
        if self.dataset != self.finetune_dataset:
            if not eval:
                shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
            else:
                if self.dataset == '4' or self.dataset == 'ny':
                    shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                elif self.dataset == '7' or self.dataset == 'chi':
                    shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                elif self.dataset == '8' or self.dataset == 'dc':
                    shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
            if self.dataset == '4' or self.dataset == 'ny':
                h_pems04 = shared_pems04_feat.expand(self.batch_size, self.pems04_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems04)
                pred = self.pems04_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems04_adj.shape[0], -1))
            elif self.dataset == '7' or self.dataset == 'chi':
                h_pems07 = shared_pems07_feat.expand(self.batch_size, self.pems07_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems07)
                pred = self.pems07_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems07_adj.shape[0], -1))
            elif self.dataset == '8' or self.dataset == 'dc':
                h_pems08 = shared_pems08_feat.expand(self.batch_size, self.pems08_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems08)
                pred = self.pems08_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems08_adj.shape[0], -1))

            if not eval:
                return pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat
            else:
                return pred
        else:
            if self.dataset == '4' or self.dataset == 'ny':
                shared_pems04_feat = self.shared_pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                pems04_feat = self.pems04_featExtractor(vec_pems04, self.pems04_adj).to(self.device)
                pems04_feat = self.combine_pems04_linear(
                    self.private_pems04_linear(pems04_feat) + self.shared_pems04_linear(shared_pems04_feat))
                h_pems04 = pems04_feat.expand(self.batch_size, self.pems04_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems04)
                pred = self.pems04_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems04_adj.shape[0], -1))
            elif self.dataset == '7' or self.dataset == 'chi':
                shared_pems07_feat = self.shared_pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                pems07_feat = self.pems07_featExtractor(vec_pems07, self.pems07_adj).to(self.device)
                pems07_feat = self.combine_pems07_linear(
                    self.private_pems07_linear(pems07_feat) + self.shared_pems07_linear(shared_pems07_feat))
                h_pems07 = pems07_feat.expand(self.batch_size, self.pems07_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems07)
                pred = self.pems07_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems07_adj.shape[0], -1))
            elif self.dataset == '8' or self.dataset == 'dc':
                shared_pems08_feat = self.shared_pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
                pems08_feat = self.pems08_featExtractor(vec_pems08, self.pems08_adj).to(self.device)
                pems08_feat = self.combine_pems08_linear(
                    self.private_pems08_linear(pems08_feat) + self.shared_pems08_linear(shared_pems08_feat))
                h_pems08 = pems08_feat.expand(self.batch_size, self.pems08_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor(feat, h_pems08)
                pred = self.pems08_linear(pred)
                pred = pred.reshape((self.batch_size, self.pems08_adj.shape[0], -1))

            return pred

    def functional_forward(self, vec_pems04, vec_pems07, vec_pems08, feat, eval, params, bn_vars, bn_training, data_set="4"):
        if self.dataset != self.finetune_dataset:
            if not eval:
                shared_pems04_feat = self.shared_pems04_featExtractor.functional_forward(vec_pems04, self.pems04_adj,
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer1.weight"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer1.bias"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer2.weight"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer2.bias"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.batch_norm.weight"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.batch_norm.bias"],
                                                                                         bn_vars[
                                                                                             "pems04_featExtractor.batch_norm.running_mean"],
                                                                                         bn_vars[
                                                                                             "pems04_featExtractor.batch_norm.running_var"]

                                                                                         , bn_training).to(self.device)
                shared_pems07_feat = self.shared_pems07_featExtractor.functional_forward(vec_pems07, self.pems07_adj,
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer1.weight"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer1.bias"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer2.weight"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer2.bias"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.batch_norm.weight"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.batch_norm.bias"],
                                                                                         bn_vars[
                                                                                             "pems07_featExtractor.batch_norm.running_mean"],
                                                                                         bn_vars[
                                                                                             "pems07_featExtractor.batch_norm.running_var"]
                                                                                         , bn_training).to(self.device)
                shared_pems08_feat = self.shared_pems08_featExtractor.functional_forward(vec_pems08, self.pems08_adj,
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer1.weight"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer1.bias"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer2.weight"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer2.bias"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.batch_norm.weight"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.batch_norm.bias"],
                                                                                         bn_vars[
                                                                                             "pems08_featExtractor.batch_norm.running_mean"],
                                                                                         bn_vars[
                                                                                             "pems08_featExtractor.batch_norm.running_var"]
                                                                                         , bn_training).to(self.device)
            else:
                if data_set == '4' or data_set == 'ny':
                    shared_pems04_feat = self.shared_pems04_featExtractor.functional_forward(vec_pems04,
                                                                                             self.pems04_adj,
                                                                                             params[
                                                                                                 "shared_pems04_featExtractor.adj_encoderlayer1.weight"],
                                                                                             params[
                                                                                                 "shared_pems04_featExtractor.adj_encoderlayer1.bias"],
                                                                                             params[
                                                                                                 "shared_pems04_featExtractor.adj_encoderlayer2.weight"],
                                                                                             params[
                                                                                                 "shared_pems04_featExtractor.adj_encoderlayer2.bias"],
                                                                                             params[
                                                                                                 "shared_pems04_featExtractor.batch_norm.weight"],
                                                                                             params[
                                                                                                 "shared_pems04_featExtractor.batch_norm.bias"],
                                                                                             bn_vars[
                                                                                                 "shared_pems04_featExtractor.batch_norm.running_mean"],
                                                                                             bn_vars[
                                                                                                 "shared_pems04_featExtractor.batch_norm.running_var"]
                                                                                             , bn_training).to(self.device)
                elif data_set == '7' or data_set == 'chi':
                    shared_pems07_feat = self.shared_pems07_featExtractor.functional_forward(vec_pems07,
                                                                                             self.pems07_adj,
                                                                                             params[
                                                                                                 "shared_pems07_featExtractor.adj_encoderlayer1.weight"],
                                                                                             params[
                                                                                                 "shared_pems07_featExtractor.adj_encoderlayer1.bias"],
                                                                                             params[
                                                                                                 "shared_pems07_featExtractor.adj_encoderlayer2.weight"],
                                                                                             params[
                                                                                                 "shared_pems07_featExtractor.adj_encoderlayer2.bias"],
                                                                                             params[
                                                                                                 "shared_pems07_featExtractor.batch_norm.weight"],
                                                                                             params[
                                                                                                 "shared_pems07_featExtractor.batch_norm.bias"],
                                                                                             bn_vars[
                                                                                                 "shared_pems07_featExtractor.batch_norm.running_mean"],
                                                                                             bn_vars[
                                                                                                 "shared_pems07_featExtractor.batch_norm.running_var"]
                                                                                             , bn_training).to(self.device)
                elif data_set == '8' or data_set == 'dc':
                    shared_pems08_feat = self.shared_pems08_featExtractor.functional_forward(vec_pems08,
                                                                                             self.pems08_adj,
                                                                                             params[
                                                                                                 "shared_pems08_featExtractor.adj_encoderlayer1.weight"],
                                                                                             params[
                                                                                                 "shared_pems08_featExtractor.adj_encoderlayer1.bias"],
                                                                                             params[
                                                                                                 "shared_pems08_featExtractor.adj_encoderlayer2.weight"],
                                                                                             params[
                                                                                                 "shared_pems08_featExtractor.adj_encoderlayer2.bias"],
                                                                                             params[
                                                                                                 "shared_pems08_featExtractor.batch_norm.weight"],
                                                                                             params[
                                                                                                 "shared_pems08_featExtractor.batch_norm.bias"],
                                                                                             bn_vars[
                                                                                                 "shared_pems08_featExtractor.batch_norm.running_mean"],
                                                                                             bn_vars[
                                                                                                 "shared_pems08_featExtractor.batch_norm.running_var"]
                                                                                             , bn_training).to(self.device)
            if data_set == '4' or data_set == 'ny':
                h_pems04 = shared_pems04_feat.expand(self.batch_size, self.pems04_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor.functional_forward(feat, h_pems04,
                                                               params["speed_predictor.vgru_cell.linear1.weights"],
                                                               params["speed_predictor.vgru_cell.linear1.biases"],
                                                               params["speed_predictor.vgru_cell.linear2.weights"],
                                                               params["speed_predictor.vgru_cell.linear2.biases"],
                                                               params["speed_predictor.vgru_cell.weights"],
                                                               params["speed_predictor.vgru_cell.bias"],
                                                               params["speed_predictor.vgru_cell.linear.weight"],
                                                               params["speed_predictor.vgru_cell.linear.bias"])
                pred = functional_linear(params["pems04_linear.weight"],
                                         params["pems04_linear.bias"], pred)
                pred = pred.reshape((self.batch_size, self.pems04_adj.shape[0], -1))
            elif data_set == '7' or data_set == 'chi':
                h_pems07 = shared_pems07_feat.expand(self.batch_size, self.pems07_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor.functional_forward(feat, h_pems07,
                                                               params["speed_predictor.vgru_cell.linear1.weights"],
                                                               params["speed_predictor.vgru_cell.linear1.biases"],
                                                               params["speed_predictor.vgru_cell.linear2.weights"],
                                                               params["speed_predictor.vgru_cell.linear2.biases"],
                                                               params["speed_predictor.vgru_cell.weights"],
                                                               params["speed_predictor.vgru_cell.bias"],
                                                               params["speed_predictor.vgru_cell.linear.weight"],
                                                               params["speed_predictor.vgru_cell.linear.bias"]
                                                               )
                pred = functional_linear(params["pems07_linear.weight"],
                                         params["pems07_linear.bias"], pred)
                pred = pred.reshape((self.batch_size, self.pems07_adj.shape[0], -1))
            elif data_set == '8' or data_set == 'dc':
                h_pems08 = shared_pems08_feat.expand(self.batch_size, self.pems08_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor.functional_forward(feat, h_pems08,
                                                               params["speed_predictor.vgru_cell.linear1.weights"],
                                                               params["speed_predictor.vgru_cell.linear1.biases"],
                                                               params["speed_predictor.vgru_cell.linear2.weights"],
                                                               params["speed_predictor.vgru_cell.linear2.biases"],
                                                               params["speed_predictor.vgru_cell.weights"],
                                                               params["speed_predictor.vgru_cell.bias"],
                                                               params["speed_predictor.vgru_cell.linear.weight"],
                                                               params["speed_predictor.vgru_cell.linear.bias"]
                                                               )
                pred = functional_linear(params["pems08_linear.weight"],
                                         params["pems08_linear.bias"], pred)
                pred = pred.reshape((self.batch_size, self.pems08_adj.shape[0], -1))

            if not eval:
                return pred, shared_pems04_feat, shared_pems07_feat, shared_pems08_feat
            else:
                return pred
        else:
            if data_set == '4' or data_set == 'ny':
                shared_pems04_feat = self.shared_pems04_featExtractor.functional_forward(vec_pems04, self.pems04_adj,
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer1.weight"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer1.bias"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer2.weight"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.adj_encoderlayer2.bias"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.batch_norm.weight"],
                                                                                         params[
                                                                                             "shared_pems04_featExtractor.batch_norm.bias"],
                                                                                         bn_vars[
                                                                                             "shared_pems04_featExtractor.batch_norm.running_mean"],
                                                                                         bn_vars[
                                                                                             "shared_pems04_featExtractor.batch_norm.running_var"]
                                                                                         , bn_training).to(self.device)
                pems04_feat = self.pems04_featExtractor.functional_forward(vec_pems04, self.pems04_adj,
                                                                           params[
                                                                               "pems04_featExtractor.adj_encoderlayer1.weight"],
                                                                           params[
                                                                               "pems04_featExtractor.adj_encoderlayer1.bias"],
                                                                           params[
                                                                               "pems04_featExtractor.adj_encoderlayer2.weight"],
                                                                           params[
                                                                               "pems04_featExtractor.adj_encoderlayer2.bias"],
                                                                           params[
                                                                               "pems04_featExtractor.batch_norm.weight"],
                                                                           params[
                                                                               "pems04_featExtractor.batch_norm.bias"],
                                                                           bn_vars[
                                                                               "pems04_featExtractor.batch_norm.running_mean"],
                                                                           bn_vars[
                                                                               "pems04_featExtractor.batch_norm.running_var"], bn_training).to(
                    self.device)
                pems04_feat = functional_linear(params["combine_pems04_linear.weight"],
                                                params["combine_pems04_linear.bias"],
                                                # self.private_pems04_linear(pems04_feat) +
                                                functional_linear(params["private_pems04_linear.weight"],
                                                                  params["private_pems04_linear.bias"], pems04_feat)
                                                +
                                                functional_linear(params["shared_pems04_linear.weight"],
                                                                  params["shared_pems04_linear.bias"],
                                                                  shared_pems04_feat)
                                                # self.shared_pems04_linear(shared_pems04_feat)
                                                )
                h_pems04 = pems04_feat.expand(self.batch_size, self.pems04_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor.functional_forward(feat, h_pems04,
                                                               params["speed_predictor.vgru_cell.linear1.weights"],
                                                               params["speed_predictor.vgru_cell.linear1.biases"],
                                                               params["speed_predictor.vgru_cell.linear2.weights"],
                                                               params["speed_predictor.vgru_cell.linear2.biases"],
                                                               params["speed_predictor.vgru_cell.weights"],
                                                               params["speed_predictor.vgru_cell.bias"],
                                                               params["speed_predictor.vgru_cell.linear.weight"],
                                                               params["speed_predictor.vgru_cell.linear.bias"]
                                                               )
                # pred = self.pems04_linear(pred)
                pred = functional_linear(params["pems04_linear.weight"], params["pems04_linear.bias"], pred)
                pred = pred.reshape((self.batch_size, self.pems04_adj.shape[0], -1))
            elif data_set == '7' or data_set == 'chi':
                shared_pems07_feat = self.shared_pems07_featExtractor.functional_forward(vec_pems07, self.pems07_adj,
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer1.weight"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer1.bias"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer2.weight"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.adj_encoderlayer2.bias"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.batch_norm.weight"],
                                                                                         params[
                                                                                             "shared_pems07_featExtractor.batch_norm.bias"],
                                                                                         bn_vars[
                                                                                             "shared_pems07_featExtractor.batch_norm.running_mean"],
                                                                                         bn_vars[
                                                                                             "shared_pems07_featExtractor.batch_norm.running_var"]
                                                                                         , bn_training).to(self.device)
                pems07_feat = self.pems07_featExtractor.functional_forward(vec_pems07, self.pems07_adj,
                                                                           params[
                                                                               "pems07_featExtractor.adj_encoderlayer1.weight"],
                                                                           params[
                                                                               "pems07_featExtractor.adj_encoderlayer1.bias"],
                                                                           params[
                                                                               "pems07_featExtractor.adj_encoderlayer2.weight"],
                                                                           params[
                                                                               "pems07_featExtractor.adj_encoderlayer2.bias"],
                                                                           params[
                                                                               "pems07_featExtractor.batch_norm.weight"],
                                                                           params[
                                                                               "pems07_featExtractor.batch_norm.bias"],
                                                                           bn_vars[
                                                                               "pems07_featExtractor.batch_norm.running_mean"],
                                                                           bn_vars[
                                                                               "pems07_featExtractor.batch_norm.running_var"], bn_training).to(
                    self.device)
                # pems07_feat = self.combine_pems07_linear(
                pems07_feat = functional_linear(params["combine_pems07_linear.weight"],
                                                params["combine_pems07_linear.bias"],
                                                # self.private_pems07_linear(pems07_feat) +
                                                functional_linear(params["private_pems07_linear.weight"],
                                                                  params["private_pems07_linear.bias"],
                                                                  pems07_feat)
                                                +
                                                functional_linear(params["shared_pems07_linear.weight"],
                                                                  params["shared_pems07_linear.bias"],
                                                                  shared_pems07_feat)
                                                # self.shared_pems07_linear(shared_pems07_feat)
                                                )
                h_pems07 = pems07_feat.expand(self.batch_size, self.pems07_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor.functional_forward(feat, h_pems07,
                                                               params["speed_predictor.vgru_cell.linear1.weights"],
                                                               params["speed_predictor.vgru_cell.linear1.biases"],
                                                               params["speed_predictor.vgru_cell.linear2.weights"],
                                                               params["speed_predictor.vgru_cell.linear2.biases"],
                                                               params["speed_predictor.vgru_cell.weights"],
                                                               params["speed_predictor.vgru_cell.bias"],
                                                               params["speed_predictor.vgru_cell.linear.weight"],
                                                               params["speed_predictor.vgru_cell.linear.bias"]
                                                               )
                # pred = self.pems07_linear(pred)
                pred = functional_linear(params["pems07_linear.weight"], params["pems07_linear.bias"], pred)
                pred = pred.reshape((self.batch_size, self.pems07_adj.shape[0], -1))
            elif data_set == '8' or data_set == 'dc':
                shared_pems08_feat = self.shared_pems08_featExtractor.functional_forward(vec_pems08, self.pems08_adj,
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer1.weight"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer1.bias"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer2.weight"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.adj_encoderlayer2.bias"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.batch_norm.weight"],
                                                                                         params[
                                                                                             "shared_pems08_featExtractor.batch_norm.bias"],
                                                                                         bn_vars[
                                                                                             "shared_pems08_featExtractor.batch_norm.running_mean"],
                                                                                         bn_vars[
                                                                                             "shared_pems08_featExtractor.batch_norm.running_var"]
                                                                                         , bn_training).to(self.device)
                pems08_feat = self.pems08_featExtractor.functional_forward(vec_pems08, self.pems08_adj,
                                                                           params[
                                                                               "pems08_featExtractor.adj_encoderlayer1.weight"],
                                                                           params[
                                                                               "pems08_featExtractor.adj_encoderlayer1.bias"],
                                                                           params[
                                                                               "pems08_featExtractor.adj_encoderlayer2.weight"],
                                                                           params[
                                                                               "pems08_featExtractor.adj_encoderlayer2.bias"],
                                                                           params[
                                                                               "pems08_featExtractor.batch_norm.weight"],
                                                                           params[
                                                                               "pems08_featExtractor.batch_norm.bias"],
                                                                           bn_vars[
                                                                               "pems08_featExtractor.batch_norm.running_mean"],
                                                                           bn_vars[
                                                                               "pems08_featExtractor.batch_norm.running_var"], bn_training).to(
                    self.device)
                # pems08_feat = self.combine_pems08_linear(
                pems08_feat = functional_linear(params["combine_pems08_linear.weight"],
                                                params["combine_pems08_linear.bias"],
                                                # self.private_pems08_linear(pems08_feat) +
                                                functional_linear(params["private_pems08_linear.weight"],
                                                                  params["private_pems08_linear.bias"],
                                                                  pems08_feat)
                                                +
                                                functional_linear(params["shared_pems08_linear.weight"],
                                                                  params["shared_pems08_linear.bias"],
                                                                  shared_pems08_feat)
                                                # self.shared_pems08_linear(shared_pems08_feat)
                                                )
                h_pems08 = pems08_feat.expand(self.batch_size, self.pems08_adj.shape[0], self.encode_dim)
                pred = self.speed_predictor.functional_forward(feat, h_pems08,
                                                               params["speed_predictor.vgru_cell.linear1.weights"],
                                                               params["speed_predictor.vgru_cell.linear1.biases"],
                                                               params["speed_predictor.vgru_cell.linear2.weights"],
                                                               params["speed_predictor.vgru_cell.linear2.biases"],
                                                               params["speed_predictor.vgru_cell.weights"],
                                                               params["speed_predictor.vgru_cell.bias"],
                                                               params["speed_predictor.vgru_cell.linear.weight"],
                                                               params["speed_predictor.vgru_cell.linear.bias"]
                                                               )
                # pred = self.pems08_linear(pred)
                pred = functional_linear(params["pems08_linear.weight"], params["pems08_linear.bias"], pred)
                pred = pred.reshape((self.batch_size, self.pems08_adj.shape[0], -1))

            return pred
