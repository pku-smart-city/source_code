import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from model.tgcn import TGCN
import pickle as pkl
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import BatchNorm, GraphNorm
import networkx as nx
import scipy.sparse as sp

# 零膨胀 Define the NB class first, not mixture version
class NBNorm_ZeroInflated(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm_ZeroInflated,self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.p_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)

        self.pi_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
                                    
        self.out_dim = c_out # output horizon

    def forward(self,x):
        x = x.permute(0,2,1,3) # [64, 32, 400, 1]
        (B, _, N,_) = x.shape # B: batch_size; N: input nodes
        n  = self.n_conv(x).squeeze_(-1)
        p  = self.p_conv(x).squeeze_(-1)
        pi = self.pi_conv(x).squeeze_(-1) # prob_layer

        # Reshape
        n = n.view([B,self.out_dim,N])
        p = p.view([B,self.out_dim,N])
        pi = pi.view([B,self.out_dim,N]) 

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n) # Some parameters can be tuned here
        p = F.sigmoid(p)
        pi = F.sigmoid(pi)
        return n.permute([0,2,1]), p.permute([0,2,1]), pi.permute([0,2,1])
    
class GCN_Layer(nn.Module):
    def __init__(self,num_of_features,num_of_filter):
        """One layer of GCN
        
        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer,self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features = num_of_features,
                    out_features = num_of_filter),
            nn.ReLU()
        )
    def forward(self,input,adj):
        """计算一层GCN
        
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size,_,_ = input.shape
        if isinstance(adj,np.ndarray):
            adj = torch.from_numpy(adj).to(input.device)
        else:
            adj = adj.to(input.device)

        adj = adj.repeat(batch_size,1,1)
        input = torch.bmm(adj, input)
        output = self.gcn_layer(input)
        return output

def normalize_adj(adj):
    """ 归一化邻接矩阵：A_normalized = D^(-1/2) * A * D^(-1/2) """
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))  # 计算每个节点的度
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # D^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 处理除零错误
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 转换为对角矩阵
    return (d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt).tocoo() 

# ST-Net
class STNet(nn.Module):
    def __init__(self, hidden_dim, input_dim, gru_hidden_dim, output_dim, num_layers=2):
        super(STNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        
        self.bn = nn.BatchNorm1d(gru_hidden_dim)
        self.gcn_layers = nn.ModuleList([GCN_Layer(gru_hidden_dim, gru_hidden_dim) for _ in range(num_layers)])
        
        self.output_layer = nn.Linear(gru_hidden_dim, output_dim)    
    def forward(self, XS, XT, AS,AT):   # XS.shape=(32, 6, 54, 20, 20)
        # 构造增广矩阵
        NS, NT = AS.shape[0], AT.shape[0] 
        AS_norm = normalize_adj(AS).toarray()
        AT_norm = normalize_adj(AT).toarray()
        A_aug = np.block([
        [AS_norm, np.zeros((NS, NT))],  # 左上角: AS
        [np.zeros((NT, NS)), AT_norm]   # 右下角: AT
        ])
        A = torch.tensor(A_aug, dtype=torch.float32)

        # 提取时间周期性
        XS1 = XS.reshape(XS.shape[0],XS.shape[1],XS.shape[2],-1).permute(0,3,1,2)
        XT1 = XT.reshape(XT.shape[0],XT.shape[1],XT.shape[2],-1).permute(0,3,1,2)
        X = torch.cat([XS1, XT1], dim=1)  # [32, 800, 6, 54]
        X = self.fc(X)  # [32, 800, 6, 32]
        bs, node_num, time_dim, hidden_dim = X.shape
        X = X.view(bs*node_num,time_dim,hidden_dim)
        h_0 = torch.zeros(1, X.shape[0], X.shape[2]).to(X.device)  # Initial hidden state [1, 32, 25600]
        X, _ = self.gru(X, h_0)  # [25600, 6, 32]
        x1 = X[:,-1,:]
        x2 = self.bn(x1).view(bs,node_num,-1)  # [32, 800, 32]

        # 提取空间性
        for gcn_layer in self.gcn_layers:
            x2 = gcn_layer(x2, A)  # [32, 800, 32]
        
        # Final output layer
        output = self.output_layer(x2)

        
        return output[:,:NS,:],output[:,NS:,:]

class DenseGATConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GATConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        # TODO Add support for edge features.
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, 1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, 1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        x = self.lin(x).view(B, N, H, C)  # [B, N, H, C]

        alpha_src = torch.sum(x * self.att_src, dim=-1)  # [B, N, H]
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)  # [B, N, H]

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H]

        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))
        out = out.movedim(1, 2)  # [B,N,H,C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layer=1):
        super(GNN, self).__init__()
        self.gnn1 = DenseGATConv(in_channels, hidden_channels, heads=2)
        self.bn1 = GraphNorm(hidden_channels)
        self.gnn2 = DenseGATConv(hidden_channels, out_channels, heads=2)
        self.bn2 = GraphNorm(out_channels)

    def forward(self, x, adj, mask=None):
        batch_size, node_num, _ = x.shape
        x = self.gnn1(x, adj, mask)
        x = self.bn1(x)
        x = self.gnn2(x, adj, mask)
        x = self.bn2(x)

        return x


def dense_diff_pool(x, adj, s, mask=None, normalize: bool = True):
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper
    
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s # 每个节点属于不同集群的概率

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x) # x
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s) # 论文中公式 s^{l}

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2) # 计算所有元素的平方和，再开平方
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean() # 熵损失，entropy func

    return out, out_adj, link_loss, ent_loss


class DiffPool(nn.Module): # Hierarchical node clustering
    def __init__(self, input_dim, hidden_dim, num_nodes1=100, num_nodes2=10):
        super(DiffPool, self).__init__()
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes1)
        self.gnn1_embed = GNN(input_dim, hidden_dim, input_dim)

        self.gnn2_pool = GNN(input_dim, hidden_dim, num_nodes2)
        self.gnn2_embed = GNN(input_dim, hidden_dim, input_dim)
        self.temperature = hidden_dim ** 0.5

    def forward(self, x, adj, origin_seq, mask=None):
        n2z_s = self.gnn1_pool(x, adj, mask) 
        n2z_x = self.gnn1_embed(x, adj, mask) 

        zone_x, adj_z, l1, e1 = dense_diff_pool(n2z_x, adj, n2z_s, mask)
        z2s_s = self.gnn2_pool(zone_x, adj_z, mask)
        z2s_x = self.gnn2_embed(zone_x, adj_z, mask)
        zone_temp = torch.bmm(torch.softmax(n2z_s, dim=-1).transpose(1, 2), origin_seq.transpose(1, 2)).transpose(1, 2)

        semantic_x, adj_s, l2, e2 = dense_diff_pool(z2s_x, adj_z, z2s_s) # semantic_x的值很大
        semantic_x = torch.mean(semantic_x, dim=1, keepdim=True) # [32, 1, 32]

        semantic_temp = torch.bmm(torch.softmax(z2s_s, dim=-1).transpose(1, 2), zone_temp.transpose(1, 2))#  [32, 10, 1]
        semantic_temp = semantic_temp.mean(dim=-1, keepdim=True) # [32, 10, 1]

        link_loss = l1 + l2
        ent_loss = e1 + e2

        qs = torch.cat([zone_x, semantic_x], dim=1)

        return link_loss, ent_loss, [zone_x, zone_temp.transpose(1, 2)], [semantic_x, semantic_temp], qs 


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
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
    """
    Extension of grad reverse layer
    """
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


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim, device='cuda:3'):
        super(DomainDiscriminator, self).__init__()

        self.adversarial_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, 2)).to(device)

    def forward(self, embed, alpha, if_reverse):
        if if_reverse:
            embed = GradReverse.grad_reverse(embed, alpha)
        else:
            embed = Grad.grad(embed, alpha)

        out = self.adversarial_mlp(embed)

        return F.log_softmax(out, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim=16, device='cuda:3'):
        super(Encoder, self).__init__()
        self.gnn1 = TGCN(hidden_dim)
        self.gnn2 = TGCN(hidden_dim)
        self.gnn3 = TGCN(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim*3+33, hidden_dim), #(127,32) 127 = hidden_dim*3+31
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), #(127,32) 127 = hidden_dim*3+31
                                 nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, hidden_dim)) # chicago
        self.device = device
        self.hidden_dim = hidden_dim

        self.semantic1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fuse_weight = nn.Parameter(torch.randn(1, hidden_dim)).unsqueeze(0).to(device)

    def forward(self, node_seq, label_seq, t, adj1, adj2, adj3, pre_train=False,city='nyc'):
        batch_size = node_seq.shape[0]
        length = node_seq.shape[1]
        node_seq = node_seq.reshape(batch_size, length, -1)
        node_num = node_seq.shape[2]

        adj1 = torch.from_numpy(adj1).to(self.device).float()
        adj2 = torch.from_numpy(adj2).to(self.device).float()
        node_embed1 = self.gnn1(node_seq, adj1)
        node_embed2 = self.gnn2(node_seq, adj2)
        adj3 = torch.from_numpy(adj3).to(self.device).float()
        node_embed3 = self.gnn3(node_seq, adj3) # [32, 400, 32]
        t = t.unsqueeze(dim=1).repeat(1,node_num,1) 
        # t = t.repeat(1, node_num, 1)
        concat_node = torch.cat([node_embed1, node_embed2, node_embed3, t], dim=-1)
        fused_node = self.mlp(concat_node) # [64, 400, 32]
        rec_adj1 = F.sigmoid(torch.bmm(self.semantic1(fused_node), fused_node.transpose(1, 2))) # 和论文中的公式有点不一样，重建邻接矩阵 [64, 400, 400]
        rec_adj2 = F.sigmoid(torch.bmm(self.semantic2(fused_node), fused_node.transpose(1, 2)))
        rec_adj3 = F.sigmoid(torch.bmm(self.semantic3(fused_node), fused_node.transpose(1, 2)))
        rec_loss = F.mse_loss(rec_adj1, adj1) + F.mse_loss(rec_adj2, adj2) + F.mse_loss(rec_adj3, adj3)
        
        weighted_fuse_node = fused_node * self.fuse_weight
        fused_adj = F.softmax(F.relu(torch.bmm(weighted_fuse_node, weighted_fuse_node.transpose(1, 2))), dim=-1) # a momentary graph
        ent_loss = (-fused_adj * torch.log(fused_adj + 1e-15)).sum(dim=-1).mean() # 这里是什么损失？

        return fused_node, fused_adj, rec_loss+ent_loss
         

class Scoring(nn.Module):
    # 这里的权重网络有问题
    def __init__(self, emb_dim, source_mask, target_mask): 
        """
        emb_dim: 特征维度
        source_mask: 源城市掩码 (20, 20)
        target_mask: 目标城市掩码 (20, 20)
        """
        super().__init__()
        self.emb_dim = emb_dim
        
        # 权重网络
        self.score = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 2)
        )
        
        # 将 mask 展开到 (400,) 形状
        self.source_mask = source_mask.reshape(-1)  # (400,)
        self.target_mask = target_mask.reshape(-1)  # (400,)

    def forward(self, source_emb, target_emb):
        """
        source_emb: 源城市特征 [bs, 400, emb_dim]
        target_emb: 目标城市特征 [bs, 400, emb_dim]
        """
        batch_size = source_emb.size(0)  # 获取 batch size
        
        # 将 mask 扩展到 batch 维度，形状变为 (64, 400)
        s_mask_2 = torch.from_numpy(
            np.tile(self.source_mask.astype(bool), (batch_size, 1))
        ).to(source_emb.device)
        t_mask_2 = torch.from_numpy(
            np.tile(self.target_mask.astype(bool), (batch_size, 1))
        ).to(target_emb.device)

        # 使用目标城市掩码提取目标上下文向量，形状为 (16,)
        target_context = torch.tanh(
            self.score(target_emb).mean(0)  # 通过掩码选择并求均值
        ) # (400,16)

        # 转换源城市特征，形状为 [64, 400, 16]
        source_trans_emb = self.score(source_emb) # (32, 400, 16)

        # 计算源城市特征和目标上下文的相似性得分，形状为 [64, 400]
        source_score = (source_trans_emb * target_context).sum(-1)  # 计算内积

        # 可选：将内积改为余弦相似度
        target_norm = target_context.pow(2).sum().sqrt()  # 目标上下文的范数
        source_norm = source_trans_emb.pow(2).sum(-1).sqrt()  # 源城市特征的范数
        source_score = source_score / (source_norm + 1e-8)  # 避免除以0
        source_score = source_score / (target_norm + 1e-8)

        # 激活函数 + 掩码筛选，输出形状为 [有效点数,]
        output_score = F.relu(torch.tanh(source_score))

        return output_score

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
    
class SFMGTL(nn.Module):
    def __init__(self, args,GSNet_Model1,GSNet_Model2,hidden_dim=16, device='cuda:3',th_mask_source=None,th_mask_target=None,\
                 num_of_graph_feature = None,nums_of_graph_filters=[],seq_len=None,num_of_time_feature=None,knowledge_number=16):
        super(SFMGTL, self).__init__()
        self.node_num = args.west_east_map * args.north_south_map
        self.s_mask = th_mask_source
        self.t_mask = th_mask_target

        # self.encoder = STNet(hidden_dim, input_dim=args.num_of_x_feat, gru_hidden_dim=args.gru_hidden_size, output_dim=32, num_layers=args.gcn_num_layer)

        self.s_encoder = GSNet_Model1
        self.t_encoder = GSNet_Model1
        # self.s_encoder = Encoder(hidden_dim=hidden_dim, device=device)
        # self.t_encoder = Encoder(hidden_dim=hidden_dim, device=device)

        self.discriminator = DomainDiscriminator(hidden_dim,device)
        self.device = device

        self.s_diffpool = DiffPool(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
        self.t_diffpool = DiffPool(input_dim=hidden_dim, hidden_dim=hidden_dim).to(device)

        self.s_head = nn.Linear(hidden_dim, 1)
        self.s_head_zone = nn.Linear(hidden_dim, 1)
        self.s_head_semantic = nn.Linear(hidden_dim, 1)

        self.t_head = nn.Linear(hidden_dim, 1)
        self.t_head_zone = nn.Linear(hidden_dim, 1)
        self.t_head_semantic = nn.Linear(hidden_dim, 1)

        # Common Knowledge
        self.common_attention = nn.ModuleList()
        for _ in range(9):
            self.common_attention.append(nn.Linear(hidden_dim, hidden_dim, bias=False).to(device))
            self.common_attention.append(nn.Linear(hidden_dim, hidden_dim, bias=False).to(device))
            self.common_attention.append(nn.Linear(hidden_dim, hidden_dim, bias=False).to(device))

        self.knowledge_number = knowledge_number # 参数实验2
        self.pk = 12
        self.dk = 16
        self.source_knowledge = nn.Parameter(torch.randn(self.knowledge_number,self.pk, self.dk)).to(device)
        self.target_knowledge = nn.Parameter(torch.randn(self.knowledge_number,self.pk, self.dk)).to(device)

        self.node_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to(device)
        self.zone_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to(device)
        self.semantic_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to(device)

        self.private_node_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to(device)
        self.private_zone_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to(device)
        self.private_semantic_knowledge = nn.Parameter(torch.randn(self.knowledge_number, hidden_dim)).to(device)

        self.temperature = hidden_dim ** 0.5
        self.scoring = Scoring(hidden_dim, th_mask_source, th_mask_target).to(device)
        
        tnb_outdim = args.pred_lag
        self.s_tnb = NBNorm_ZeroInflated(hidden_dim,tnb_outdim).to(device=device)
        self.t_tnb = NBNorm_ZeroInflated(hidden_dim,tnb_outdim).to(device=device)

        self.semantic1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fuse_weight = nn.Parameter(torch.randn(1, hidden_dim)).unsqueeze(0).to(device)

        # 平衡因子生成
        self.mmd = MMD_loss()
    def forward(self, args, s_x, s_xt,s_y, s_yt, s_adj1, s_adj2, s_adj3,
                      t_x,t_xt, t_y, t_yt, t_adj1, t_adj2, t_adj3, alpha, if_reverse):
        s_bs,pre_len , W,H = s_y.shape
        t_bs = t_y.shape[0]
        
        s_fused_node,s_fused_adj,s_rec_loss = self.s_encoder(s_x, s_yt,s_xt, s_adj1, s_adj2, s_adj3) # risk_adj,road_adj,poi_adj x.shape=32, 6, 54, 20, 20 out.shape = (32, 400, 32)
        t_fused_node,t_fused_adj,t_rec_loss = self.t_encoder(t_x, t_yt,t_xt, t_adj1, t_adj2, t_adj3) # xt.shape = [32, 6, 2, 400]

        # 图重构，层次节点聚类
        s_link_loss, s_ent_loss, s_zone_temp, s_semantic_temp, s_qs = self.s_diffpool(s_fused_node, s_fused_adj, s_y.view(s_y.shape[0], 1, -1)) # 节点聚类，不是真正的聚类，，
        t_link_loss, t_ent_loss, t_zone_temp, t_semantic_temp, t_qs = self.t_diffpool(t_fused_node, t_fused_adj, t_y.view(t_y.shape[0], 1, -1)) 
        s_aux_loss = s_link_loss + s_ent_loss
        t_aux_loss = t_link_loss + t_ent_loss

        # 提取动态交通模式的元知识
        node_q = self.common_attention[0](s_fused_node) # FC [32, 400, 32]
        # sfmgtl
        k = self.common_attention[1](self.node_knowledge) # node_knowledge 参数化
        v = self.common_attention[2](self.node_knowledge)
        attn = torch.matmul(node_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        s_fused_node = attn @ v # [32, 400, 32]

        zone_q = self.common_attention[3](s_zone_temp[0])
        k = self.common_attention[4](self.zone_knowledge)
        v = self.common_attention[5](self.zone_knowledge)
        attn = torch.matmul(zone_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        s_zone_temp[0] = attn @ v

        semantic_q = self.common_attention[6](s_semantic_temp[0])
        k = self.common_attention[7](self.semantic_knowledge)
        v = self.common_attention[8](self.semantic_knowledge)
        attn = torch.matmul(semantic_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        s_semantic_temp[0] = attn @ v

        s_pred = self.s_head(s_fused_node) # 改
        # abla-v2
        s_pred_zone = self.s_head_zone(s_zone_temp[0])
        s_pred_semantic = self.s_head_semantic(s_semantic_temp[0])
        s_zone_loss = F.mse_loss(s_pred_zone, s_zone_temp[1])
        s_semantic_loss = F.mse_loss(s_pred_semantic, s_semantic_temp[1])
        s_aux_loss += s_zone_loss
        s_aux_loss += s_semantic_loss
        s_aux_loss += s_rec_loss  # abla-v1

        s_q = torch.cat([node_q, zone_q, semantic_q], dim=1)
        # s_q = node_q # abla-v2

        ######################################## target 上面进行类似的操作 ##################################### 
        node_q = self.common_attention[0](t_fused_node)
        # sfmgtl
        k = self.common_attention[1](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        v = self.common_attention[2](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        attn = torch.matmul(node_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        t_fused_node = attn @ v

        zone_q = self.common_attention[3](t_zone_temp[0])
        k = self.common_attention[4](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        v = self.common_attention[5](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        attn = torch.matmul(zone_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        t_zone_temp[0] = attn @ v

        semantic_q = self.common_attention[6](t_semantic_temp[0])
        k = self.common_attention[7](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        v = self.common_attention[8](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        attn = torch.matmul(semantic_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        t_semantic_temp[0] = attn @ v

        t_pred = self.t_head(t_fused_node)
        # abla-v2
        t_pred_zone = self.t_head_zone(t_zone_temp[0])
        t_pred_semantic = self.t_head_semantic(t_semantic_temp[0])
        t_zone_loss = F.mse_loss(t_pred_zone, t_zone_temp[1])
        t_semantic_loss = F.mse_loss(t_pred_semantic, t_semantic_temp[1])
        t_aux_loss += t_zone_loss
        t_aux_loss += t_semantic_loss
        t_aux_loss += t_rec_loss  # abla-v1
        t_q = torch.cat([node_q, zone_q, semantic_q], dim=1)
        # t_q = node_q # abla-v2

        #### Adversarial 对抗训练
        s_cls = self.discriminator(s_q, alpha, if_reverse).view(-1, 2)
        t_cls = self.discriminator(t_q, alpha, if_reverse).view(-1, 2)
        s_label = torch.zeros(s_cls.shape[0]).long().to(self.device)
        t_label = torch.ones(t_cls.shape[0]).long().to(self.device)

        acc_t = ((torch.exp(t_cls)[:, 1] > 0.5).float().sum()) / (t_cls.shape[0])
        acc_s = ((torch.exp(s_cls)[:, 0] > 0.5).float().sum()) / (s_cls.shape[0])
        accuracy = (acc_s + acc_t) / 2
        adversarial_s_loss = F.nll_loss(s_cls, s_label)
        adversarial_t_loss = F.nll_loss(t_cls, t_label)

        # 生成权重
        # source_weights = self.scoring(s_fused_node, t_fused_node)  # s.shape = (32,400,32) weights.shape = (32,400)
        # source_weights=0

        # 平衡因子生成
        source_ids = np.random.randint(0, np.sum(self.s_mask), size = (128, ))
        target_ids = np.random.randint(0, np.sum(self.t_mask), size = (128, ))
        mmd_loss = self.mmd(s_fused_node[:,self.s_mask.reshape(-1).astype(bool),:][:,source_ids, :].permute(1,0,2).reshape(len(source_ids),-1), s_fused_node[:,self.t_mask.reshape(-1).astype(bool),:][:,target_ids, :].permute(1,0,2).reshape(len(source_ids),-1))

        T_complex = mmd_loss/(mmd_loss+ s_link_loss+t_link_loss) # 另一个值总觉得应该重新算一个 DTW, link_loss=0.0006,mmd_loss=0.1074
        T = T_complex.real

        # 零膨胀
        _b,_n,_hs = s_fused_node.shape # [64, 400, 32]
        n_s_nb,p_s_nb,pi_s_nb = self.s_tnb(s_fused_node.view(_b,_n,_hs,1)) #n.shape = [64, 400, 400]
        _b,_n,_ht = t_fused_node.shape
        n_t_nb,p_t_nb,pi_t_nb = self.t_tnb(t_fused_node.view(_b,_n,_ht,1))
        
        # abla-v3
        #  adversarial_s_loss,adversarial_t_loss = 0,0
        return s_pred.transpose(1, 2).reshape(s_bs,pre_len,W,H), t_pred.transpose(1, 2).reshape(t_bs,pre_len,W,H), s_aux_loss, t_aux_loss, \
                accuracy, (adversarial_s_loss+adversarial_t_loss), T,[n_s_nb,p_s_nb,pi_s_nb],[n_t_nb,p_t_nb,pi_t_nb]

    def evaluation(self, x, xt,y, yt, adj1, adj2, adj3):
        bs,pre_len , W,H = y.shape
        fused_node, fused_adj, rec_loss = self.t_encoder(x, yt,xt,adj1, adj2, adj3) 
        # fused_node, fused_adj, rec_loss = self.t_encoder(x[:,:,0,:,:], yt,yt,adj1, adj2, adj3)

        link_loss, ent_loss, zone_temp, semantic_temp, t_qs = self.t_diffpool(fused_node, fused_adj,
                                                                   y.view(y.shape[0], 1, -1))
        aux_loss = link_loss + ent_loss
        
        node_q = self.common_attention[0](fused_node)
        k = self.common_attention[1](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        v = self.common_attention[2](torch.cat([self.node_knowledge, self.private_node_knowledge], dim=0))
        attn = torch.matmul(node_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        fused_node = attn @ v

        zone_q = self.common_attention[3](zone_temp[0])
        k = self.common_attention[4](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        v = self.common_attention[5](torch.cat([self.zone_knowledge, self.private_zone_knowledge], dim=0))
        attn = torch.matmul(zone_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        zone_temp[0] = attn @ v

        semantic_q = self.common_attention[6](semantic_temp[0])
        k = self.common_attention[7](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        v = self.common_attention[8](torch.cat([self.semantic_knowledge, self.private_semantic_knowledge], dim=0))
        attn = torch.matmul(semantic_q / self.temperature, k.transpose(0, 1))
        attn = F.softmax(attn, dim=-1)
        semantic_temp[0] = attn @ v

        pred = self.t_head(fused_node)
        # abla-v2
        pred_zone = self.t_head_zone(zone_temp[0])
        pred_semantic = self.t_head_semantic(semantic_temp[0])
        zone_loss = F.mse_loss(pred_zone, zone_temp[1])
        semantic_loss = F.mse_loss(pred_semantic, semantic_temp[1])
        aux_loss += zone_loss
        aux_loss += semantic_loss
        aux_loss += rec_loss  # abla-v1

        # 零膨胀
        _b,_n,_hs = fused_node.shape # [64, 400, 32]
        n_nb,p_nb,pi_nb = self.t_tnb(fused_node.view(_b,_n,_hs,1)) #n.shape = [64, 400, 400]


        return pred.transpose(1, 2).reshape(bs,pre_len,W,H),aux_loss,[n_nb,p_nb,pi_nb]