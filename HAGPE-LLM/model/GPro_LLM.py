import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import DotDict
from model.utils import *
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .llm import LLMModel
from sklearn.cluster import KMeans
import pickle
import os


class GPro_LLM_ModelConfig(DotDict):
    """
    Configuration for GPro-LLM (Global Prototype-enhanced LLM)
    """
    def __init__(self, loc_size=None, tim_size=None, uid_size=None, geohash_size=None, category_size=None, 
                 tim_emb_size=None, loc_emb_size=None, hidden_size=None, user_emb_size=None, 
                 model_class=None, device=None, loc_noise_mean=None, loc_noise_sigma=None, 
                 tim_noise_mean=None, tim_noise_sigma=None, user_noise_mean=None, user_noise_sigma=None, 
                 tau=None, pos_eps=None, neg_eps=None, dropout_rate_1=None, dropout_rate_2=None, 
                 category_vector=None, rnn_type='BiLSTM', num_layers=3, k=8, momentum=0.95, 
                 temperature=0.1, theta=0.18, n_components=4, shift_init=0.0, scale_init=0.0, 
                 min_clip=-5., max_clip=3., hypernet_hidden_sizes=None, max_delta_mins=1440,
                 downstream='POI', tpp='pdf', loss='pdf', dropout_spatial=None, epsilon=None, 
                 learnable_param_size=1,
                 # GPro-LLM specific parameters
                 num_prototypes=8,          # Number of global prototypes (K)
                 num_hgnn_layers=2,         # Number of HGNN convolution layers
                 hgnn_dropout=0.5,          # Dropout rate for HGNN layers
                 gate_hidden_size=128,      # Hidden size for triple-gate fusion MLP
                 prototype_dim=256,         # Dimension of prototype representations
                 stage='pretrain',          # 'pretrain' or 'finetune'
                 prototype_path=None,       # Path to save/load prototypes
                 freeze_frontend=False):    # Whether to freeze frontend modules
        super().__init__()
        
        # Original MobilityLLM parameters
        self.max_delta_mins = max_delta_mins
        self.loc_size = loc_size
        self.uid_size = uid_size
        self.tim_size = tim_size
        self.geohash_size = geohash_size
        self.category_size = category_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.user_emb_size = user_emb_size
        self.hidden_size = hidden_size
        self.model_class = model_class
        self.device = device
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.loc_noise_mean = loc_noise_mean
        self.loc_noise_sigma = loc_noise_sigma
        self.tim_noise_mean = tim_noise_mean
        self.tim_noise_sigma = tim_noise_sigma
        self.user_noise_mean = user_noise_mean
        self.user_noise_sigma = user_noise_sigma
        self.tau = tau
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.downstream = downstream
        self.category_vector = category_vector
        self.learnable_param_size = learnable_param_size
        self.k = k
        self.momentum = momentum
        self.theta = theta
        self.temperature = temperature
        self.n_components = n_components
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.shift_init = shift_init
        self.scale_init = scale_init
        self.hypernet_hidden_sizes = hypernet_hidden_sizes
        self.decoder_input_size = user_emb_size + hidden_size * 2
        self.loss = loss
        self.tpp = tpp
        self.dropout_spatial = dropout_spatial
        self.epsilon = epsilon
        
        # GPro-LLM specific
        self.num_prototypes = num_prototypes
        self.num_hgnn_layers = num_hgnn_layers
        self.hgnn_dropout = hgnn_dropout
        self.gate_hidden_size = gate_hidden_size
        self.prototype_dim = prototype_dim
        self.stage = stage
        self.prototype_path = prototype_path
        self.freeze_frontend = freeze_frontend


class HGNNConv(nn.Module):
    """
    Hypergraph Neural Network Convolution Layer (HGNN, Feng et al. 2019)
    
    Computes: X_out = LeakyReLU(D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X W + b)
    
    Efficient implementation avoids forming the NxN Laplacian matrix:
    X_out = D_v^{-1/2} @ H @ D_e^{-1} @ H^T @ (D_v^{-1/2} @ X @ W)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(HGNNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, X, H):
        """
        Args:
            X: Node features [N_total, in_features]
            H: Incidence matrix [N_total, M] (M = num_hyperedges)
        
        Returns:
            Output features [N_total, out_features]
        """
        # Compute degree vectors
        D_v = H.sum(dim=1)  # [N_total] node degree
        D_e = H.sum(dim=0)  # [M] hyperedge degree
        
        # Degree normalization (handle zero-degree nodes by setting inv to 0)
        D_v_inv_sqrt = torch.zeros_like(D_v)
        non_zero_v = D_v > 0
        D_v_inv_sqrt[non_zero_v] = 1.0 / (D_v[non_zero_v].sqrt() + 1e-8)
        
        D_e_inv = torch.zeros_like(D_e)
        non_zero_e = D_e > 0
        D_e_inv[non_zero_e] = 1.0 / (D_e[non_zero_e] + 1e-8)
        
        # Step 1: Linear transformation
        XW = torch.matmul(X, self.weight)  # [N_total, out_features]
        
        # Step 2: D_v^{-1/2} @ XW (row-wise scaling)
        a = XW * D_v_inv_sqrt.unsqueeze(1)  # [N_total, out_features]
        
        # Step 3: H^T @ a → [M, out_features]
        b = torch.matmul(H.t(), a)  # [M, out_features]
        
        # Step 4: D_e^{-1} @ b (row-wise scaling)
        c = b * D_e_inv.unsqueeze(1)  # [M, out_features]
        
        # Step 5: H @ c → [N_total, out_features]
        d = torch.matmul(H, c)  # [N_total, out_features]
        
        # Step 6: D_v^{-1/2} @ d (row-wise scaling)
        e = d * D_v_inv_sqrt.unsqueeze(1)  # [N_total, out_features]
        
        # Add bias
        if self.bias is not None:
            e = e + self.bias
        
        return F.leaky_relu(e, negative_slope=0.2)


class TripleGateFusion(nn.Module):
    """
    Triple-source Gating Fusion Module
    
    Adaptively fuses three representations using learned gate weights:
    - V: individual preference
    - V_enhanced: HGNN-enhanced preference (captures group structure via hypergraph)
    - V_global: soft-retrieved global preference (weighted sum of prototypes)
    
    Gate computation:
        concat = [V; V_enhanced; V_global]    → [B, L, 3d]
        gate = softmax(MLP(concat))            → [B, L, 3]  (3 weights sum to 1)
        V_fused = gate_0 * V + gate_1 * V_enhanced + gate_2 * V_global
    """
    def __init__(self, hidden_size, gate_hidden_size=128):
        super(TripleGateFusion, self).__init__()
        self.hidden_size = hidden_size
        
        # Gate network: concat 3 sources → 3 gate weights
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, gate_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_size, 3),
            nn.Softmax(dim=-1)  # 3 weights sum to 1
        )
    
    def forward(self, V, V_enhanced, V_global):
        """
        Args:
            V: Individual preference [batch, seq_len, hidden_size]
            V_enhanced: HGNN-enhanced preference [batch, seq_len, hidden_size]
            V_global: Soft-retrieved global preference [batch, seq_len, hidden_size]
        
        Returns:
            V_fused: Fused representation [batch, seq_len, hidden_size]
            gate_weights: Gate weights for interpretability [batch, seq_len, 3]
        """
        # Concatenate three sources
        concat = torch.cat([V, V_enhanced, V_global], dim=-1)  # [B, L, 3d]
        
        # Compute gate weights (softmax ensures they sum to 1)
        gate_weights = self.gate_layer(concat)  # [B, L, 3]
        
        # Weighted fusion
        w0 = gate_weights[..., 0:1]  # [B, L, 1]
        w1 = gate_weights[..., 1:2]
        w2 = gate_weights[..., 2:3]
        
        V_fused = w0 * V + w1 * V_enhanced + w2 * V_global  # [B, L, d]
        
        return V_fused, gate_weights


class HypergraphPrototypeModule(nn.Module):
    """
    Hypergraph-based Global Prototype Module for GPro-LLM (v4)
    
    Produces two complementary group-aware representations:
    1. V_enhanced: HGNN convolution output (structure-aware aggregation via hypergraph)
    2. V_global: Soft-retrieved global preference (V @ prototype_V^T → weights → weighted sum)
    
    These are combined with the original V through TripleGateFusion to produce V_fused.
    
    Hypergraph construction:
    - Nodes: N individual V representations + K prototype_V (cluster centers)
    - Intra-cluster hyperedges: K hyperedges, each connecting all individuals in a cluster
      with the corresponding prototype (capturing within-group preference patterns)
    - Inter-cluster hyperedges: K hyperedges, each connecting a prototype with its 
      2 most similar prototypes (capturing between-group relationships)
    """
    def __init__(self, hidden_size, num_prototypes=8, num_hgnn_layers=2, hgnn_dropout=0.5, gate_hidden_size=128):
        super(HypergraphPrototypeModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.num_hgnn_layers = num_hgnn_layers
        
        # Global prototypes (cluster centers) - initialized from K-means, not learnable
        self.prototype_H = nn.Parameter(torch.randn(num_prototypes, hidden_size), requires_grad=False)
        self.prototype_V = nn.Parameter(torch.randn(num_prototypes, hidden_size), requires_grad=False)
        
        # Inter-cluster neighbors: top-2 most similar prototype for each prototype (cosine similarity)
        # Computed after loading prototypes
        self.register_buffer('inter_cluster_neighbors', torch.zeros(num_prototypes, 2, dtype=torch.long))
        
        # Scale factor for soft retrieval attention
        self.scale = np.sqrt(hidden_size)
        
        # HGNN convolution layers with residual connections
        self.hgnn_layers = nn.ModuleList([
            HGNNConv(hidden_size, hidden_size) for _ in range(num_hgnn_layers)
        ])
        
        self.dropout = nn.Dropout(hgnn_dropout)
        
        # Triple-gate fusion: V + V_enhanced + V_global → V_fused
        self.gate_fusion = TripleGateFusion(hidden_size, gate_hidden_size)
    
    def compute_inter_cluster_neighbors(self):
        """
        Compute the top-2 most similar prototypes for each prototype using cosine similarity.
        This determines the inter-cluster hyperedge structure.
        Called after loading prototypes.
        """
        with torch.no_grad():
            proto_norm = F.normalize(self.prototype_V, p=2, dim=-1)
            sim = torch.matmul(proto_norm, proto_norm.T)  # [K, K]
            sim.fill_diagonal_(-float('inf'))  # Exclude self
            _, top2 = sim.topk(2, dim=-1)  # [K, 2]
            self.inter_cluster_neighbors.copy_(top2)
            print(f"Inter-cluster neighbors computed: {top2.tolist()}")
    
    def build_incidence_matrix(self, N, cluster_assignments, mask_flat, device):
        """
        Build the hypergraph incidence matrix H.
        
        Nodes: N individual V's + K prototypes = (N + K) total nodes
        Hyperedges: K intra-cluster + K inter-cluster = 2K total hyperedges
        
        Args:
            N: number of individual nodes (batch_size * seq_len)
            cluster_assignments: [N] cluster index (0..K-1) for each individual node
            mask_flat: [N] boolean, True for valid (non-padded) positions
            device: torch device
        
        Returns:
            H: Incidence matrix [N+K, 2K]
        """
        K = self.num_prototypes
        total_nodes = N + K
        num_hyperedges = 2 * K
        
        H = torch.zeros(total_nodes, num_hyperedges, device=device)
        
        # --- Intra-cluster hyperedges (columns 0..K-1) ---
        # Each hyperedge k connects: prototype_k + all individual V's assigned to cluster k
        valid_indices = torch.where(mask_flat)[0]
        valid_clusters = cluster_assignments[valid_indices]
        H[valid_indices, valid_clusters] = 1.0
        
        # Prototypes always belong to their own cluster's hyperedge
        proto_indices = torch.arange(K, device=device)
        H[N + proto_indices, proto_indices] = 1.0
        
        # --- Inter-cluster hyperedges (columns K..2K-1) ---
        # Each hyperedge K+k connects: prototype_k + its 2 most similar prototypes
        neighbors = self.inter_cluster_neighbors  # [K, 2]
        H[N + proto_indices, K + proto_indices] = 1.0  # Self
        H[N + neighbors[:, 0], K + proto_indices] = 1.0  # Neighbor 1
        H[N + neighbors[:, 1], K + proto_indices] = 1.0  # Neighbor 2
        
        return H
    
    def forward(self, V, mask=None):
        """
        Args:
            V: Individual preference representation [batch, seq_len, hidden_size]
            mask: Valid position mask [batch, seq_len], True for valid positions
        
        Returns:
            V_fused: Gate-fused representation [batch, seq_len, hidden_size]
            gate_weights: Gate weights [batch, seq_len, 3]
            cluster_assignments: Cluster assignment per position [batch, seq_len]
        """
        batch_size, seq_len, _ = V.shape
        N = batch_size * seq_len
        
        # Flatten V
        V_flat = V.reshape(N, self.hidden_size)  # [N, hidden]
        
        # Create validity mask for non-padded positions
        if mask is not None:
            mask_flat = mask.reshape(N).bool()
        else:
            mask_flat = torch.ones(N, dtype=torch.bool, device=V.device)
        
        # ---- 1. Soft retrieval: V_global ----
        # Compute cosine similarity between V and prototype_V
        V_norm = F.normalize(V_flat, p=2, dim=-1)
        proto_norm = F.normalize(self.prototype_V, p=2, dim=-1)
        sim = torch.matmul(V_norm, proto_norm.T)  # [N, K]
        
        # Scaled cosine similarity → softmax → attention weights
        attention_weights = F.softmax(sim / self.scale * np.sqrt(self.hidden_size), dim=-1)  # [N, K]
        
        # Weighted sum of prototype_V
        V_global_flat = torch.matmul(attention_weights, self.prototype_V)  # [N, hidden]
        V_global = V_global_flat.reshape(batch_size, seq_len, self.hidden_size)
        
        # ---- 2. Cluster assignment for hypergraph ----
        cluster_assignments = sim.argmax(dim=-1)  # [N]
        
        # ---- 3. HGNN: V_enhanced ----
        # Build hypergraph incidence matrix
        H = self.build_incidence_matrix(N, cluster_assignments, mask_flat, V.device)
        
        # Initial node features: concatenate individual V's and prototype V's
        X = torch.cat([V_flat, self.prototype_V], dim=0)  # [N+K, hidden]
        
        # HGNN message passing with residual connections
        for layer in self.hgnn_layers:
            X_res = X
            X = layer(X, H)          # HGNN convolution
            X = self.dropout(X)
            X = X + X_res            # Residual connection to preserve individual information
        
        # Extract enhanced individual V representations (first N nodes)
        V_enhanced = X[:N].reshape(batch_size, seq_len, self.hidden_size)
        
        # For padded positions, zero out both outputs
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            V_enhanced = V_enhanced * mask_float
            V_global = V_global * mask_float
        
        # ---- 4. Triple-gate fusion: V + V_enhanced + V_global → V_fused ----
        V_fused, gate_weights = self.gate_fusion(V, V_enhanced, V_global)
        
        # Zero out padded positions in V_fused
        if mask is not None:
            V_fused = V_fused * mask_float
        
        return V_fused, gate_weights, cluster_assignments.reshape(batch_size, seq_len)
    
    def set_prototypes(self, prototype_H, prototype_V):
        """Set prototype parameters from K-means clustering results"""
        self.prototype_H.data = torch.from_numpy(prototype_H).float().to(self.prototype_H.device)
        self.prototype_V.data = torch.from_numpy(prototype_V).float().to(self.prototype_V.device)
        self.compute_inter_cluster_neighbors()
        
    def save_prototypes(self, path):
        """Save prototypes to file"""
        prototype_dict = {
            'prototype_H': self.prototype_H.cpu().numpy(),
            'prototype_V': self.prototype_V.cpu().numpy(),
            'inter_cluster_neighbors': self.inter_cluster_neighbors.cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(prototype_dict, f)
    
    def load_prototypes(self, path):
        """Load prototypes from file"""
        with open(path, 'rb') as f:
            prototype_dict = pickle.load(f)
        self.prototype_V.data = torch.from_numpy(prototype_dict['prototype_V']).float().to(self.prototype_V.device)
        if 'prototype_H' in prototype_dict:
            self.prototype_H.data = torch.from_numpy(prototype_dict['prototype_H']).float().to(self.prototype_H.device)
        # Compute inter-cluster neighbors from prototype_V similarity
        self.compute_inter_cluster_neighbors()


class GPro_LLM(nn.Module):
    """
    GPro-LLM: Two-Stage Next POI Prediction Framework based on Global Prototype-Enhanced LLM
    
    Stage 1 (pretrain): Train LLM backbone, extract V representations, cluster into prototypes
    Stage 2 (finetune): Build hypergraph on individual V + prototype V, use HGNN convolution
                         to aggregate group information, produce enhanced representations
    """
    def __init__(self, config):
        super(GPro_LLM, self).__init__()
        
        self.config = config
        self.device = config['device']
        self.stage = config.get('stage', 'pretrain')
        self.freeze_frontend = config.get('freeze_frontend', False)
        
        # Basic parameters
        self.loc_size = config['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = config['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.user_size = config['uid_size']
        self.user_emb_size = config['user_emb_size']
        self.hidden_size = config['hidden_size']
        self.category_size = config['category_size']
        self.geohash_size = config['geohash_size']
        self.learnable_param_size = config['learnable_param_size']
        self.model_class = config['model_class']
        self.downstream = config['downstream']
        
        # GPro-LLM specific parameters
        self.num_prototypes = config.get('num_prototypes', 8)
        self.num_hgnn_layers = config.get('num_hgnn_layers', 2)
        self.hgnn_dropout = config.get('hgnn_dropout', 0.5)
        self.gate_hidden_size = config.get('gate_hidden_size', 128)
        self.prototype_path = config.get('prototype_path', 'prototypes.pkl')
        
        # ================== Frontend Modules (Shared in both stages) ==================
        # Embedding layers
        self.emb_loc = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.loc_emb_size)
        self.emb_tim = nn.Embedding(num_embeddings=self.tim_size, embedding_dim=self.tim_emb_size)
        self.emb_user = nn.Embedding(num_embeddings=self.user_size, embedding_dim=self.user_emb_size)
        
        # Category and geohash dense layers
        self.category_dense = nn.Linear(768, self.category_size)
        self.geohash_dense = nn.Linear(12, self.geohash_size)
        
        # Spatial encoder (LLM-based)
        self.spatial_encoder = LLMModel(
            model_path="/data/ZhangXinyue/MobilityLLM/params/" + self.model_class,
            model_class=self.model_class,
            loc_size=self.loc_size,
            learnable_param_size=self.learnable_param_size,
            device=self.device
        )
        
        # ================== Hypergraph Prototype Module (Stage 2) ==================
        self.spatial_output_size = 256  # LLMModel default output_size
        
        if self.stage == 'finetune':
            self.prototype_module = HypergraphPrototypeModule(
                hidden_size=self.spatial_output_size,
                num_prototypes=self.num_prototypes,
                num_hgnn_layers=self.num_hgnn_layers,
                hgnn_dropout=self.hgnn_dropout,
                gate_hidden_size=self.gate_hidden_size
            )
            # Load pre-computed prototypes
            if os.path.exists(self.prototype_path):
                self.prototype_module.load_prototypes(self.prototype_path)
                print(f"Loaded prototypes from {self.prototype_path}")
        
        # ================== Backend Modules (Task-specific heads) ==================
        if self.downstream == 'POI':
            self.projection = nn.Sequential(
                nn.Linear(self.spatial_output_size + self.user_emb_size, self.spatial_output_size + self.user_emb_size),
                nn.ReLU()
            )
            self.dense = nn.Linear(
                in_features=self.spatial_output_size + self.user_emb_size, 
                out_features=self.loc_size
            )
        elif self.downstream == 'TUL':
            self.dense = nn.Linear(in_features=self.spatial_output_size, out_features=self.user_size)
            self.projection = nn.Sequential(nn.Linear(self.spatial_output_size, self.spatial_output_size), nn.ReLU())
        elif self.downstream == 'TPP':
            self.projection = nn.Sequential(
                nn.Linear(self.spatial_output_size + self.user_emb_size, self.spatial_output_size + self.user_emb_size),
                nn.ReLU()
            )
            self.dense = nn.Sequential(
                nn.Linear(self.hidden_size * 2 + self.user_emb_size, (self.hidden_size * 2 + self.user_emb_size) // 4),
                nn.LeakyReLU(),
                nn.Linear((self.hidden_size * 2 + self.user_emb_size) // 4, (self.hidden_size * 2 + self.user_emb_size) // 16),
                nn.LeakyReLU(),
                nn.Linear((self.hidden_size * 2 + self.user_emb_size) // 16, 1),
            )
        
        # Initialize weights
        self.apply(self._init_weight)
        
        # Freeze frontend if in finetune stage
        if self.stage == 'finetune' and self.freeze_frontend:
            self._freeze_frontend_modules()
    
    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    def _freeze_frontend_modules(self):
        """Freeze all frontend modules (embeddings, spatial encoder)"""
        print("Freezing frontend modules...")
        
        for param in self.emb_loc.parameters():
            param.requires_grad = False
        for param in self.emb_tim.parameters():
            param.requires_grad = False
        for param in self.emb_user.parameters():
            param.requires_grad = False
        for param in self.category_dense.parameters():
            param.requires_grad = False
        for param in self.geohash_dense.parameters():
            param.requires_grad = False
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False
            
        print("Frontend modules frozen.")
    
    def spatial_encode(self, x, time, category, geohash_, all_len, cur_len, batch_size, downstream='POI'):
        """
        Encode spatial features using LLM
        """
        spatial_out = self.spatial_encoder(
            x, 
            torch.tensor(all_len).to(self.device), 
            time, 
            category, 
            geohash_
        )
        
        if downstream == 'POI':
            final_out = spatial_out[0, (all_len[0] - cur_len[0]): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([
                    final_out, 
                    spatial_out[i, (all_len[i] - cur_len[i]): all_len[i], :]
                ], dim=0)
        elif downstream == 'TPP':
            if all_len[0] == cur_len[0]:
                left = all_len[0] - cur_len[0]
                right = all_len[0]
            else:
                left = all_len[0] - cur_len[0] - 1
                right = all_len[0] - 1
            final_out = spatial_out[0, left: right, :]
            for i in range(1, batch_size):
                if all_len[i] == cur_len[i]:
                    left = all_len[i] - cur_len[i]
                    right = all_len[i]
                else:
                    left = all_len[i] - cur_len[i] - 1
                    right = all_len[i] - 1
                final_out = torch.cat([final_out, spatial_out[i, left: right, :]], dim=0)
        elif downstream == 'TUL':
            final_out = spatial_out[0, (all_len[0] - 1): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([
                    final_out, 
                    torch.mean(spatial_out[i, : all_len[i], :], dim=0, keepdim=True)
                ], dim=0)
        
        return final_out
    
    def extract_representations(self, batch):
        """
        Extract H (intention) and V (preference) representations for prototype generation
        Used in Stage 1 to collect representations for K-means clustering
        """
        loc = batch.X_all_loc
        tim = batch.X_all_tim
        user = batch.X_users
        geohash_ = batch.X_all_geohash
        cur_len = batch.target_lengths
        all_len = batch.X_lengths
        loc_cat = batch.X_all_loc_category
        
        batch_size = loc.shape[0]
        
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)
        geohash_ = self.geohash_dense(geohash_)
        
        x = loc.to(self.device)
        time = tim.float().to(self.device)
        
        all_tau = [torch.cat((batch.X_tau[0, :all_len[0] - cur_len[0]], batch.Y_tau[0, :cur_len[0]]), dim=-1)]
        for i in range(1, batch_size):
            cur_tau = torch.cat((batch.X_tau[i, :all_len[i] - cur_len[i]], batch.Y_tau[i, :cur_len[i]]), dim=-1)
            all_tau.append(cur_tau)
        all_tau = pad_sequence(all_tau, batch_first=False).to(self.device)
        
        H_full = self.spatial_encode(x, all_tau.transpose(0,1), loc_cat, geohash_, all_len, cur_len, batch_size, downstream='POI')
        V_full = H_full  # V = H in current implementation
        
        return H_full, V_full, user_emb, cur_len
    
    def forward(self, batch, mode='test', downstream='POI'):
        """
        Forward pass for GPro-LLM
        """
        loc = batch.X_all_loc
        tim = batch.X_all_tim
        user = batch.X_users
        geohash_ = batch.X_all_geohash
        cur_len = batch.target_lengths
        all_len = batch.X_lengths
        loc_cat = batch.X_all_loc_category
        
        batch_size = loc.shape[0]
        
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)
        geohash_ = self.geohash_dense(geohash_)
        
        x = loc.to(self.device)
        time = tim.float().to(self.device)
        
        all_tau = [torch.cat((batch.X_tau[0, :all_len[0] - cur_len[0]], batch.Y_tau[0, :cur_len[0]]), dim=-1)]
        for i in range(1, batch_size):
            cur_tau = torch.cat((batch.X_tau[i, :all_len[i] - cur_len[i]], batch.Y_tau[i, :cur_len[i]]), dim=-1)
            all_tau.append(cur_tau)
        all_tau = pad_sequence(all_tau, batch_first=False).to(self.device)
        
        # Get intention representation H from spatial encoder
        H = self.spatial_encode(x, all_tau.transpose(0,1), loc_cat, geohash_, all_len, cur_len, batch_size, downstream=downstream)
        
        # Expand user embeddings to match sequence length
        all_user_emb = user_emb[0].unsqueeze(dim=0).repeat(cur_len[0], 1)
        for i in range(1, batch_size):
            all_user_emb = torch.cat([
                all_user_emb, 
                user_emb[i].unsqueeze(dim=0).repeat(cur_len[i], 1)
            ], dim=0)
        
        # Stage 2: Apply Hypergraph Prototype Module (with triple-gate fusion)
        if self.stage == 'finetune' and hasattr(self, 'prototype_module'):
            # Reshape H to [batch, seq_len, hidden] format for hypergraph module
            V_reshaped = []
            start_idx = 0
            for i in range(batch_size):
                end_idx = start_idx + cur_len[i]
                V_reshaped.append(H[start_idx:end_idx])  # V = H currently
                start_idx = end_idx
            
            # Pad to same length for batch processing
            V_padded = pad_sequence(V_reshaped, batch_first=True)  # [batch, max_len, hidden]
            
            # Create validity mask for non-padded positions
            max_len = V_padded.shape[1]
            mask = torch.zeros(batch_size, max_len, device=self.device)
            for i in range(batch_size):
                mask[i, :cur_len[i]] = 1.0
            
            # Apply hypergraph prototype module (returns V_fused via triple-gate fusion)
            V_fused, gate_weights, cluster_assignments = self.prototype_module(V_padded, mask=mask)
            
            # Reshape back to flat format (only valid positions)
            V_fused_list = []
            for i in range(batch_size):
                V_fused_list.append(V_fused[i, :cur_len[i]])
            V_fused_flat = torch.cat(V_fused_list, dim=0)
            
            # Use fused representation for prediction
            representation = torch.cat([V_fused_flat, all_user_emb], dim=1)
        else:
            # Stage 1: Use original H representation
            representation = torch.cat([H, all_user_emb], dim=1)
        
        # Prediction head
        if downstream == 'POI':
            prediction_out = self.projection(representation)
            dense = self.dense(prediction_out)
            pred = nn.LogSoftmax(dim=1)(dense)
        elif downstream == 'TUL':
            dense = self.dense(H)
            pred = nn.LogSoftmax(dim=1)(dense)
        elif downstream == 'TPP':
            prediction_out = self.projection(representation)
            pred = self.dense(prediction_out)
        
        # Compute loss
        criterion = nn.NLLLoss().to(self.device)
        if downstream == 'POI':
            loss = criterion(pred, batch.Y_location)
            _, top_k_pred = torch.topk(pred, k=self.loc_size)
        elif downstream == 'TUL':
            loss = criterion(pred, batch.X_users)
            _, top_k_pred = torch.topk(pred, k=self.user_size)
        elif downstream == 'TPP':
            criterion1 = nn.L1Loss().to(self.device)
            loss = criterion1(pred.squeeze(), self.truth_Y_tau if hasattr(self, 'truth_Y_tau') else batch.Y_tau)
            top_k_pred = pred
        
        return loss, top_k_pred, None


def extract_all_representations(model, dataloader, device, save_path='representations.pkl'):
    """
    Extract H and V representations from all training samples for K-means clustering
    """
    model.eval()
    all_H = []
    all_V = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representations"):
            batch = batch.to(device)
            H, V, _, _ = model.extract_representations(batch)
            all_H.append(H.cpu().numpy())
            all_V.append(V.cpu().numpy())
    
    all_H = np.concatenate(all_H, axis=0)
    all_V = np.concatenate(all_V, axis=0)
    
    rep_dict = {'H': all_H, 'V': all_V}
    with open(save_path, 'wb') as f:
        pickle.dump(rep_dict, f)
    
    print(f"Saved representations to {save_path}")
    print(f"H shape: {all_H.shape}, V shape: {all_V.shape}")
    
    return all_H, all_V


def generate_global_prototypes(H_path, num_prototypes=8, save_path='prototypes.pkl'):
    """
    Generate global prototypes using K-means clustering on V representations,
    and compute inter-cluster neighbor structure for hypergraph construction.
    """
    with open(H_path, 'rb') as f:
        rep_dict = pickle.load(f)
    H = rep_dict['H']
    V = rep_dict['V']
    
    print(f"Clustering {V.shape[0]} samples into {num_prototypes} prototypes...")
    
    # K-means clustering on V (preference space)
    kmeans_V = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10)
    cluster_labels = kmeans_V.fit_predict(V)
    
    # Cluster centers for V (used as prototype nodes in hypergraph)
    prototype_V = kmeans_V.cluster_centers_
    
    # Compute prototype_H as mean of H in each cluster (for reference)
    prototype_H = np.zeros((num_prototypes, H.shape[1]))
    for i in range(num_prototypes):
        mask = cluster_labels == i
        if mask.sum() > 0:
            prototype_H[i] = H[mask].mean(axis=0)
        else:
            prototype_H[i] = H.mean(axis=0)
    
    # Compute inter-cluster neighbors: top-2 most similar prototypes (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    proto_sim = cosine_similarity(prototype_V)  # [K, K]
    np.fill_diagonal(proto_sim, -np.inf)
    inter_cluster_neighbors = np.argsort(proto_sim, axis=1)[:, -2:]  # [K, 2] top-2
    
    print(f"Inter-cluster neighbors: {inter_cluster_neighbors.tolist()}")
    
    # Save prototypes with inter-cluster structure
    prototype_dict = {
        'prototype_H': prototype_H,
        'prototype_V': prototype_V,
        'cluster_labels': cluster_labels,
        'inter_cluster_neighbors': inter_cluster_neighbors
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(prototype_dict, f)
    
    print(f"Saved prototypes to {save_path}")
    print(f"Prototype H shape: {prototype_H.shape}")
    print(f"Prototype V shape: {prototype_V.shape}")
    
    return prototype_H, prototype_V
