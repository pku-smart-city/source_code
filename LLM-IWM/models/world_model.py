import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

# =========================
# Feature Module: 连续时空物理量编码器
# =========================
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.W_t = nn.Linear(1, dim)
        self.W_s = nn.Linear(1, dim)

    def forward(self, dt, ds):
        t_enc = self.W_t(torch.log1p(dt).unsqueeze(-1))
        s_enc = self.W_s(torch.log1p(ds).unsqueeze(-1))
        return torch.cat([t_enc, s_enc], dim=-1)

# =========================
# Spatial Module: 自适应门控全局空间超图
# =========================
class GatedGlobalSpatialHypergraph(nn.Module):
    def __init__(self, feature_dim, incidence_matrix):
        super().__init__()
        self.register_buffer('H', incidence_matrix.float())
        D_v = self.H.sum(dim=1).clamp(min=1e-6)
        D_e = self.H.sum(dim=0).clamp(min=1e-6)

        self.register_buffer('inv_sqrt_D_v', torch.pow(D_v, -0.5))
        self.register_buffer('inv_D_e', 1.0 / D_e)

        self.theta = nn.Linear(feature_dim, feature_dim)
        self.act = nn.GELU()

        self.gate_proj = nn.Linear(feature_dim * 2, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, node_features):
        node_features_norm = node_features * self.inv_sqrt_D_v.unsqueeze(1)
        edge_feat = torch.matmul(self.H.T, node_features_norm) * self.inv_D_e.unsqueeze(1)
        node_feat_updated = torch.matmul(self.H, edge_feat) * self.inv_sqrt_D_v.unsqueeze(1)

        hgnn_out = self.act(self.theta(node_feat_updated))

        gate = torch.sigmoid(self.gate_proj(torch.cat([node_features, hgnn_out], dim=-1)))
        fused_features = gate * hgnn_out + (1 - gate) * node_features
        return self.layer_norm(fused_features)

# =========================
# World Model: Spatial-HGNN + Temporal-BiGRU
# =========================
class HSTWorldModel(nn.Module):
    def __init__(self, num_users, num_pois, incidence_matrix, intent_dim=8, hidden_dim=128):
        super().__init__()
        self.poi_dim = 64

        self.user_embedding = nn.Embedding(num_users + 1, 32, padding_idx=num_users)
        self.weekday_embedding = nn.Embedding(8, 8, padding_idx=7)
        self.base_poi_embedding = nn.Embedding(num_pois + 1, self.poi_dim, padding_idx=num_pois)

        self.global_spatial_hgnn = GatedGlobalSpatialHypergraph(self.poi_dim, incidence_matrix)
        self.st_encoder = SpatioTemporalEncoder(dim=16)

        self.time_embedding = nn.Embedding(25, 16, padding_idx=24)
        self.intent_embedding = nn.Embedding(intent_dim + 1, 16, padding_idx=intent_dim)

        self.feature_dim = self.poi_dim + 16 + 8 + 16 + 2 + 32 + 32

        self.fine_gru = nn.GRU(self.feature_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.coarse_gru = nn.GRU(self.feature_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

        self.gru_aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.history_to_intent = nn.Linear(hidden_dim, intent_dim)
        self.action_embedding = nn.Embedding(intent_dim + 1, 32, padding_idx=intent_dim)
        self.dynamics_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.poi_proj = nn.Linear(hidden_dim, self.poi_dim)
        self.next_loc_head = nn.Linear(hidden_dim, 2)

    def forward(self, user_id, poi_seq, time_seq, weekday_seq, intent_seq, loc_seq, dt_seq, ds_seq, action_intent):
        u_emb = self.user_embedding(user_id).unsqueeze(1).expand(-1, poi_seq.size(1), -1)

        global_poi_embs = self.global_spatial_hgnn(self.base_poi_embedding.weight[:-1])
        padding_vector = torch.zeros(1, self.poi_dim, device=global_poi_embs.device)
        full_poi_embs = torch.cat([global_poi_embs, padding_vector], dim=0)

        seq_poi_embs = F.embedding(poi_seq, full_poi_embs)
        st_physics_emb = self.st_encoder(dt_seq, ds_seq)

        x = torch.cat([
            u_emb,
            seq_poi_embs,
            self.time_embedding(time_seq),
            self.weekday_embedding(weekday_seq),
            self.intent_embedding(intent_seq),
            loc_seq,
            st_physics_emb
        ], dim=-1)

        _, h1 = self.fine_gru(x[:, -3:])
        _, h2 = self.coarse_gru(x)
        h_gru_raw = torch.cat([h1[0], h1[1], h2[0], h2[1]], dim=-1)
        h_gru = self.gru_aggregation(h_gru_raw)

        pred_int = self.history_to_intent(h_gru)
        a = self.action_embedding(action_intent)
        world = self.dynamics_fusion(torch.cat([h_gru, a], dim=-1))

        pred_loc = self.next_loc_head(world)
        poi_query = self.poi_proj(world)

        pred_poi = torch.matmul(poi_query, full_poi_embs[:-1].T)
        return pred_int, pred_loc, pred_poi

class TrajDataset(data_utils.Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]