import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x):
    """
    Compress value magnitude (DreamerV3 core technique).
    [Fix] Add 1e-6 to avoid NaN from log(0).
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0 + 1e-6)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=3):
        super().__init__()
        self.net = nn.ModuleList()
        curr = in_dim
        for _ in range(layers):
            self.net.append(nn.Sequential(
                nn.Linear(curr, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU()
            ))
            curr = hidden_dim
        self.final = nn.Linear(curr, out_dim)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return self.final(x)

class SocialAttentionV2(nn.Module):
    def __init__(self, feat_dim=6, embed_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 5, 6)
        B = x.size(0)
        h = F.elu(self.input_proj(x)) 
        q = self.query_token.expand(B, -1, -1) 
        attn_out, _ = self.mha(q, h, h)
        return self.norm(attn_out.squeeze(1))

class RSSMWorldModelV3(nn.Module):
    def __init__(self, ego_dim=10, map_dim=8, action_dim=3, latent_dim=32, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 1) Normalization layers (ego dim = 10)
        self.ego_norm = nn.LayerNorm(ego_dim)
        self.map_norm = nn.LayerNorm(map_dim)
        self.soc_norm = nn.LayerNorm(6)
        
        # 2) Encoders
        self.ego_enc = nn.Linear(ego_dim, 128)
        self.map_enc = nn.Linear(map_dim, 128)
        self.social_enc = SocialAttentionV2(feat_dim=6, embed_dim=128)
        
        # 3) RSSM core
        self.fusion = nn.Sequential(
            nn.Linear(128*3 + latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )
        self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.rnn_norm = nn.LayerNorm(hidden_dim)
        
        # 4) Posterior network
        self.z_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, latent_dim * 2)
        )
        
        # 5) Decoders
        decoder_in = hidden_dim + latent_dim
        self.ego_decoder = MLP(decoder_in, 256, ego_dim, layers=3)
        
        # Output 20 dims (5 vehicles * (dx, dy, dvx, dvy))
        self.social_decoder = MLP(decoder_in, 256, 20, layers=2)
        
        self.collision_head = MLP(decoder_in, 128, 1, layers=3)

    def reparameterize(self, mean, std):
        if self.training:
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, ego, road, social, action, prev_h=None, prev_z=None):
        B = ego.size(0)
        device = ego.device
        
        # Data guard: replace potential NaNs.
        ego = torch.nan_to_num(ego)
        road = torch.nan_to_num(road)
        social = torch.nan_to_num(social)
        
        # Input encoding
        ego = symlog(self.ego_norm(ego))
        road = self.map_norm(road)
        social = self.soc_norm(social.view(B, 5, 6))

        if prev_h is None: prev_h = torch.zeros(B, self.hidden_dim).to(device)
        if prev_z is None: prev_z = torch.zeros(B, self.latent_dim).to(device)

        e_ego = F.elu(self.ego_enc(ego))
        e_map = F.elu(self.map_enc(road))
        e_soc = self.social_enc(social)
        
        # RSSM Step
        combined = torch.cat([e_ego, e_map, e_soc, prev_z, action], dim=-1)
        fused = self.fusion(combined)
        
        h = self.rnn_norm(self.rnn_cell(fused, prev_h))
        # Clamp h to avoid exploding gradients.
        h = torch.clamp(h, -50, 50) 
        
        stats = self.z_net(h)
        z_mean, z_std_raw = torch.chunk(stats, 2, dim=-1)
        z_mean = torch.clamp(z_mean, -5, 5)
        z_std = 1.5 * torch.sigmoid(z_std_raw) + 0.1 
        
        z = self.reparameterize(z_mean, z_std)
        latent_state = torch.cat([h, z], dim=-1)
        
        # Decode
        coll_logits = self.collision_head(latent_state)
        # Clamp logits range.
        coll_logits = torch.clamp(coll_logits, -15, 15)
        
        return self.ego_decoder(latent_state), \
               self.social_decoder(latent_state), \
               coll_logits, \
               h, z, (z_mean, z_std)
