import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import glob
import random

# --- 1. Dataset definition (integrated) ---
class CARLA_Sequence_Dataset(Dataset):
    def __init__(self, file_list, stats_path, seq_len=30):
        self.seq_len = seq_len
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file missing: {stats_path}")
        self.stats = torch.load(stats_path)
        
        self.sequences = []
        total_frames = 0
        total_collisions = 0
        
        print(f"Loading {len(file_list)} files...")
        for f in file_list:
            try:
                df = pd.read_csv(f)
                if df.isnull().values.any(): df = df.dropna()
                if len(df) < self.seq_len + 1: continue

                # Normalize.
                for col, stat in self.stats.items():
                    if col in df.columns:
                        std_val = stat['std'] if stat['std'] > 1e-6 else 1.0
                        df[col] = (df[col] - stat['mean']) / std_val
                
                data_array = df.values.astype(np.float32)
                if data_array.shape[1] != 52: continue # 48 features + 3 actions + 1 collision

                self.sequences.append(data_array)
                total_collisions += (data_array[:, 51] > 0.5).sum()
                total_frames += len(data_array)
            except:
                continue
        
        self.indices = []
        for seq_idx, data in enumerate(self.sequences):
            num_windows = len(data) - self.seq_len
            for t in range(num_windows):
                self.indices.append((seq_idx, t))
        
        # Compute positive-sample weight.
        self.pos_weight = (total_frames - total_collisions) / (total_collisions + 1e-6)
        print(f"Dataset ready. Samples: {len(self.indices)} | PosWeight: {self.pos_weight:.2f}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx, start_t = self.indices[idx]
        data_clip = self.sequences[seq_idx][start_t : start_t + self.seq_len + 1]
        
        return {
            'ego': torch.from_numpy(data_clip[:-1, 0:10]),
            'road': torch.from_numpy(data_clip[:-1, 10:18]),
            'social': torch.from_numpy(data_clip[:-1, 18:48]),
            'action': torch.from_numpy(data_clip[:-1, 48:51]),
            'target_ego': torch.from_numpy(data_clip[1:, 0:10]),
            'target_social': torch.from_numpy(self._get_social_target(data_clip[1:])),
            'target_coll': torch.from_numpy(data_clip[1:, 51:52])
        }

    def _get_social_target(self, clip):
        targets = []
        for i in range(5):
            base = 18 + i * 6
            targets.append(clip[:, base : base+4])
        return np.concatenate(targets, axis=1)

# --- 2. Core loss functions ---
def kl_loss_fn(mean, std, free_bits=1.0):
    kl = -0.5 * torch.sum(1 + 2 * torch.log(std + 1e-6) - mean.pow(2) - std.pow(2), dim=-1)
    kl_loss = torch.max(kl, torch.full_like(kl, free_bits))
    return kl_loss.mean()

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

# --- 3. Main training function ---
def train_rssm():
    # --- Settings tuned for RTX 3080/3090 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 30         # Restored to 30
    batch_size = 64      # Fits 10-12GB VRAM
    lr = 3e-4            # Restored standard learning rate
    epochs = 50
    data_dir = "collected_data"
    stats_path = "meta_stats.pth"

    # Import model (assumed in models.py)
    from models import RSSMWorldModelV3

    # 1) Scan files and split.
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        print("Error: No data found!")
        return
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    
    train_ds = CARLA_Sequence_Dataset(all_files[:split_idx], stats_path, seq_len=seq_len)
    val_ds = CARLA_Sequence_Dataset(all_files[split_idx:], stats_path, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 2) Initialize model and optimizer.
    model = RSSMWorldModelV3(ego_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_score = float('inf')

    # 3) Training loop.
    for epoch in range(epochs):
        model.train()
        beta = 0.05 * min(1.0, epoch / 10.0) 
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items()}
            h, z = None, None
            
            optimizer.zero_grad()
            
            loss_ego, loss_soc, loss_kl = 0, 0, 0
            
            # Unroll over timesteps.
            for t in range(seq_len):
                pred_ego, pred_soc, _, h, z, (zm, zs) = model(
                    inputs['ego'][:, t], inputs['road'][:, t], 
                    inputs['social'][:, t], inputs['action'][:, t], h, z
                )
                
                loss_ego += F.smooth_l1_loss(pred_ego, inputs['target_ego'][:, t])
                loss_soc += F.smooth_l1_loss(pred_soc, inputs['target_social'][:, t])
                loss_kl += kl_loss_fn(zm, zs)
            
            total_loss = (loss_ego + loss_soc + beta * loss_kl) / seq_len
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'L': f"{total_loss.item():.3f}", 'Ego': f"{(loss_ego/seq_len).item():.3f}"})

        # --- 4. Validation (closed-loop rollout) ---
        model.eval()
        val_errors = []
        tp, fp, fn = 0, 0, 0
        burn_in = 5
        rollout_steps = 10

        print(f"Validating...")
        with torch.no_grad():
            for v_batch in val_loader:
                v_inputs = {k: v.to(device) for k, v in v_batch.items()}
                rh, rz = None, None
                
                # Collision metrics (open-loop).
                for t in range(seq_len):
                    _, _, vc, rh, rz, _ = model(
                        v_inputs['ego'][:, t], v_inputs['road'][:, t], 
                        v_inputs['social'][:, t], v_inputs['action'][:, t], rh, rz
                    )
                    preds = (torch.sigmoid(vc) > 0.5).float().squeeze(-1)
                    targs = v_inputs['target_coll'][:, t].squeeze(-1)
                    tp += ((preds == 1) & (targs == 1)).sum().item()
                    fp += ((preds == 1) & (targs == 0)).sum().item()
                    fn += ((preds == 0) & (targs == 1)).sum().item()

                # Rollout error (closed-loop).
                rh, rz = None, None
                curr_ego = v_inputs['ego'][:, 0]
                for t in range(burn_in):
                    _, _, _, rh, rz, _ = model(
                        v_inputs['ego'][:, t], v_inputs['road'][:, t], 
                        v_inputs['social'][:, t], v_inputs['action'][:, t], rh, rz
                    )
                    curr_ego = v_inputs['target_ego'][:, t]
                
                rollout_err = 0
                for t in range(burn_in, burn_in + rollout_steps):
                    pred_e, _, _, rh, rz, _ = model(
                        curr_ego, v_inputs['road'][:, t], 
                        v_inputs['social'][:, t], v_inputs['action'][:, t], rh, rz
                    )
                    rollout_err += F.l1_loss(pred_e, v_inputs['target_ego'][:, t]).item()
                    curr_ego = pred_e
                val_errors.append(rollout_err / rollout_steps)

        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        avg_rollout_err = np.mean(val_errors)
        current_score = avg_rollout_err
        
        print(f"Val: RolloutErr={avg_rollout_err:.4f} | Coll_F1={f1:.3f}")
        
        if current_score < best_score:
            best_score = current_score
            torch.save(model.state_dict(), "rssm_v3_best.pth")
            print("⭐ Best Model Saved.")

if __name__ == "__main__":
    train_rssm()
