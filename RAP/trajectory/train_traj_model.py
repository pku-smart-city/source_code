# train_traj_model.py
# -*- coding: utf-8 -*-

import os
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from traj_pred_model import TrajModelConfig, SlotTrajPredictor
from traj_dataset import TrajNPZDataset


def masked_mse(pred, target, slot_valid, future_valid):
    """
    pred/target: [B,3,P,2]
    slot_valid: [B,3]
    future_valid: [B,3,P]
    """
    mask = slot_valid[:, :, None, None].to(pred.device) * future_valid[:, :, :, None].to(pred.device)  # [B,3,P,1]
    diff2 = (pred - target) ** 2  # [B,3,P,2]
    diff2 = diff2 * mask          # [B,3,P,2]
    denom = mask.sum() * pred.shape[-1] 
    denom = denom.clamp(min=1.0)
    return diff2.sum() / denom


def masked_mse_per_step(pred, target, slot_valid, future_valid):
    device = pred.device
    B, S, P, D = pred.shape  # D=2
    assert D == 2

    # base mask: [B,3,P] bool/float
    base_mask = slot_valid[:, :, None].to(device) * future_valid.to(device)  # [B,3,P]
    base_mask = base_mask.float()

    diff2 = (pred - target) ** 2  # [B,3,P,2]

    per_p = []
    for p in range(P):
        # mask_p: [B,3,1] -> broadcast to [B,3,2]
        mask_p = base_mask[:, :, p].unsqueeze(-1)  # [B,3,1]
        diff2_p = diff2[:, :, p, :] * mask_p       # [B,3,2]

        denom = mask_p.sum() * D
        denom = denom.clamp(min=1.0)
        mse_p = diff2_p.sum() / denom
        per_p.append(mse_p)

    return torch.stack(per_p, dim=0)  # [P]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="traj_dataset.npz")
    parser.add_argument("--out", type=str, default="traj_model.pt")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--gcn_hidden", type=int, default=256)
    parser.add_argument("--gru_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    # cosine lr
    parser.add_argument("--eta_min", type=float, default=1e-6, help="")
    parser.add_argument("--tmax", type=int, default=0, help="")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrajNPZDataset(args.data)

    n_total = len(ds)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    sample = ds[0]
    P = sample["target"].shape[1]  # [3,P,2] -> P
    cfg = TrajModelConfig(
        feat_dim=5,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        num_gcn_layers=2,
        num_pred_steps=P,
        num_slots=3,
        dropout=args.dropout
    )

    model = SlotTrajPredictor(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    T_max = args.tmax if args.tmax > 0 else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=args.eta_min)

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        # ===================== train =====================
        model.train()
        loss_sum = 0.0
        step_loss_sum = torch.zeros(P, device=device)  
        count_sum = 0

        for batch in train_loader:
            hist = batch["hist_feats"].to(device)   # [B,T,V,5]
            adj = batch["adj"].to(device)           # [B,V,V]
            slot = batch["slot_idx"].to(device)     # [B,3]
            tgt = batch["target"].to(device)        # [B,3,P,2]
            valid = batch["slot_valid"].to(device)  # [B,3]
            fv = batch["future_valid"].to(device)   # [B,3,P]

            pred = model(hist, adj, slot)           # [B,3,P,2]
            loss = masked_mse(pred, tgt, valid, fv)

            with torch.no_grad():
                step_mse = masked_mse_per_step(pred, tgt, valid, fv)  # [P]

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = hist.shape[0]
            loss_sum += float(loss.item()) * bs
            step_loss_sum += step_mse * bs
            count_sum += bs

        train_loss = loss_sum / max(1, count_sum)
        train_step_loss = (step_loss_sum / max(1, count_sum)).detach().cpu().numpy().tolist()

        # ===================== val =====================
        model.eval()
        val_sum = 0.0
        val_step_sum = torch.zeros(P, device=device)
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                hist = batch["hist_feats"].to(device)
                adj = batch["adj"].to(device)
                slot = batch["slot_idx"].to(device)
                tgt = batch["target"].to(device)
                valid = batch["slot_valid"].to(device)
                fv = batch["future_valid"].to(device)

                pred = model(hist, adj, slot)
                loss = masked_mse(pred, tgt, valid, fv)
                step_mse = masked_mse_per_step(pred, tgt, valid, fv)

                bs = hist.shape[0]
                val_sum += float(loss.item()) * bs
                val_step_sum += step_mse * bs
                val_count += bs

        val_loss = val_sum / max(1, val_count)
        val_step_loss = (val_step_sum / max(1, val_count)).detach().cpu().numpy().tolist()

        step_str_train = " ".join([f"mse@p{p}={train_step_loss[p]:.3f}" for p in range(P)])
        step_str_val = " ".join([f"mse@p{p}={val_step_loss[p]:.3f}" for p in range(P)])

        print(f"[train] epoch {ep}/{args.epochs} "
              f"train_loss={train_loss:.6f} ({step_str_train}) "
              f"val_loss={val_loss:.6f} ({step_str_val})")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "cfg": asdict(cfg),
                "state_dict": model.state_dict(),
            }, args.out)
            print(f"[train] saved best checkpoint to {args.out} (val={best_val:.6f})")

        scheduler.step()
        cur_lr = opt.param_groups[0]["lr"]
        print(f"[train] lr={cur_lr:.6e}")

    print("[train] done.")


if __name__ == "__main__":
    main()
