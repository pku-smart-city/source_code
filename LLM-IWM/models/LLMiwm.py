import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import os
import json
import math
import re
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

from models.utils import (
    set_seed, haversine_distance, map_category_to_intent,
    parse_hour, parse_weekday, parse_timestamp,
    INTENT_CLASSES, INTENT_TO_ID
)
from models.llm_module import huggingfaceAPIcall
from models.world_model import HSTWorldModel, TrajDataset

# =========================
# 主模型: LLM-IWM 框架
# =========================
class LLMiwm:
    def __init__(self, epochs=50, batch_size=128, lr=1e-3, patience=10, seed=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seq_len = 5
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self.seed = set_seed(seed)
        print(f"\n[Info] 当前实验使用的随机种子 (Seed): {self.seed}")

        self.save_dir = f"./output/run_{self.run_id}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.llm_cache = {}

    def sequence_to_tensors(self, seq, target_intent=None):
        poi_seq, time_seq, weekday_seq, intent_seq, loc_seq = [], [], [], [], []
        dt_seq, ds_seq = [], []

        pad_len = max(0, self.seq_len - len(seq))
        for _ in range(pad_len):
            poi_seq.append(len(self.poiInfos))
            time_seq.append(24)
            weekday_seq.append(7)
            intent_seq.append(len(INTENT_CLASSES))
            loc_seq.append([0.0, 0.0])
            dt_seq.append(0.0)
            ds_seq.append(0.0)

        last_lat, last_lon, last_time = 0.0, 0.0, 0.0
        if len(seq) > 0:
            last_p, last_ts = seq[-1]
            last_lat = float(self.poiInfos[last_p]["latitude"])
            last_lon = float(self.poiInfos[last_p]["longitude"])
            last_time = parse_timestamp(last_ts)

        for p, ts in seq[-self.seq_len:]:
            poi_seq.append(self.poi2id.get(p, 0))
            time_seq.append(parse_hour(ts))
            weekday_seq.append(parse_weekday(ts))

            intent = self.poiInfos[p]["intent"]
            intent_seq.append(INTENT_TO_ID[intent])

            lat = float(self.poiInfos[p]["latitude"])
            lon = float(self.poiInfos[p]["longitude"])
            norm_lat = (lat - self.lat_mean) / self.lat_std
            norm_lon = (lon - self.lon_mean) / self.lon_std
            loc_seq.append([norm_lat, norm_lon])

            curr_time = parse_timestamp(ts)
            dt_seq.append(abs(last_time - curr_time) / 3600.0)
            ds_seq.append(haversine_distance(lat, lon, last_lat, last_lon))

        if target_intent is None:
            target_intent = intent_seq[-1]

        return (
            torch.tensor(poi_seq, dtype=torch.long),
            torch.tensor(time_seq, dtype=torch.long),
            torch.tensor(weekday_seq, dtype=torch.long),
            torch.tensor(intent_seq, dtype=torch.long),
            torch.tensor(loc_seq, dtype=torch.float),
            torch.tensor(dt_seq, dtype=torch.float),
            torch.tensor(ds_seq, dtype=torch.float),
            torch.tensor(target_intent, dtype=torch.long)
        )

    def train_world_model(self, val_recents=None, val_targets=None):
        self.world_model = HSTWorldModel(
            num_users=len(self.user2id),
            num_pois=len(self.poiInfos),
            incidence_matrix=self.incidence_matrix,
            intent_dim=len(INTENT_CLASSES)
        ).to(self.device)

        optimizer = optim.AdamW(self.world_model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        criterion_poi = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion_int = nn.CrossEntropyLoss()
        criterion_loc = nn.MSELoss()

        train_samples = []
        for u, hist in self.longs.items():
            tgt_user_id = self.user2id.get(u, len(self.user2id))
            if len(hist) > 2:
                for i in range(1, len(hist)):
                    seq = hist[max(0, i - self.seq_len):i]
                    tgt_p, tgt_ts = hist[i]

                    tgt_poi_id = self.poi2id.get(tgt_p, 0)
                    tgt_intent_id = INTENT_TO_ID[self.poiInfos[tgt_p]["intent"]]

                    lat = float(self.poiInfos[tgt_p]["latitude"])
                    lon = float(self.poiInfos[tgt_p]["longitude"])
                    norm_lat = (lat - self.lat_mean) / self.lat_std
                    norm_lon = (lon - self.lon_mean) / self.lon_std
                    tgt_loc = [norm_lat, norm_lon]

                    tensors = self.sequence_to_tensors(seq, target_intent=tgt_intent_id)

                    train_samples.append((
                        torch.tensor(tgt_user_id, dtype=torch.long),
                        *tensors,
                        torch.tensor(tgt_poi_id, dtype=torch.long),
                        torch.tensor(tgt_intent_id, dtype=torch.long),
                        torch.tensor(tgt_loc, dtype=torch.float)
                    ))

        if not train_samples: return
        train_loader = data_utils.DataLoader(TrajDataset(train_samples), batch_size=self.batch_size, shuffle=True)

        val_samples = []
        if val_recents is not None and val_targets is not None:
            for traj, gt in val_targets.items():
                u = self.traj2u.get(traj, "")
                tgt_user_id = self.user2id.get(u, len(self.user2id))
                seq = val_recents.get(traj, [])
                if not seq: continue
                tgt_p, tgt_ts = gt
                tgt_poi_id = self.poi2id.get(tgt_p, 0)
                tgt_intent_id = INTENT_TO_ID[self.poiInfos[tgt_p]["intent"]]

                lat = float(self.poiInfos[tgt_p]["latitude"])
                lon = float(self.poiInfos[tgt_p]["longitude"])
                norm_lat = (lat - self.lat_mean) / self.lat_std
                norm_lon = (lon - self.lon_mean) / self.lon_std
                tgt_loc = [norm_lat, norm_lon]

                tensors = self.sequence_to_tensors(seq, target_intent=tgt_intent_id)
                val_samples.append((
                    torch.tensor(tgt_user_id, dtype=torch.long),
                    *tensors,
                    torch.tensor(tgt_poi_id, dtype=torch.long),
                    torch.tensor(tgt_intent_id, dtype=torch.long),
                    torch.tensor(tgt_loc, dtype=torch.float)
                ))
            val_loader = data_utils.DataLoader(TrajDataset(val_samples), batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.world_model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                u_id, p_seq, t_seq, wd_seq, i_seq, l_seq, dt_seq, ds_seq, act_int, tgt_poi, tgt_int, tgt_loc = [
                    b.to(self.device) for b in batch]

                optimizer.zero_grad()
                pred_int, pred_loc, pred_poi = self.world_model(u_id, p_seq, t_seq, wd_seq, i_seq, l_seq, dt_seq,
                                                                ds_seq, act_int)

                loss_poi = criterion_poi(pred_poi, tgt_poi)
                loss_int = criterion_int(pred_int, tgt_int)
                loss_loc = criterion_loc(pred_loc, tgt_loc)
                loss = loss_poi + loss_int + 0.1 * loss_loc
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            if val_loader:
                self.world_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        u_id, p_seq, t_seq, wd_seq, i_seq, l_seq, dt_seq, ds_seq, act_int, tgt_poi, tgt_int, tgt_loc = [
                            b.to(self.device) for b in batch]
                        pred_int, pred_loc, pred_poi = self.world_model(u_id, p_seq, t_seq, wd_seq, i_seq, l_seq,
                                                                        dt_seq, ds_seq, act_int)
                        loss_poi = criterion_poi(pred_poi, tgt_poi)
                        loss_int = criterion_int(pred_int, tgt_int)
                        loss_loc = criterion_loc(pred_loc, tgt_loc)
                        val_loss += (loss_poi + loss_int + 0.1 * loss_loc).item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch + 1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = deepcopy(self.world_model.state_dict())
                    patience_counter = 0
                    weights_path = os.path.join(self.save_dir, "dual_stream_wm_weights.pth")
                    torch.save(best_model_state, weights_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        break
            else:
                best_model_state = deepcopy(self.world_model.state_dict())
                weights_path = os.path.join(self.save_dir, "dual_stream_wm_weights.pth")
                torch.save(best_model_state, weights_path)

        if best_model_state is not None: self.world_model.load_state_dict(best_model_state)

    def build_candidate_set(self, u, groundTruth, trajectory=None, K=1000):
        poiList = list(self.poiInfos.keys())
        gt = groundTruth[0]
        hist = []
        if trajectory and trajectory in self.recents: hist = self.recents[trajectory]
        if len(hist) == 0 and u in self.longs: hist = self.longs[u]
        if len(hist) == 0:
            neg = random.sample(poiList, min(K, len(poiList)))
            neg = [p for p in neg if p != gt]
            return neg[:K - 1] + [gt]
        lats = [float(self.poiInfos[p]["latitude"]) for p, _ in hist]
        lons = [float(self.poiInfos[p]["longitude"]) for p, _ in hist]
        center_lat, center_lon = np.mean(lats), np.mean(lons)
        dist_list = []
        for p in poiList:
            lat = float(self.poiInfos[p]["latitude"])
            lon = float(self.poiInfos[p]["longitude"])
            d = haversine_distance(lat, lon, center_lat, center_lon)
            dist_list.append((p, d))
        dist_list.sort(key=lambda x: x[1])
        k = min(K, len(dist_list))
        near, mid, far = dist_list[:k // 3], dist_list[k // 3:2 * k // 3], dist_list[2 * k // 3:]
        neg = []
        if near: neg += [p for p, _ in random.sample(near, min(len(near), k // 3))]
        if mid: neg += [p for p, _ in random.sample(mid, min(len(mid), k // 3))]
        if far: neg += [p for p, _ in random.sample(far, min(len(far), k // 3))]
        neg = [p for p in neg if p != gt]
        return neg[:k - 1] + [gt]

    def prepare_statistics(self):
        self.poi2id = {p: i for i, p in enumerate(self.poiInfos)}
        self.user2id = {u: i for i, u in enumerate(self.longs.keys())}

        coords = [(float(v["latitude"]), float(v["longitude"])) for v in self.poiInfos.values()]

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        self.lat_mean, self.lat_std = np.mean(lats), np.std(lats) + 1e-6
        self.lon_mean, self.lon_std = np.mean(lons), np.std(lons) + 1e-6

        self.min_lat, self.max_lat = min(lats), max(lats)
        self.min_lon, self.max_lon = min(lons), max(lons)
        self.global_max_dist = max(haversine_distance(a[0], a[1], b[0], b[1]) for a in coords for b in coords)

        self.user_poi_pref = defaultdict(lambda: defaultdict(int))
        for u, hist in self.longs.items():
            hist = sorted(hist, key=lambda x: x[1])
            for p, _ in hist: self.user_poi_pref[u][p] += 1

        grid_resolution = 0.02
        grid2id = {}
        poi_grid_pairs = []

        for poi, info in self.poiInfos.items():
            lat, lon = float(info["latitude"]), float(info["longitude"])
            grid_lat_idx = int((lat - self.min_lat) / grid_resolution)
            grid_lon_idx = int((lon - self.min_lon) / grid_resolution)
            grid_hash = f"G_{grid_lat_idx}_{grid_lon_idx}"
            if grid_hash not in grid2id: grid2id[grid_hash] = len(grid2id)
            poi_grid_pairs.append((self.poi2id[poi], grid2id[grid_hash]))

        num_pois = len(self.poiInfos)
        num_grids = len(grid2id)
        H_tensor = torch.zeros((num_pois, num_grids), dtype=torch.float32)
        for pid, gid in poi_grid_pairs: H_tensor[pid, gid] = 1.0

        self.incidence_matrix = H_tensor
        self.total_checkins = sum(sum(pref.values()) for pref in self.user_poi_pref.values())

    def get_llm_intent_prior(self, rec):
        hist_texts = []
        for p, ts in rec[-self.seq_len:]:
            cat = self.poiInfos[p].get("category", "Unknown")
            h, wd = parse_hour(ts), parse_weekday(ts)
            hist_texts.append(f"- Visited a '{cat}' location around {h}:00 on weekday index {wd}.")

        prompt = (
            "You are an expert human mobility predictor. "
            "Based on a user's recent location history:\n"
            f"{chr(10).join(hist_texts)}\n"
            "What is their most likely NEXT intent? "
            f"Evaluate the probabilities for these categories: [{', '.join(INTENT_CLASSES)}]. "
            "You MUST output ONLY a valid JSON dictionary where keys are the categories and values are the probabilities (summing to 1.0)."
        )

        if prompt in self.llm_cache:
            response = self.llm_cache[prompt]
        else:
            response = huggingfaceAPIcall(prompt, max_tokens=200).strip()
            self.llm_cache[prompt] = response

        llm_intent_probs = np.ones(len(INTENT_CLASSES)) / len(INTENT_CLASSES)
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                prob_dict = json.loads(match.group(0))
                probs = [float(prob_dict.get(intent, prob_dict.get(intent.lower(), 0.0))) for intent in INTENT_CLASSES]
                if sum(probs) > 0: llm_intent_probs = np.array(probs) / sum(probs)
        except:
            pass
        return llm_intent_probs

    def run(self, data, datasetName, fs_str="full"):
        self.datasetName = datasetName
        self.fs_str = fs_str
        self.longs, self.val_recents, self.val_targets, self.recents, self.targets, self.poiInfos, self.traj2u = data
        self.longs = {str(k): v for k, v in self.longs.items()}
        self.traj2u = {k: str(v) for k, v in self.traj2u.items()}
        for poi in self.poiInfos: self.poiInfos[poi]["intent"] = map_category_to_intent(self.poiInfos[poi]["category"])

        self.prepare_statistics()
        self.train_world_model(val_recents=self.val_recents, val_targets=self.val_targets)

        hit1 = hit5 = hit10 = rr = 0
        if hasattr(self, 'world_model'): self.world_model.eval()

        self.predictions_log = {}

        for traj, gt in tqdm(self.targets.items()):
            u = self.traj2u[traj]
            candidateSet = self.build_candidate_set(u, gt, trajectory=traj)
            pred = self.runeach(traj, candidateSet, gt)

            self.predictions_log[traj] = {
                "ground_truth": gt[0],
                "predictions": pred[:100]
            }

            if gt[0] in pred:
                idx = pred.index(gt[0]) + 1
                if idx == 1: hit1 += 1
                if idx <= 5: hit5 += 1
                if idx <= 10: hit10 += 1
                rr += 1 / idx

        num = len(self.targets)
        hit1 /= num; hit5 /= num; hit10 /= num; mrr = rr / num

        print(f"\n===== 测试结果 ({self.datasetName}) =====")
        print(f"ACC@1: {hit1:.4f}, ACC@5: {hit5:.4f}, ACC@10: {hit10:.4f}, MRR: {mrr:.4f}")

        metrics_path = os.path.join(self.save_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {self.datasetName}\n")
            f.write(f"Random Seed: {self.seed}\n")
            f.write(f"ACC@1: {hit1:.4f}\nACC@5: {hit5:.4f}\nACC@10: {hit10:.4f}\nMRR: {mrr:.4f}\n")

        json_path = os.path.join(self.save_dir, "predictions_log.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.predictions_log, f, ensure_ascii=False, indent=2)

        print(f"Files successfully saved to {self.save_dir}")
        return (hit1, hit5, hit10, mrr)

    def runeach(self, trajectory, candidateSet, groundTruth):
        u = self.traj2u[trajectory]
        rec = self.recents.get(trajectory, [])
        if len(rec) == 0: return candidateSet

        u_idx = self.user2id.get(u, len(self.user2id))
        user_id_tensor = torch.tensor([u_idx], dtype=torch.long).to(self.device)
        llm_intent_probs = self.get_llm_intent_prior(rec)
        pred_poi_probs_by_intent = {}

        u_total = sum(self.user_poi_pref[u].values())
        u_unique = len(self.user_poi_pref[u])
        dynamic_temp = max(1.0, min(2.0, 1.0 + (u_unique / (u_total + 1e-6))))

        with torch.no_grad():
            for curr_intent_id in range(len(INTENT_CLASSES)):
                tensors = self.sequence_to_tensors(rec, target_intent=curr_intent_id)
                p_seq, t_seq, wd_seq, i_seq, l_seq, dt_seq, ds_seq, act_int = [x.unsqueeze(0).to(self.device) for x in tensors]
                pred_int, pred_loc, pred_poi = self.world_model(user_id_tensor, p_seq, t_seq, wd_seq, i_seq, l_seq, dt_seq, ds_seq, act_int)
                pred_poi_probs_by_intent[curr_intent_id] = F.softmax(pred_poi[0] / dynamic_temp, dim=-1).cpu().numpy()

            expected_pred_poi = np.zeros_like(pred_poi_probs_by_intent[0])
            for i in range(len(INTENT_CLASSES)):
                expected_pred_poi += llm_intent_probs[i] * pred_poi_probs_by_intent[i]

        wm_entropy = -np.sum(expected_pred_poi * np.log(expected_pred_poi + 1e-12))
        max_entropy = math.log(len(expected_pred_poi) + 1e-12)
        llm_entropy = -np.sum(llm_intent_probs * np.log(llm_intent_probs + 1e-12))
        max_llm_entropy = math.log(len(INTENT_CLASSES) + 1e-12)

        vocab = len(self.poiInfos)

        conf_hist = 1 - math.exp(-u_total / (u_unique + 1e-6))
        conf_wm = 1 - (wm_entropy / max_entropy)
        conf_llm = 1 - (llm_entropy / max_llm_entropy)
        conf_sum = conf_hist + conf_wm + conf_llm + 1e-6

        w_hist, w_wm, w_llm = conf_hist / conf_sum, conf_wm / conf_sum, conf_llm / conf_sum

        volume_ratio = max(1.0, self.total_checkins / 100000.0)
        alpha = volume_ratio / math.sqrt(u_total + vocab + 1e-6)
        min_explore_prob = 1.0 / vocab

        results = []
        for cand_poi in candidateSet:
            p_hist = (self.user_poi_pref[u].get(cand_poi, 0) + alpha) / (u_total + alpha * vocab)
            score_exploit = math.log(max(p_hist, 1e-12))
            intent_id = INTENT_TO_ID[self.poiInfos[cand_poi]["intent"]]
            p_explore = pred_poi_probs_by_intent[intent_id][self.poi2id.get(cand_poi, 0)]
            score_explore = math.log(max(p_explore, min_explore_prob))
            score_llm = math.log(max(llm_intent_probs[intent_id], 1e-12))

            final_score = w_hist * score_exploit + w_wm * score_explore + w_llm * score_llm
            results.append((cand_poi, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in results]