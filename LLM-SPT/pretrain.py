import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
import sys
import yaml
import itertools
import gc
from util import *
from scipy.sparse import issparse
import random
from model import ST_LLM, PromptNetwork
from ranger21 import Ranger
from datetime import datetime, timedelta
from prompt_generator_s import generate_prompt
from transformers import AutoTokenizer

# 设置CUDA内存分配配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for training")
parser.add_argument("--data", nargs="+", type=str, default=["CHIbike_drop"], help="Data paths for source cities")
parser.add_argument("--input_dim", type=int, default=3, help="Input dimension")
parser.add_argument("--channels", type=int, default=64, help="Number of features")
parser.add_argument("--num_nodes", type=int, default=266, help="Number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="Input length")
parser.add_argument("--output_len", type=int, default=12, help="Output length")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--print_every", type=int, default=50, help="Print interval")
parser.add_argument("--wdecay", type=float, default=0.0001, help="Weight decay rate")
parser.add_argument("--es_patience", type=int, default=100, help="Early stopping patience")
parser.add_argument("--prune_ratio", type=float, default=0.4, help="Pruning ratio for model")
parser.add_argument("--freeze_ratio", type=float, default=0.7, help="Freezing ratio for model")
args = parser.parse_args()


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


class trainer:
    def __init__(self, scaler, lrate, wdecay, config, input_dim, channels, num_nodes, input_len, output_len, device,
                 prune_ratio, freeze_ratio, save_dir, pruning_indices=None):  # 添加pruning_indices参数
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
        # 初始化模型时传入pruning_indices
        self.model = ST_LLM(device, input_dim, channels, num_nodes, input_len, output_len, prune_ratio, freeze_ratio,
                            keep_indices=pruning_indices)
        self.prompt_network = PromptNetwork(input_dim, channels, num_nodes, input_len, output_len)
        self.model = self.model.to(device=device, dtype=torch.bfloat16)
        self.prompt_network = self.prompt_network.to(device=device, dtype=torch.bfloat16)

        self.config = config
        self.regions_to_log = config.get('regions_to_log', [0])
        self.time_format = config.get('time_format', 'half an hour')
        self.log_path = os.path.join(save_dir, "prompt_log.txt")
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len

        # 创建并打开提示日志文件
        os.makedirs(save_dir, exist_ok=True)
        self.log_file = open(self.log_path, 'w')
        self.line_count = 0

        self.model_optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.prompt_optimizer = Ranger(self.prompt_network.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5

        self.valid_regions = [rid for rid in self.regions_to_log if rid < num_nodes]
        if len(self.valid_regions) != len(self.regions_to_log):
            print(f"Warning: Filtered valid region IDs are {self.valid_regions}")

        self.debug_log = open(os.path.join(save_dir, "debug_log.txt"), 'w')
        self.device = device
        self.save_dir = save_dir

    def train(self, input, real_val, metadata):
        history_input = input.permute(0, 2, 3, 1)
        history_data = self.scaler.inverse_transform(history_input)
        history_data = history_data.squeeze(-1) if history_data.size(-1) == 1 else history_data[..., 0]

        self.model.train()
        self.prompt_network.train()
        self.model_optimizer.zero_grad()
        self.prompt_optimizer.zero_grad()

        input = input.to(device=self.model.device, dtype=torch.bfloat16)
        prompted_input = self.prompt_network(input)

        initial_output = self.model(prompted_input)
        initial_preds = initial_output.squeeze(-1)
        initial_preds = self.scaler.inverse_transform(initial_preds.unsqueeze(-1)).squeeze(-1)
        initial_preds = initial_preds.permute(0, 2, 1)

        _, personalized_prompts, _ = generate_prompt(
            initial_preds.detach().cpu(),
            metadata.cpu(),
            self.valid_regions,
            history_data.cpu(),
            self.num_nodes,
            self.output_len,
            self.debug_log
        )

        prompt_tokens = self.tokenizer(
            personalized_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        output = self.model(prompted_input, prompt_tokens)
        preds = output.squeeze(-1)
        preds = self.scaler.inverse_transform(preds.unsqueeze(-1)).squeeze(-1)

        real = real_val[:, :self.output_len, :]
        real = real.permute(0, 2, 1)

        loss = self.loss(preds, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.prompt_network.parameters(), self.clip)
        self.model_optimizer.step()
        self.prompt_optimizer.step()

        mape = util.MAPE_torch(preds, real, 0.0).item()
        rmse = util.RMSE_torch(preds, real, 0.0).item()

        # 释放内存
        del history_input, history_data, initial_output, initial_preds, personalized_prompts, prompt_tokens
        # del output, preds, real, loss
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计

        return loss.item(), mape, rmse

    def eval(self, input, real_val, metadata):
        history_input = input.permute(0, 2, 3, 1)
        history_data = self.scaler.inverse_transform(history_input)
        history_data = history_data.squeeze(-1) if history_data.size(-1) == 1 else history_data[..., 0]

        self.model.eval()
        self.prompt_network.eval()
        input = input.to(device=self.model.device, dtype=torch.bfloat16)
        prompted_input = self.prompt_network(input)

        initial_output = self.model(prompted_input)
        initial_preds = initial_output.squeeze(-1)
        initial_preds = self.scaler.inverse_transform(initial_preds.unsqueeze(-1)).squeeze(-1)
        initial_preds = initial_preds.permute(0, 2, 1)

        # 生成提示
        _, personalized_prompts, _ = generate_prompt(
            initial_preds.detach().cpu(),
            metadata.cpu(),
            self.valid_regions,
            history_data.cpu(),
            self.num_nodes,
            self.output_len,
            self.debug_log
        )

        prompt_tokens = self.tokenizer(
            personalized_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        output = self.model(prompted_input, prompt_tokens)
        preds = output.squeeze(-1)
        preds = self.scaler.inverse_transform(preds.unsqueeze(-1)).squeeze(-1)

        real = real_val[:, :self.output_len, :]
        real = real.permute(0, 2, 1)

        loss = self.loss(preds, real, 0.0)
        mape = util.MAPE_torch(preds, real, 0.0).item()
        rmse = util.RMSE_torch(preds, real, 0.0).item()

        # 调用generate_prompts记录提示
        self.generate_prompts(input, real_val, metadata)

        # 释放内存
        del history_input, history_data, initial_output, initial_preds, personalized_prompts, prompt_tokens
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计

        return loss.item(), mape, rmse

    def generate_prompts(self, input, real_val, metadata):
        self.model.eval()
        self.prompt_network.eval()
        input = input.to(device=self.model.device, dtype=torch.bfloat16)
        metadata = metadata.clone().detach().to(torch.long).cpu()
        history_input = input.permute(0, 2, 3, 1)
        history_data = self.scaler.inverse_transform(history_input)
        history_data = history_data.squeeze(-1) if history_data.size(-1) == 1 else history_data[..., 0]

        prompted_input = self.prompt_network(input)

        initial_output = self.model(prompted_input)
        initial_preds = initial_output.squeeze(-1)
        initial_preds = self.scaler.inverse_transform(initial_preds.unsqueeze(-1)).squeeze(-1)
        initial_preds = initial_preds.permute(0, 2, 1)

        _, personalized_prompts, _ = generate_prompt(
            initial_preds.detach().cpu(),
            metadata.cpu(),
            self.valid_regions,
            history_data.cpu(),
            self.num_nodes,
            self.output_len,
            self.debug_log
        )
        prompt_tokens = self.tokenizer(
            personalized_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        output = self.model(prompted_input, prompt_tokens)
        preds = output.squeeze(-1)
        preds = self.scaler.inverse_transform(preds.unsqueeze(-1)).squeeze(-1)
        preds = preds.transpose(1, 2)

        real = real_val[:, :self.output_len, :]
        real = real.permute(0, 2, 1)

        if metadata is not None:
            input_prompts, personalized_prompts, output_values_list = generate_prompt(
                preds.detach().cpu(),
                metadata,
                self.valid_regions,
                history_data.cpu(),
                self.num_nodes,
                self.output_len,
                self.debug_log
            )

            for input_prompt, personalized_prompt, output_values in zip(input_prompts, personalized_prompts,
                                                                        output_values_list):
                if self.line_count < 2200:
                    self.log_file.write(input_prompt + "\n")
                    self.log_file.write(personalized_prompt + "\n")
                    self.log_file.write(output_values + "\n")
                    self.log_file.flush()
                    self.line_count += 1
                else:
                    break

        # 清理中间变量
        del input_prompts, output_values_list, history_input, history_data
        del initial_output, initial_preds, personalized_prompts, prompt_tokens
        torch.cuda.empty_cache()

    def save_model(self, path):
        """保存模型为两个文件，模仿原始代码的方式"""
        # 保存主模型
        torch.save(self.model.state_dict(), os.path.join(path, "best_model.pth"))
        # 保存提示网络
        torch.save(self.prompt_network.state_dict(), os.path.join(path, "best_prompt_network.pth"))
        # 保存剪枝索引
        if hasattr(self.model, 'get_pruning_indices'):
            pruning_indices = self.model.get_pruning_indices()
            torch.save(pruning_indices, os.path.join(path, "pruning_indices.pt"))

    def load_model(self, path, device):
        """加载模型权重，模仿原始代码的方式"""
        # 加载主模型
        model_path = os.path.join(path, "best_model.pth")
        if os.path.exists(model_path):
            # 设置weights_only=True以避免FutureWarning
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

        # 加载提示网络
        prompt_path = os.path.join(path, "best_prompt_network.pth")
        if os.path.exists(prompt_path):
            # 设置weights_only=True以避免FutureWarning
            self.prompt_network.load_state_dict(torch.load(prompt_path, map_location=device, weights_only=False))

        # 加载剪枝索引
        prune_path = os.path.join(path, "pruning_indices.pt")
        if os.path.exists(prune_path) and hasattr(self.model, 'set_pruning_indices'):
            pruning_indices = torch.load(prune_path, map_location=device, weights_only=False)
            self.model.set_pruning_indices(pruning_indices)

    def close_logs(self):
        """关闭所有打开的日志文件"""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        if self.debug_log and not self.debug_log.closed:
            self.debug_log.close()

    def __del__(self):
        """对象销毁时确保资源释放"""
        self.close_logs()
        # 将模型移到CPU并删除引用
        if hasattr(self, 'model') and self.model is not None:
            self.model.to('cpu')
            del self.model
        if hasattr(self, 'prompt_network') and self.prompt_network is not None:
            self.prompt_network.to('cpu')
            del self.prompt_network
        # 释放优化器和其他资源
        if hasattr(self, 'model_optimizer') and self.model_optimizer is not None:
            del self.model_optimizer
        if hasattr(self, 'prompt_optimizer') and self.prompt_optimizer is not None:
            del self.prompt_optimizer
        torch.cuda.empty_cache()
        gc.collect()


def main():
    # DAG-based multi-source pretraining
    config_path = os.path.join("config", "prompt_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['prompt']

    seed_it(6666)
    source_cities = args.data
    device = torch.device(args.device)

    # 创建基础保存目录
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    data_str = "_".join(source_cities)
    base_save = f'./logs/pretrain-{time_str}-{data_str}/'
    os.makedirs(base_save, exist_ok=True)

    # 保存命令行参数
    with open(os.path.join(base_save, "args.txt"), 'w') as f:
        f.write(f"Command: python {' '.join(sys.argv)}\n")
        f.write(f"Datasets: {', '.join(source_cities)}\n")

    # Group datasets by city
    city_datasets = {}
    for dataset in source_cities:
        if 'taxi' in dataset:
            city_name = dataset.split('taxi')[0]
            if city_name not in city_datasets:
                city_datasets[city_name] = {'taxi': [], 'bike': []}
            city_datasets[city_name]['taxi'].append(dataset)
        elif 'bike' in dataset:
            city_name = dataset.split('bike')[0]
            if city_name not in city_datasets:
                city_datasets[city_name] = {'taxi': [], 'bike': []}
            city_datasets[city_name]['bike'].append(dataset)

    # 初始化变量
    loss = float("inf")
    epochs_since_best_mae = 0
    his_loss = []
    val_time = []
    train_time = []
    result = []
    best_epoch = 0
    print(args)

    if not os.path.exists(base_save):
        os.makedirs(base_save)

    print("Starting DAG-based pretraining...")

    # Create city list and generate all permutations
    cities = list(city_datasets.keys())
    all_orders = list(itertools.permutations(cities))
    print(f"Found {len(all_orders)} possible city orders")

    # Initialize best model tracking
    best_valid_loss = float('inf')
    best_model_state = None
    best_prompt_state = None
    best_pruning_indices = None  # 跟踪最佳剪枝索引
    best_order = None
    best_order_str = ""
    best_dataloader = None

    # Process each city order permutation
    for order_idx, city_order in enumerate(all_orders):
        print(f"\n{'=' * 50}")
        print(f"Training order #{order_idx + 1}/{len(all_orders)}: {city_order}")
        print(f"{'=' * 50}")

        # Create save directory for this order
        order_str = "_".join(city_order)
        order_save = os.path.join(base_save, f"order_{order_idx}_{order_str}")
        os.makedirs(order_save, exist_ok=True)

        # Save order info
        with open(os.path.join(order_save, "order_info.txt"), 'w') as f:
            f.write(f"Order: {order_str}\n")
            f.write(f"City sequence: {' -> '.join(city_order)}\n")

        # 按顺序训练每个城市
        order_valid_loss = 0
        engine = None
        current_pruning_indices = None  # 当前顺序的剪枝索引

        for city_idx, city in enumerate(city_order):
            city_datasets_list = city_datasets[city]['taxi'] + city_datasets[city]['bike']
            city_data_paths = [f"{d}/" for d in city_datasets_list]

            print(f"\n{'=' * 30}")
            print(f"Training city {city} ({city_idx + 1}/{len(city_order)}) in order {order_str}")
            print(f"{'=' * 30}")

            # 创建城市保存目录
            city_save = os.path.join(order_save, f"city_{city}")
            os.makedirs(city_save, exist_ok=True)

            # 创建训练指标CSV文件
            train_csv_path = os.path.join(city_save, f"train_{city}.csv")
            if not os.path.exists(train_csv_path):
                with open(train_csv_path, 'w') as f:
                    f.write(
                        "epoch,train_loss,train_mape,train_rmse,valid_loss,valid_mape,valid_rmse\n")

            # 确定节点数量
            taxi_included = any('taxi' in d for d in city_datasets_list)
            bike_included = any('bike' in d for d in city_datasets_list)

            if taxi_included and bike_included:
                current_num_nodes = max(266, 250)
            elif taxi_included:
                current_num_nodes = 266
            elif bike_included:
                current_num_nodes = 250
            print(f"Setting num_nodes to {current_num_nodes} for {city}")

            # 加载当前城市的数据集
            dataloader = util.load_dataset(city_data_paths, args.batch_size, args.batch_size, args.batch_size)
            dataloader["metadata"] = {
                "train": torch.from_numpy(dataloader["metadata"]["train"]),
                "val": torch.from_numpy(dataloader["metadata"]["val"]),
                "test": torch.from_numpy(dataloader["metadata"]["test"])
            }
            scaler = dataloader["scaler"]
            torch.cuda.empty_cache()
            gc.collect()


            # 初始化训练器
            if engine is None:
                engine = trainer(
                    scaler,
                    args.lrate,
                    args.wdecay,
                    config,
                    args.input_dim,
                    args.channels,
                    current_num_nodes,
                    args.input_len,
                    args.output_len,
                    device,
                    args.prune_ratio,
                    args.freeze_ratio,
                    city_save,
                    pruning_indices=None  # 第一个城市不传入剪枝索引
                )
                prune_save_path = os.path.join(city_save, "pruned_indices.pt")
                if hasattr(engine.model, 'save_indices_path'):
                    engine.model.save_indices_path = prune_save_path
            else:
                # 更新节点数量并重新初始化训练器，传入剪枝索引
                engine = trainer(
                    scaler,
                    args.lrate,
                    args.wdecay,
                    config,
                    args.input_dim,
                    args.channels,
                    current_num_nodes,
                    args.input_len,
                    args.output_len,
                    device,
                    args.prune_ratio,
                    args.freeze_ratio,
                    city_save,
                    pruning_indices=current_pruning_indices  # 传入剪枝索引
                )

                # 加载上一个城市的模型权重
                prev_city_save = os.path.join(order_save, f"city_{city_order[city_idx - 1]}")
                engine.load_model(prev_city_save, device)
                print(f"Loaded model weights from previous city {city_order[city_idx - 1]} for {city}")

            # 训练当前城市
            best_val_loss = float('inf')
            epochs_since_best = 0
            # 训练完成后释放缓存
            torch.cuda.empty_cache()
            gc.collect()

            for epoch in range(1, args.epochs + 1):
                # 训练阶段
                train_loss = []
                train_mape = []
                train_rmse = []
                t1 = time.time()

                for iter, (x, y, m) in enumerate(dataloader["train_loader"].get_iterator()):
                    trainx = torch.Tensor(x).to(device)
                    trainx = trainx.permute(0, 3, 2, 1).to(torch.bfloat16)
                    trainy = torch.Tensor(y).to(device).to(torch.bfloat16)
                    metadata_batch = torch.Tensor(m).to(device)
                    real_val = trainy[:, :, :, 0]

                    metrics = engine.train(trainx, real_val, metadata=metadata_batch)
                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])

                    # 释放内存
                    del trainx, trainy, metadata_batch, real_val
                    torch.cuda.empty_cache()
                    gc.collect()

                t2 = time.time()
                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)
                train_time_epoch = t2 - t1

                # 验证阶段
                valid_loss = []
                valid_mape = []
                valid_rmse = []
                s1 = time.time()

                for iter, (x, y, m) in enumerate(dataloader["val_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.permute(0, 3, 2, 1).to(torch.bfloat16)
                    testy = torch.Tensor(y).to(device).to(torch.bfloat16)
                    metadata_batch = torch.Tensor(m).to(device)
                    real_val = testy[:, :, :, 0]

                    metrics = engine.eval(testx, real_val, metadata=metadata_batch)
                    valid_loss.append(metrics[0])
                    valid_mape.append(metrics[1])
                    valid_rmse.append(metrics[2])

                    # 释放内存
                    del testx, testy, metadata_batch, real_val
                    torch.cuda.empty_cache()
                    gc.collect()

                s2 = time.time()
                mvalid_loss = np.mean(valid_loss)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                valid_time_epoch = s2 - s1

                # 记录时间
                train_time.append(train_time_epoch)
                val_time.append(valid_time_epoch)

                # 打印日志
                log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}"
                print(log.format(epoch, mtrain_loss, mtrain_rmse, mtrain_mape), flush=True)

                log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}"
                print(log.format(epoch, mvalid_loss, mvalid_rmse, mvalid_mape), flush=True)

                # 保存指标到CSV
                with open(train_csv_path, 'a') as f:
                    f.write(f"{epoch},{mtrain_loss:.6f},{mtrain_mape:.6f},{mtrain_rmse:.6f},")
                    f.write(f"{mvalid_loss:.6f},{mvalid_mape:.6f},{mvalid_rmse:.6f}\n")
                    # f.write(f"{train_time_epoch:.2f},{valid_time_epoch:.2f}\n")

                # 保存最佳模型
                if mvalid_loss < best_val_loss:
                    best_val_loss = mvalid_loss
                    engine.save_model(city_save)
                    print(f"Saved best model for city {city} at epoch {epoch}")
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1

                # 早期停止
                if epochs_since_best >= args.es_patience and epoch >= 20:
                    print("Early stopping triggered.")
                    break

                # 更新当前剪枝索引
            if hasattr(engine.model, 'get_pruning_indices'):
                current_pruning_indices = engine.model.get_pruning_indices()
                print(f"Updated pruning indices for next city: {current_pruning_indices}")

            # 释放模型资源
            engine.model.to('cpu')
            engine.prompt_network.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

            # 累加当前城市的验证损失
            order_valid_loss += best_val_loss

            # 释放数据加载器资源
            del dataloader, scaler
            torch.cuda.empty_cache()
            gc.collect()

        # 记录当前顺序的总体性能
        print(f"Order {order_str} completed with total validation loss: {order_valid_loss}")

        # 保存最佳顺序和模型
        if order_valid_loss < best_valid_loss:
            best_valid_loss = order_valid_loss
            best_order = city_order
            best_order_str = order_str

            # 保存最佳模型状态
            last_city_save = os.path.join(order_save, f"city_{city_order[-1]}")
            # 设置weights_only=True以避免FutureWarning
            best_model_state = torch.load(os.path.join(last_city_save, "best_model.pth"), weights_only=False)
            best_prompt_state = torch.load(os.path.join(last_city_save, "best_prompt_network.pth"), weights_only=False)

            # 保存最佳剪枝索引
            prune_path = os.path.join(last_city_save, "pruning_indices.pt")
            if os.path.exists(prune_path):
                best_pruning_indices = torch.load(prune_path, weights_only=False)

        # 释放当前顺序的所有资源
        del engine, current_pruning_indices
        torch.cuda.empty_cache()
        gc.collect()

    # 最终测试使用最佳顺序的模型
    print(f"\n{'=' * 50}")
    print(f"Best city order: {best_order} with validation loss: {best_valid_loss}")
    print(f"{'=' * 50}")

    # 修改点：直接将最佳模型保存到基础目录
    torch.save(best_model_state, os.path.join(base_save, "best_model.pth"))
    torch.save(best_prompt_state, os.path.join(base_save, "best_prompt_network.pth"))
    if best_pruning_indices is not None:
        torch.save(best_pruning_indices, os.path.join(base_save, "pruning_indices.pt"))
    print(f"Saved best model directly to base directory: {base_save}")

    # 保存最佳顺序信息
    with open(os.path.join(base_save, "best_order.txt"), 'w') as f:
        f.write(f"Best city order: {best_order_str}\n")
        f.write(f"Best validation loss: {best_valid_loss}\n")

    # 加载最后一个城市的数据集进行最终测试
    last_city = best_order[-1]
    last_datasets = city_datasets[last_city]
    last_data_paths = [f"{d}/" for d in last_datasets['taxi'] + last_datasets['bike']]
    dataloader = util.load_dataset(last_data_paths, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]

    # 确定最后一个城市的节点数量
    taxi_included = any('taxi' in d for d in last_data_paths)
    bike_included = any('bike' in d for d in last_data_paths)

    if taxi_included and bike_included:
        final_num_nodes = max(266, 250)
    elif taxi_included:
        final_num_nodes = 266
    elif bike_included:
        final_num_nodes = 250
    print(f"Setting num_nodes to {final_num_nodes} for final evaluation")

    # 初始化训练器并加载最佳模型
    engine = trainer(
        scaler,
        args.lrate,
        args.wdecay,
        config,
        args.input_dim,
        args.channels,
        final_num_nodes,
        args.input_len,
        args.output_len,
        device,
        args.prune_ratio,
        args.freeze_ratio,
        base_save,  # 直接使用基础目录作为保存路径
        pruning_indices=best_pruning_indices  # 传入最佳剪枝索引
    )

    # 加载最佳模型权重（从基础目录加载）
    engine.model.load_state_dict(torch.load(os.path.join(base_save, "best_model.pth"), map_location=device))
    engine.prompt_network.load_state_dict(torch.load(os.path.join(base_save, "best_prompt_network.pth"), map_location=device))
    if best_pruning_indices is not None and hasattr(engine.model, 'set_pruning_indices'):
        engine.model.set_pruning_indices(best_pruning_indices)

    # 运行测试
    outputs = []
    realy = torch.Tensor(dataloader["y_test"][..., 0]).to(device)
    realy = realy.permute(0, 2, 1)  # [batch, node, time]

    for iter, (x, y, m) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.permute(0, 3, 2, 1).to(dtype=torch.bfloat16)
        with torch.no_grad():
            prompted_testx = engine.prompt_network(testx)
            preds = engine.model(prompted_testx)
        outputs.append(preds.squeeze(-1))

        # 释放内存
        del testx, prompted_testx, preds
        torch.cuda.empty_cache()
        gc.collect()

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]  # [batch, node, time]
    yhat = scaler.inverse_transform(yhat.unsqueeze(-1)).squeeze(-1)

    # 计算指标
    amae = []
    amape = []
    armse = []
    step2_mae = None
    step2_mape = None
    step2_rmse = None

    for j in range(args.output_len):
        pred = yhat[..., j]
        real = realy[..., j]
        metrics = util.metric(pred, real)
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

        # 记录第二个时间步的指标
        if j == 1:
            step2_mae = metrics[0]
            step2_mape = metrics[1]
            step2_rmse = metrics[2]

    # 计算平均指标
    avg_mae = np.mean(amae)
    avg_mape = np.mean(amape)
    avg_rmse = np.mean(armse)

    # 保存测试结果到基础目录
    test_metrics = {
        'avg_mae': avg_mae,
        'avg_mape': avg_mape,
        'avg_rmse': avg_rmse,
        'step2_mae': step2_mae,
        'step2_mape': step2_mape,
        'step2_rmse': step2_rmse,
        'best_order': best_order_str
    }

    test_df = pd.DataFrame([test_metrics])
    test_csv_path = os.path.join(base_save, "test.csv")
    test_df.round(8).to_csv(test_csv_path, index=False)
    print(f"Saved test results to {test_csv_path}")

    # 打印结果
    print("\nFinal Test Results:")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MAPE: {avg_mape:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Step 2 MAE: {step2_mae:.4f}")
    print(f"Step 2 MAPE: {step2_mape:.4f}")
    print(f"Step 2 RMSE: {step2_rmse:.4f}")
    print(f"Best city order: {best_order_str}")

    # 关闭日志文件
    if hasattr(engine, 'log_file') and engine.log_file:
        engine.log_file.close()
    if hasattr(engine, 'debug_log') and engine.debug_log:
        engine.debug_log.close()

    print("Training completed successfully")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))