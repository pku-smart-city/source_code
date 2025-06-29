import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
import yaml
from util import *
from scipy.sparse import issparse
import random
from model import ST_LLM, PromptNetwork
from ranger21 import Ranger
from datetime import datetime, timedelta
from prompt_generator_s import generate_prompt
from transformers import AutoTokenizer


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for training")
parser.add_argument("--data", nargs="+", type=str, default=["CHIbike_drop"], help="Data path")
parser.add_argument("--input_dim", type=int, default=3, help="Input dimension")
parser.add_argument("--channels", type=int, default=64, help="Number of features")
parser.add_argument("--num_nodes", type=int, default=266, help="Number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="Input length")
parser.add_argument("--output_len", type=int, default=12, help="Output length")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--U", type=int, default=2, help="Unfrozen attention layer")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--print_every", type=int, default=50, help="Print interval")
parser.add_argument("--wdecay", type=float, default=0.0001, help="Weight decay rate")
parser.add_argument("--es_patience", type=int, default=100, help="Early stopping patience")
args = parser.parse_args()

data_str = "_".join(args.data)

# Define save directory with timestamp
args.save = f'./logs/pretrain-{str(time.strftime("%Y-%m-%d-%H:%M:%S"))}-{data_str}/'


class trainer:
    def __init__(self, scaler, lrate, wdecay, config, input_dim, channels, num_nodes, input_len, output_len, U, device):
        # Initialize tokenizer, main model and prompt network
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
        self.model = ST_LLM(device, input_dim, channels, num_nodes, input_len, output_len, U)
        self.prompt_network = PromptNetwork(input_dim, channels, num_nodes, input_len, output_len)
        self.model = self.model.to(device=device, dtype=torch.bfloat16)
        self.prompt_network = self.prompt_network.to(device=device, dtype=torch.bfloat16)

        # Configuration and logging setup
        self.config = config
        self.regions_to_log = config.get('regions_to_log', [0])
        self.time_format = config.get('time_format', 'half an hour')
        self.log_path = os.path.join(args.save, "prompt_log.txt")
        self.num_nodes = num_nodes
        self.input_len = input_len

        # Optimizers and loss function
        self.model_optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.prompt_optimizer = Ranger(self.prompt_network.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5

        # Validate region IDs
        self.valid_regions = [rid for rid in self.regions_to_log if rid < num_nodes]
        if len(self.valid_regions) != len(self.regions_to_log):
            print(f"Warning: Filtered valid region IDs are {self.valid_regions}")

        # Create save directory and open log files
        os.makedirs(args.save, exist_ok=True)
        self.log_file = open(self.log_path, 'w')
        self.output_len = output_len
        self.debug_log = open(os.path.join(args.save, "debug_log.txt"), 'w')
        self.line_count = 0
        self.device = device

    def train(self, input, real_val, metadata):
        """Training step with personalized prompts"""
        # Prepare input data
        history_input = input.permute(0, 2, 3, 1)
        history_data = self.scaler.inverse_transform(history_input)
        history_data = history_data.squeeze(-1) if history_data.size(-1) == 1 else history_data[..., 0]

        self.model.train()
        self.prompt_network.train()
        self.model_optimizer.zero_grad()
        self.prompt_optimizer.zero_grad()

        # Move data to device
        input = input.to(device=self.model.device, dtype=torch.bfloat16)
        prompted_input = self.prompt_network(input)

        # Initial prediction for prompt generation
        initial_output = self.model(prompted_input)
        initial_preds = initial_output.squeeze(-1)
        initial_preds = self.scaler.inverse_transform(initial_preds.unsqueeze(-1)).squeeze(-1)
        initial_preds = initial_preds.permute(0, 2, 1)

        # Generate personalized prompts using initial predictions
        _, personalized_prompts, _ = generate_prompt(
            initial_preds.detach().cpu(),
            metadata.cpu(),
            self.valid_regions,
            history_data.cpu(),
            self.num_nodes,
            self.output_len,
            self.debug_log
        )

        # Tokenize prompts and get final predictions
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

        # Compute loss and update model
        loss = self.loss(preds, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.prompt_network.parameters(), self.clip)
        self.model_optimizer.step()
        self.prompt_optimizer.step()

        # Calculate evaluation metrics
        mape = util.MAPE_torch(preds, real, 0.0).item()
        rmse = util.RMSE_torch(preds, real, 0.0).item()
        torch.cuda.empty_cache()
        return loss.item(), mape, rmse

    def eval(self, input, real_val, metadata):
        """Evaluation step with personalized prompts (similar to training but without gradient updates)"""
        history_input = input.permute(0, 2, 3, 1)
        history_data = self.scaler.inverse_transform(history_input)
        history_data = history_data.squeeze(-1) if history_data.size(-1) == 1 else history_data[..., 0]

        self.model.eval()
        self.prompt_network.eval()
        input = input.to(device=self.model.device, dtype=torch.bfloat16)
        prompted_input = self.prompt_network(input)

        # Initial prediction for prompt generation
        initial_output = self.model(prompted_input)
        initial_preds = initial_output.squeeze(-1)
        initial_preds = self.scaler.inverse_transform(initial_preds.unsqueeze(-1)).squeeze(-1)
        initial_preds = initial_preds.permute(0, 2, 1)

        # Generate and process prompts
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

        # Compute evaluation metrics
        loss = self.loss(preds, real, 0.0)
        mape = util.MAPE_torch(preds, real, 0.0).item()
        rmse = util.RMSE_torch(preds, real, 0.0).item()

        torch.cuda.empty_cache()
        return loss.item(), mape, rmse

    def generate_prompts(self, input, real_val, metadata):
        """Generate and log personalized prompts"""
        self.model.eval()
        self.prompt_network.eval()
        input = input.to(device=self.model.device, dtype=torch.bfloat16)
        metadata = metadata.clone().detach().to(torch.long).cpu()
        history_input = input.permute(0, 2, 3, 1)
        history_data = self.scaler.inverse_transform(history_input)
        history_data = history_data.squeeze(-1) if history_data.size(-1) == 1 else history_data[..., 0]

        prompted_input = self.prompt_network(input)

        # Get initial predictions
        initial_output = self.model(prompted_input)
        initial_preds = initial_output.squeeze(-1)
        initial_preds = self.scaler.inverse_transform(initial_preds.unsqueeze(-1)).squeeze(-1)
        initial_preds = initial_preds.permute(0, 2, 1)

        # Generate prompts
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

        # Final predictions and logging
        output = self.model(prompted_input, prompt_tokens)
        preds = output.squeeze(-1)
        preds = self.scaler.inverse_transform(preds.unsqueeze(-1)).squeeze(-1)
        preds = preds.transpose(1, 2)

        real = real_val[:, :self.output_len, :]
        real = real.permute(0, 2, 1)

        if metadata is not None:
            # Generate and log final prompts
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


def seed_it(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    """Main training loop"""
    # Load configuration
    config_path = os.path.join("config", "prompt_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['prompt']

    # Set random seed
    seed_it(6666)
    folders = args.data
    data = "_".join(folders)

    # Set number of nodes based on dataset
    if 'taxi' in data:
        args.num_nodes = 266
    elif 'bike' in data:
        args.num_nodes = 250

    device = torch.device(args.device)
    data_paths = [f"{folder}/" for folder in folders]
    dataloader = util.load_dataset(data_paths, args.batch_size, args.batch_size, args.batch_size)

    # Process metadata
    dataloader["metadata"] = {
        "train": torch.from_numpy(dataloader["metadata"]["train"]),
        "val": torch.from_numpy(dataloader["metadata"]["val"]),
        "test": torch.from_numpy(dataloader["metadata"]["test"])
    }

    scaler = dataloader["scaler"]

    # Initialize training variables
    loss = float("inf")
    test_log = float("inf")
    epochs_since_best_mae = 0

    his_loss = []
    val_time = []
    train_time = []
    result = []
    best_test_metrics = None
    best_epoch = 0
    print(args)

    # Create save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Initialize trainer
    engine = trainer(
        scaler,
        args.lrate,
        args.wdecay,
        config,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.U,
        device,
    )

    print("start pretraining...", flush=True)
    # Training loop
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []

        t1 = time.time()
        # Training epoch
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
            torch.cuda.empty_cache()

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # Validation epoch
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
            torch.cuda.empty_cache()

        s2 = time.time()
        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        # Calculate average metrics
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        # Log training metrics
        train_m = dict(
            train_loss=mtrain_loss,
            train_rmse=mtrain_rmse,
            train_mape=mtrain_mape,
            valid_loss=mvalid_loss,
            valid_rmse=mvalid_rmse,
            valid_mape=mvalid_mape,
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}"
        print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape), flush=True)

        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}"
        print(log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape), flush=True)

        # Save best model based on validation loss
        if mvalid_loss < loss:
            print("### Update tasks appear ###")
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), args.save + "best_model.pth")
            torch.save(engine.prompt_network.state_dict(), args.save + "best_prompt_network.pth")
            best_epoch = i
            epochs_since_best_mae = 0

            # Evaluate on test set
            engine.model.load_state_dict(torch.load(args.save + "best_model.pth"))
            engine.prompt_network.load_state_dict(torch.load(args.save + "best_prompt_network.pth"))
            outputs = []
            realy = torch.Tensor(dataloader["y_test"][..., 0]).to(device)
            realy = realy.permute(0, 2, 1)

            for iter, (x, y, m) in enumerate(dataloader["test_loader"].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.permute(0, 3, 2, 1).to(dtype=torch.bfloat16)
                with torch.no_grad():
                    prompted_testx = engine.prompt_network(testx)
                    preds = engine.model(prompted_testx)
                outputs.append(preds)
                torch.cuda.empty_cache()

            yhat = torch.cat(outputs, dim=0)
            yhat = yhat[: realy.size(0), :, :]

            # Calculate test metrics
            amae = []
            amape = []
            armse = []
            future_amae = []
            future_amape = []
            future_armse = []

            for j in range(args.output_len):
                pred = scaler.inverse_transform(yhat[:, :, j].unsqueeze(-1)).squeeze(-1)
                real = realy[:, :, j]
                metrics = util.metric(pred, real)

                if j < 2:
                    future_amae.append(metrics[0])
                    future_amape.append(metrics[1])
                    future_armse.append(metrics[2])

                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

            test_metrics = {
                'epoch': best_epoch,
                'test_mae': np.mean(amae),
                'test_rmse': np.mean(armse),
                'test_mape': np.mean(amape),
                'future_2_mae': np.mean(future_amae) if len(future_amae) > 1 else None,
                'future_2_rmse': np.mean(future_armse) if len(future_armse) > 1 else None,
                'future_2_mape': np.mean(future_amape) if len(future_amape) > 1 else None,
            }
            best_test_metrics = test_metrics

            log = "Epoch: {:03d}, Valid Loss: {:.4f} | Test Loss: {:.4f}, epoch:{}"
            print(log.format(i, mvalid_loss, test_metrics['test_mae'], best_epoch), flush=True)
        else:
            epochs_since_best_mae += 1
            print("No update")

        # Save training metrics to CSV
        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{args.save}/train.csv")

        # Generate and log prompts
        for iter, (x, y, m) in enumerate(dataloader["val_loader"].get_iterator()):
            metadata = dataloader["metadata"]["val"][
                       iter * args.batch_size: (iter + 1) * args.batch_size].clone().detach().to(torch.long).to(device)
            testx = torch.Tensor(x).to(device)
            testx = testx.permute(0, 3, 2, 1).to(torch.bfloat16)
            testy = torch.Tensor(y).to(device).to(torch.bfloat16)
            real_val = testy[:, :, :, 0]
            engine.generate_prompts(testx, real_val, metadata)

        # Early stopping
        if epochs_since_best_mae >= args.es_patience and i >= 20:
            break

    # Print training summary
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", best_epoch)
    print("The valid loss of the best model", str(round(his_loss[best_epoch - 1], 4)))

    # Print and save test metrics
    if best_test_metrics:
        log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
        print(log.format(best_test_metrics['test_mae'], best_test_metrics['test_rmse'], best_test_metrics['test_mape']))

        if best_test_metrics['future_2_mae'] is not None:
            future_log = "Future Step 2 MAE: {:.4f}, Future Step 2 MAPE: {:.4f}, Future Step 2 RMSE: {:.4f}"
            print(future_log.format(best_test_metrics['future_2_mae'], best_test_metrics['future_2_mape'],
                                    best_test_metrics['future_2_rmse']))

        test_metrics = pd.Series(best_test_metrics)
        test_csv = pd.DataFrame([test_metrics])
        test_csv.round(8).to_csv(f"{args.save}/test.csv")

    # Close log files
    log_path = os.path.join(args.save, "prompt_log.txt")
    if os.path.exists(log_path):
        engine.log_file.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))