from dataLoader.dataloader import load_data
from model.ODModel import ODModel
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from datetime import datetime
import argparse
from utils.nll_loss import nb_zeroinflated_nll_loss

def train_model(model, device_id, train_loader, val_loader, epochs=500, patience=10, learning_rate=0.001):
    
    print('\n************************** Training **************************')

    start_time = time.time()
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    # 训练过程
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()
            train_n, train_p, train_pi = model(inputs)
            loss = nb_zeroinflated_nll_loss(targets, train_n, train_p, train_pi)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 计算训练集的平均损失
        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0
        val_mae = 0
        val_rmse = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device).float(), targets.to(device).float()

                n_val,p_val,pi_val = model(inputs)

                loss = nb_zeroinflated_nll_loss(targets, n_val,p_val,pi_val)
                val_loss += loss.item()

                val_pred = (1 - pi_val.detach().cpu().numpy()) * (
                            n_val.detach().cpu().numpy() / p_val.detach().cpu().numpy() - n_val.detach().cpu().numpy())

                mae = np.mean(np.abs(val_pred - targets.detach().cpu().numpy()))
                val_mae += mae.item()

                mse = np.mean((val_pred - targets.detach().cpu().numpy()) ** 2)
                val_rmse += np.sqrt(mse)

        # 计算验证集的平均损失
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_rmse /= len(val_loader)

        if epoch % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train NLL Loss: {train_loss:.4f}, Val NLL Loss: {val_loss:.4f}, Val Pred MAE: {val_mae:.4f} , Val RMSE: {val_rmse:.4f}")

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "ckpt/best_model.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    return best_val_loss

def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))

    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = torch.mean(
            torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()


def test_model(model, device_id, N=110, test_loader=None, lr: float=None, log_name=None, patience=30):
    print('\n************************** Testing ***************************')

    start_time = time.time()
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/best_model.pth"))

    model.eval()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
    mape_total = 0
    criterion = nn.MSELoss()
    N = 110

    all_real_od = []
    all_pred_od = []

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            # 设置对角线掩码
            mask = torch.ones_like(targets)
            for i in range(N):
                mask[:, i, i] = 0  # 对角线上的元素设为 0


            # 推理输出
            n_test, p_test, pi_test = model(inputs)

            # 计算loss
            loss = nb_zeroinflated_nll_loss(targets, n_test, p_test, pi_test)
            test_loss += loss.item()

            # 计算预测值
            mean_pred = (1 - pi_test.detach().cpu().numpy()) * (n_test.detach().cpu().numpy() / p_test.detach().cpu().numpy() - n_test.detach().cpu().numpy())
            mean_pred = torch.tensor(mean_pred, dtype=torch.float32).to(device)

            # 计算 RMSE 和 MAE
            rmse, mae, mape = calculate_rmse_mae(mean_pred * mask, targets)
            rmse_total += rmse
            mae_total += mae
            mape_total += mape

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(mean_pred.cpu().numpy())

    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)

    print(f"Test NLL Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f} Test MAPE: {mape_total:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total test time: {total_time:.2f} s')

    torch.save(model.state_dict(),
               f"ckpt/best_model_{test_loss:.4f}_{rmse_total:.4f}_{mae_total:.4f}_lr_{lr}.pth")

    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)

    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)


    vmin = min(all_pred_od.min(), all_real_od.min())
    vmax = max(all_pred_od.max(), all_real_od.max())

    colors = "Blues"
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(all_real_od, cmap=colors, cbar=True, vmin=vmin, vmax=vmax)
    plt.title("Average True OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    plt.subplot(1, 2, 2)
    sns.heatmap(all_pred_od, cmap=colors, cbar=True, vmin=vmin, vmax=vmax)
    plt.title("Average Predicted OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")
    plt.tight_layout()
    plt.savefig('./figure/result1.png')
    plt.close()


    all_real_od_flat = all_real_od_t.flatten()
    all_pred_od_flat = all_pred_od_t.flatten()

    plt.figure(figsize=(9, 8))
    plt.scatter(all_real_od_flat, all_pred_od_flat, alpha=0.5)

    max_val = np.max([np.max(all_real_od_flat), np.max(all_pred_od_flat)])
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    lims = [0, max_val]
    plt.plot(lims, lims, color='red', linewidth=2, linestyle='-', alpha=0.7)

    plt.xlabel('True OD')
    plt.ylabel('Pred OD')
    plt.title('The relationship between real and predicted OD')
    plt.savefig('./figure/result2.png')
    plt.close()


    with open(log_name, 'a') as log_file:
        log_file.write(
            f"Lr = {lr},patience = {patience},Test Loss: {test_loss:.4f} RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f}\n")


def realtime_infer(model, device_id, N, sample_path, scaler, ckpt_path):
    '''
    实时推理
    :param sample_path: 实时速度数据的路径 --> [N,]
    :param model: 模型
    :param scaler: 归一化器
    :param ckpt_path: 模型最优参数路径
    :return: od_pred：估计的OD矩阵 --> [N,N]
    '''
    print('\n*********************** Real-time Inference ************************')

    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path + "best_model.pth"))

    model.eval()

    sample = np.load(sample_path)

    T = sample.shape[0]
    x = np.arange(0, T)

    plt.figure(figsize=(10, 5))
    plt.plot(x, sample, marker='o')

    plt.title('Current Speed Data')
    plt.xlabel('RegionID')
    plt.ylabel('Speed')
    plt.savefig('./figure/realtime_infer_input.png')
    plt.close()

    sample = scaler.transform(sample.reshape(-1, 1)).reshape(sample.shape)  # [N,]
    sample = torch.from_numpy(np.expand_dims(sample, axis=0)).float()  # [1,N]

    with torch.no_grad():
        input = sample.to(device)
        mask = torch.ones(N, N).to(device)
        for i in range(N):
            mask[i, i] = 0
        n_test, p_test, pi_test = model(input)

        # 计算预测值
        output = (1 - pi_test.detach().cpu().numpy()) * (n_test.detach().cpu().numpy() / p_test.detach().cpu().numpy() - n_test.detach().cpu().numpy())
        output = torch.tensor(output).float().to(device)
        prediction = (output.squeeze(dim=0) * mask).cpu().numpy()

    plt.figure(figsize=(8, 7))
    plt.subplot(1, 1, 1)
    sns.heatmap(prediction, cmap="Blues", cbar=True, vmin=prediction.min(), vmax=prediction.max())
    plt.title("Real-time OD Estimation Result", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")
    plt.tight_layout()
    plt.savefig('./figure/realtime_infer_result.png')
    plt.close()

    print("The real-time inference is done and the results are under the figure path.")
    return prediction





def parse_args():
    parser = argparse.ArgumentParser(description="OD Model Training and Inference")

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate (default: 0.03)')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping (default: 20)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--N', type=int, default=110, help='The number of grids (default: 110)')

    # Data handling
    parser.add_argument('--freq_shuffle', action='store_true', help='Enable frequency shuffling (default: False)')
    parser.add_argument('--od_path', type=str, default='data/dataset/OD_完整批处理.npy',
                        help='Path to OD dataset (default: data/dataset/OD_完整批处理.npy)')
    parser.add_argument('--speed_path', type=str, default='data/dataset/Speed_完整批处理.npy',
                        help='Path to Speed dataset (default: data/dataset/Speed_完整批处理.npy)')

    # Device and paths
    parser.add_argument('--device_id', type=str, default='cuda:3',
                        help='Device ID for training/inference (default: cuda:3)')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='Path to save checkpoints (default: ./ckpt/)')
    parser.add_argument('--realtime_input_path', type=str, default='data/realtimeInput/input_road_speed.npy',
                        help='Path to realtime input data (default: data/realtimeInput/input_road_speed.npy)')
    parser.add_argument('--log_filename', type=str, default=f'log/{datetime.now().strftime("%Y%m%d")}.log',
                        help='Log file path (default: log/YYYYMMDD.log)')

    # Operation mode
    parser.add_argument('--infer_type', type=str, default='testing', choices=['testing', 'realtime'],
                        help='Operation mode: testing or realtime inference (default: testing)')

    return parser.parse_args()


def main(args):
    train_loader, val_loader, test_loader, temp, freq, scaler = load_data(
        od_path=args.od_path,
        speed_path=args.speed_path,
        freq_shuffle=args.freq_shuffle,
        freq_method='STFT',
        seed=args.seed
    )

    model = ODModel(N=args.N, temp=temp, freq=freq)

    if args.infer_type == 'testing':
        train_model(
            model,
            args.device_id,
            train_loader,
            val_loader,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.lr
        )
        test_model(
            model,
            args.device_id,
            N=args.N,
            test_loader=test_loader,
            lr=args.lr,
            log_name=args.log_filename,
            patience=args.patience
        )
    elif args.infer_type == 'realtime':
        realtime_infer(
            model,
            args.device_id,
            args.N,
            sample_path=args.realtime_input_path,
            scaler=scaler,
            ckpt_path=args.ckpt_path
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)