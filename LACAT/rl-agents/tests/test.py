import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 3)
        self.relu = nn.ReLU()

    def forward(self, points_list):
        points_list = np.array(points_list)
        if len(points_list.shape) > 2:
            outputs = []
            for points in points_list:
                distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)  # 计算点之间的欧氏距离
                threshold = 0.5  # 距离阈值
                adj_matrix = (distances < threshold).astype(int)  # 基于距离阈值构建邻接矩阵
                rowsum = np.sum(adj_matrix, axis=1)
                D = np.diag(1.0 / np.sqrt(rowsum))
                normalized_adj_matrix = np.matmul(np.matmul(D, adj_matrix), D)
                node_features = torch.tensor(points, dtype=torch.float)
                normalized_adj_matrix = torch.tensor(normalized_adj_matrix)
                normalized_adj_matrix = normalized_adj_matrix.float()
                x = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
                x = self.relu(x)
                x = self.conv2(torch.matmul(normalized_adj_matrix, x))
                x = self.fc(x.mean(dim=0, keepdim=True))
                outputs.append(x)
            return torch.cat(outputs, dim=0)
        else:
            points = np.array(points_list)
            distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)  # 计算点之间的欧氏距离
            threshold = 0.5  # 距离阈值
            adj_matrix = (distances < threshold).astype(int)  # 基于距离阈值构建邻接矩阵
            rowsum = np.sum(adj_matrix, axis=1)
            D = np.diag(1.0 / np.sqrt(rowsum))
            normalized_adj_matrix = np.matmul(np.matmul(D, adj_matrix), D)
            node_features = torch.tensor(points, dtype=torch.float)
            # 前向传播
            normalized_adj_matrix = torch.tensor(normalized_adj_matrix)
            node_features = node_features.clone().detach()
            normalized_adj_matrix = normalized_adj_matrix.float()
            x = self.conv1(torch.matmul(normalized_adj_matrix, node_features))
            x = self.relu(x)
            x = self.conv2(torch.matmul(normalized_adj_matrix, x))
            x = self.fc(x.mean(dim=0, keepdim=True))
            return x


# 创建GCN模型
input_dim = 2  # 输入维度为2，即横纵坐标
hidden_dim = 64  # 隐藏层维度
output_dim = 64  # GCN输出维度
fc_output_dim = 3  # 全连接层输出维度

gcn_model = GCN(input_dim, hidden_dim, output_dim)
fc_layer = nn.Linear(output_dim, fc_output_dim)

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = optim.SGD(gcn_model.parameters(), lr=0.01)

num_epochs = 500

# 训练过程
for epoch in range(num_epochs):
    # points_list = [
    #     [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
    #     [[1, 9], [2, 7], [4, 5], [1, 1], [2, 1]],
    # ]  # 两组点的坐标作为节点特征
    # target_values = [[1, 22, 3], [10, 3, 10]]  # 目标值
    points_list = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    target_values = [1, 15, 3]
    target_values = torch.tensor(target_values, dtype=torch.float32)

    gcn_output = gcn_model(points_list)
    # print(gcn_output)
    loss = loss_function(gcn_output, target_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Output: {gcn_output.detach().numpy()}")
