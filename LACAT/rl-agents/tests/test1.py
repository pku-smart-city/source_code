import numpy as np

# 假设你有一个包含点坐标的数组
points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

# 计算点之间的欧氏距离
distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

# 定义距离阈值
threshold = 5

# 基于距离阈值构建邻接矩阵
adj_matrix = (distances < threshold).astype(int)

# 打印邻接矩阵
print(adj_matrix)