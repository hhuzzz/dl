import torch

# 创建一个矩阵
matrix = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

# 转置矩阵
transposed_matrix = matrix.transpose(0, 1)  # 交换行和列

print("原始矩阵:")
print(matrix)

print("\n转置后的矩阵:")
print(transposed_matrix)
