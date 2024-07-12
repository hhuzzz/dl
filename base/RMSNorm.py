import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 计算均方根
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # 归一化
        x_norm = x / rms
        # 缩放和平移
        return x_norm * self.scale + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        # 缩放和平移
        return x_norm * self.gamma + self.beta


# 示例使用
input_tensor = torch.randn(2, 3)
rms_norm = RMSNorm(dim=input_tensor.shape[-1])
output_tensor = rms_norm(input_tensor)
print('RMSNorm:')
print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output_tensor)

layer_norm = LayerNorm(dim=input_tensor.shape[-1])
output_tensor = layer_norm(input_tensor)
print('LayerNorm:')
print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output_tensor)