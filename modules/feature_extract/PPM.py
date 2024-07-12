# _*_coding:utf-8_*_
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(self, in_dim, out_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x] 
        for f in self.features:
            temp = f(x)
            temp = F.interpolate(temp, x_size[2:], mode="bilinear", align_corners=True)
            out.append(temp)

        # 通道维度拼接
        return torch.cat(out, 1)


if __name__ == "__main__":
    # inputs: (B, C, H, W)
    inputs = torch.rand((8, 3, 16, 16))
    # PPM params: (in_dim, out_dim, sizeList)
    ppm = PPM(3, 2, [1, 2, 3, 6])
    # outputs: (B=8, C=3+2*4=11, H=16, W=16)
    outputs = ppm(inputs)
    print("Outputs shape:", outputs.size())
