import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    def __init__(
        self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.5
    ):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.merge = merge
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout

        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scaling = self.lora_alpha / self.rank
            self.linear.weight.requires_grad = False

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        if self.rank > 0 and self.merge:
            print((self.lora_b @ self.lora_a * self.scaling).shape)
            output = F.linear(
                x,
                self.linear.weight + self.lora_b @ self.lora_a * self.scaling,
                self.linear.bias,
            )
            output = self.dropout(output)
            return output
        else:
            return self.dropout(self.linear(x))


if __name__ == "__main__":
    x = torch.randn(2, 3)
    lora = LoRALinear(3, 4, True)
    print(lora(x).shape)
