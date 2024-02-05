from models.ViT import *
from torchinfo import summary
from torch.nn import Linear

if __name__ == "__main__":
    input = torch.randn(2, 3, dtype=float)
    net = Linear(3, 4)
    x = net(input)
    for x, y in net.named_parameters():
        print(x, y.dtype)
