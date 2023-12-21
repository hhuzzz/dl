from ViT import *
from torchinfo import summary

if __name__ == "__main__":
    model = vit_base_patch16_224()
    input = torch.randn(1, 3, 224, 224)
    print(model(input).shape)
