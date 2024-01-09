# AFormer: A simple A-Shaped Transformer for Defect Semantic Segmentation
from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helpers


def exists(val):
    return val is not None


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# classes


class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(
            dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False
        )
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=heads), (q, k, v)
        )

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) (x y) c -> b (h c) x y", h=heads, x=h, y=w)
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(self, *, dim, expansion_factor):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class MiT(nn.Module):
    def __init__(
        self, *, channels, dims, heads, ff_expansion, reduction_ratio, num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (
            (dim_in, dim_out),
            (kernel, stride, padding),
            num_layers,
            ff_expansion,
            heads,
            reduction_ratio,
        ) in zip(
            dim_pairs,
            stage_kernel_stride_pad,
            num_layers,
            ff_expansion,
            heads,
            reduction_ratio,
        ):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel**2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim_out,
                                EfficientSelfAttention(
                                    dim=dim_out,
                                    heads=heads,
                                    reduction_ratio=reduction_ratio,
                                ),
                            ),
                            PreNorm(
                                dim_out,
                                MixFeedForward(
                                    dim=dim_out, expansion_factor=ff_expansion
                                ),
                            ),
                        ]
                    )
                )

            self.stages.append(
                nn.ModuleList([get_overlap_patches, overlap_patch_embed, layers])
            )

    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]

        layer_outputs = []
        for get_overlap_patches, overlap_embed, layers in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, "b c (h w) -> b c h w", h=h // ratio)

            x = overlap_embed(x)
            for attn, ff in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class BSAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, A1_B, A1_C):
        init_h = A1_B.shape[-2]
        B1 = self.conv_1(A1_B)
        B1 = rearrange(B1, "b c h w -> b (h w) c")

        C1 = self.conv_2(A1_C)
        C1 = rearrange(C1, "b c h w -> b c (h w)")

        D1 = self.conv_3(A1_C)
        D1 = rearrange(D1, "b c h w -> b (h w) c")

        S1 = F.softmax(B1 @ C1, dim=-1)
        E1 = rearrange(S1 @ D1, "b (h w) c -> b c h w", h=init_h) + A1_C

        # print(B1.shape, C1.shape, D1.shape, S1.shape, E1.shape)
        return E1


class BCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, A2_B, A2_C):
        B2 = rearrange(self.avg_1(A2_B), "b c (h h1) w -> b (c h1) (h w)", h1=2)
        C2 = rearrange(self.avg_2(A2_C), "b c (h h1) w -> b (h w) (c h1)", h1=2)
        S2 = F.softmax(B2 @ C2, dim=-1)
        A2_C_reshaped = rearrange(A2_C, "b c (h h1) w -> b h w (c h1)", h1=2)
        E2 = (
            rearrange(
                (A2_C_reshaped @ S2) * self.beta, "b h w (c c1)-> b c (h c1) w", c1=2
            )
            + A2_C
        )
        return E2


class ABFormer(nn.Module):
    def __init__(
        self,
        *,
        dims=(32, 64, 160, 256),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=2,
        channels=3,
        decoder_dim=256,
        num_classes=4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(
            partial(cast_tuple, depth=4),
            (dims, heads, ff_expansion, reduction_ratio, num_layers),
        )
        assert all(
            [
                *map(
                    lambda t: len(t) == 4,
                    (dims, heads, ff_expansion, reduction_ratio, num_layers),
                )
            ]
        ), "only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values"

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
        )

        self.to_fused = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, decoder_dim, 1), nn.Upsample(scale_factor=2**i)
                )
                for i, dim in enumerate(dims)
            ]
        )

        self.to_boundary = nn.ModuleList(
            [nn.Conv2d(dim, decoder_dim, kernel_size=3, padding=1) for dim in dims]
        )

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

        split_idx = num_classes / 2

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)

        boundary_fused = [
            to_boundary(output)
            for output, to_boundary in zip(layer_outputs, self.to_boundary)
        ]

        context_fused = [
            to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)
        ]

        # 返回BoundaryPath融合的特征 (num_classes * H/4 * W/4)
        boundary_fused = self.to_segmentation(torch.cat(boundary_fused, dim=1))

        # 返回ContextPath融合的特征 (num_classes * H/4 * W/4)
        context_fused = self.to_segmentation(torch.cat(context_fused, dim=1))

        
        return context_fused


if __name__ == "__main__":
    # mit = MiT(
    #     channels=3,
    #     dims=(32, 64, 160, 256),
    #     heads=(1, 2, 5, 8),
    #     ff_expansion=(8, 8, 4, 4),
    #     reduction_ratio=(8, 4, 2, 1),
    #     num_layers=(2, 2, 2, 2),
    # )
    # segformer = Segformer()
    # x = torch.randn(1, 3, 224, 224)
    # layer_outputs = mit(x, return_layer_outputs=True)
    # for i in range(4):
    #     print(layer_outputs[i].shape)

    # y = segformer(x)
    # print(y.shape)
    # bcam = BCAM()
    # x = torch.randn(1, 4, 16, 16)
    # y = torch.randn(1, 4, 16, 16)
    # bcam(x, y)
    abformer = ABFormer()