import torch
from torch import nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels
        print(self.grids)

    def forward(self, feats):
        b, c, h, w = feats.size()

        ar = w / h

        return torch.cat(
            [
                F.adaptive_avg_pool2d(
                    feats, (self.grids[0], max(1, round(ar * self.grids[0])))
                ).view(b, self.channels, -1),
                F.adaptive_avg_pool2d(
                    feats, (self.grids[1], max(1, round(ar * self.grids[1])))
                ).view(b, self.channels, -1),
                F.adaptive_avg_pool2d(
                    feats, (self.grids[2], max(1, round(ar * self.grids[2])))
                ).view(b, self.channels, -1),
                F.adaptive_avg_pool2d(
                    feats, (self.grids[3], max(1, round(ar * self.grids[3])))
                ).view(b, self.channels, -1),
            ],
            dim=2,
        )


class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256, inter_channels=32):
        super(LocalAttenModule, self).__init__()

        print("sigmoid")
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, inter_channels, 1, 1, 0),
            nn.Conv2d(
                inter_channels, in_channels, kernel_size=3, padding=1, bias=False
            ),
        )

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)

        res1 = res1 * x_mask

        return res1 + res2


class CFC_CRB(nn.Module):
    def __init__(
        self, in_channels=512, inter_channels=256, grids=(6, 3, 2, 1)
    ):  # 先ce后ffm
        super(CFC_CRB, self).__init__()
        self.grids = grids
        self.inter_channels = inter_channels

        self.reduce_channel = ConvBNReLU(in_channels, inter_channels, 3, 1, 1)

        self.query_conv = nn.Conv2d(
            in_channels=inter_channels, out_channels=32, kernel_size=1
        )
        self.key_conv = nn.Conv1d(
            in_channels=inter_channels, out_channels=32, kernel_size=1
        )
        self.value_conv = nn.Conv1d(
            in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1
        )
        self.key_channels = 32

        self.value_psp = PSPModule(grids, 128)
        self.key_psp = PSPModule(grids, 128)

        self.softmax = nn.Softmax(dim=-1)

        self.local_attention = LocalAttenModule(inter_channels, inter_channels // 8)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.reduce_channel(x)  # 降维- 128

        m_batchsize, _, h, w = x.size()

        query = (
            self.query_conv(x).view(m_batchsize, 32, -1).permute(0, 2, 1)
        )  ##  b c n ->  b n c

        key = self.key_conv(self.key_psp(x))  ## b c s

        sim_map = torch.matmul(query, key)

        sim_map = self.softmax(sim_map)
        # sim_map = self.attn_drop(sim_map)
        value = self.value_conv(self.value_psp(x))  # .permute(0,2,1)  ## b c s

        # context = torch.matmul(sim_map,value) ## B N S * B S C ->  B N C
        context = torch.bmm(
            value, sim_map.permute(0, 2, 1)
        )  #  B C S * B S N - >  B C N

        # context = context.permute(0,2,1).view(m_batchsize,self.inter_channels,h,w)
        context = context.view(m_batchsize, self.inter_channels, h, w)
        # out = x + self.gamma * context
        context = self.local_attention(context)

        out = x + context

        return out


class SFC_G2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SFC_G2, self).__init__()

        self.conv_8 = ConvBNReLU(64, 128, 3, 1, 1)

        self.conv_32 = ConvBNReLU(256, 128, 3, 1, 1)

        self.groups = 2

        print("groups", self.groups)

        self.conv_offset = nn.Sequential(
            ConvBNReLU(256, 64, 1, 1, 0),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False),
        )

        self.keras_init_weight()

        self.conv_offset[1].weight.data.zero_()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, cp, sp):
        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征  1 / 8  256
        sp = F.interpolate(sp, cp.size()[2:], mode="bilinear", align_corners=True)
        # x_8
        cp = self.conv_8(cp)

        conv_results = self.conv_offset(torch.cat([cp, sp], 1))

        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0 : self.groups * 2, :, :].reshape(
            n * self.groups, -1, out_h, out_w
        )
        offset_h = conv_results[:, self.groups * 2 : self.groups * 4, :, :].reshape(
            n * self.groups, -1, out_h, out_w
        )

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n * self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l, align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h, align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups * 4 :, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = F.interpolate(
            x, scale_factor=self.up_factor, mode="bilinear", align_corners=True
        )
        return x

class CSFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CFC = CFC_CRB(in_channels=256, inter_channels=128)
        self.SFC = SFC_G2()
        self.to_seg = BiSeNetOutput(in_chan=128, mid_chan=64, n_classes=5, up_factor=8)
    
    def forward(self, f_8, f_32):
        self.f_32 = self.CFC(f_32)
        out = self.to_seg(self.SFC(f_8, f_32))
        return out


if __name__ == "__main__":
    CFC = CFC_CRB(in_channels=256, inter_channels=128)
    SFC = SFC_G2()
    CSFCN = CSFCN()
    f_8 = torch.randn(1, 64, 28, 28)
    f_32 = torch.randn(1, 256, 7, 7)
    # print(CFC(f_32).shape)
    print(CSFCN(f_8, f_32).shape)
