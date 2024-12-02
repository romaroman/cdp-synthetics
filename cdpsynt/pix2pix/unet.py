from __future__ import annotations
from functools import reduce
from operator import __add__

import torch
from torch import Tensor, nn
from torch.nn.functional import pad


__all__ = ["UNetModel", "DoubleConv", "Down", "Up", "OutConv"]

def SamePad2d(ksize: int) -> nn.ZeroPad2d:
    return nn.ZeroPad2d(
        reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in ksize[::-1]])
    )


class UNetModel(nn.Module):
    def __init__(
        self,
        channels_in: int = 1,
        channels_out: int = 1,
        bilinear: bool = True,
        layer_norm: bool = False,
        filters: list[int] = None,
        kernel_size: int = 3,
    ) -> UNetModel:
        super().__init__()

        filters: list[int] = filters or [64, 128, 256, 512, 1024, 2048]

        self.channels_in: int = channels_in
        self.channels_out: int = channels_out
        self.bilinear: bool = bilinear
        self.factor: int = 2 if bilinear else 1
        self.layer_norm: bool = layer_norm
        self.ks: int = kernel_size

        self.inc = DoubleConv(channels_in, filters[0], kernel_size=3)

        self.down1 = Down(filters[0], filters[1], self.layer_norm, self.ks)
        self.down2 = Down(filters[1], filters[2], self.layer_norm, self.ks)
        self.down3 = Down(filters[2], filters[3], self.layer_norm, self.ks)
        self.down4 = Down(filters[3], filters[4], self.layer_norm, self.ks)
        self.down5 = Down(filters[4], filters[5] // self.factor, self.layer_norm, self.ks)

        self.up1 = Up(filters[5], filters[4] // self.factor, self.bilinear, kernel_size=self.ks)
        self.up2 = Up(filters[4], filters[3] // self.factor, self.bilinear, kernel_size=self.ks)
        self.up3 = Up(filters[3], filters[2] // self.factor, self.bilinear, kernel_size=self.ks)
        self.up4 = Up(filters[2], filters[1] // self.factor, self.bilinear, kernel_size=self.ks)
        self.up5 = Up(filters[1], filters[0], self.bilinear, kernel_size=self.ks)

        self.outc = OutConv(filters[0], self.channels_out, 16, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        o = self.outc(x)

        return o


class DoubleConv(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        channels_mid: int | None = None,
        layer_norm: bool = False,
        kernel_size: int = 3,
    ) -> DoubleConv:
        super().__init__()

        self.layer_norm: bool = layer_norm

        channels_mid = channels_mid or channels_out

        self.double_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, kernel_size=(kernel_size, kernel_size), padding=(kernel_size - 2, kernel_size - 2)),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_out, kernel_size=(kernel_size, kernel_size), padding=(kernel_size - 2, kernel_size - 2)),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.double_conv(x)

        if self.layer_norm:
            ln = nn.LayerNorm(x.size()[1:]).to(x.device) 
            x = ln(x)

        return x


class Down(nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, layer_norm: bool = False, kernel_size: int = 3
    ) -> Down:
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels_in, channels_out, layer_norm=layer_norm, kernel_size=kernel_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(
        self,
        channels_int: int,
        channels_out: int,
        bilinear: bool = True,
        layer_norm: bool = False,
        kernel_size: int = 3
    ) -> Up:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                channels_int, channels_out, channels_int // 2, layer_norm=layer_norm
            )
        else:
            self.up = nn.ConvTranspose2d(
                channels_int, channels_int // 2, kernel_size=(kernel_size, kernel_size), stride=(2, 2)
            )
            self.conv = DoubleConv(channels_int, channels_out, layer_norm=layer_norm)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, channels_mid: int, kernel_size: int = 3, kernel_size2: int = 1
    ) -> OutConv:
        super(OutConv, self).__init__()

        self.out = nn.Sequential(
            SamePad2d(ksize=(3, 3)),
            nn.Conv2d(channels_in, channels_mid, kernel_size=(kernel_size, kernel_size)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_out, kernel_size=(kernel_size2, kernel_size2)),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out(x)
