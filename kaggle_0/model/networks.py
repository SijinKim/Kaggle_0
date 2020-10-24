import torch
import torch.nn as nn

from model.layers import ConvBlock
from model.layers import FractionalConvBlock
from model.layers import ResBlock

from model import ops


class Generator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filters=64,
                 norm='inorm'):
        # ResNetGenerator - using 9 Resblocks
        super(Generator, self).__init__()

        self.sub_mean = ops.MeanShift(255)
        self.add_mean = ops.MeanShift(255, sign=1)

        down_layers = [
            ConvBlock(in_channels=in_channels,
                      out_channels=1 * filters,
                      kernel_size=7,
                      stride=1,
                      padding=(7-1)//2,
                      norm=norm,
                      relu=0.0,
                      reflection_padding=True),

            ConvBlock(in_channels=1 * filters,
                      out_channels=2 * filters,
                      kernel_size=3,
                      stride=2,
                      padding=(3-1)//2,
                      norm=norm,
                      relu=0.0,
                      reflection_padding=True),

            ConvBlock(in_channels=2 * filters,
                      out_channels=4 * filters,
                      kernel_size=3,
                      stride=2,
                      padding=(3-1)//2,
                      norm=norm,
                      relu=0.0,
                      reflection_padding=True)
        ]

        res_layers = []
        res_layers += [
            ResBlock(in_channels=4 * filters,
                     out_channels=4 * filters,
                     kernel_size=3,
                     stride=1,
                     padding=(3-1)//2,
                     norm=norm,
                     relu=0.0) for _ in range(9)
        ]

        up_layers = [
            FractionalConvBlock(in_channels=4 * filters,
                                out_channels=2 * filters,
                                kernel_size=3,
                                stride=2,
                                padding=(3-1)//2,
                                norm=norm,
                                relu=0.0,
                                reflection_padding=True),

            FractionalConvBlock(in_channels=2 * filters,
                                out_channels=1 * filters,
                                kernel_size=3,
                                stride=2,
                                padding=(3-1)//2,
                                norm=norm,
                                relu=0.0,
                                reflection_padding=True),

            nn.ReflectionPad2d(3),

            ConvBlock(in_channels=1 * filters,
                      out_channels=out_channels,
                      kernel_size=7,
                      stride=1,
                      padding=0,
                      norm=None,
                      relu=None)
        ]

        self.downsampling = nn.Sequential(*down_layers)
        self.res = nn.Sequential(*res_layers)
        self.upsampling = nn.Sequential(*up_layers)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.downsampling(x)
        x = self.res(x)
        x = torch.tanh(self.upsampling(x))

        x = self.add_mean(x)

        return x


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filters=64,
                 bias=False,
                 norm='inorm',
                 relu=0.2):
        # Basic discriminator - using PatchGAN's Discriminator
        super(Discriminator, self).__init__()

        layers = [
            ConvBlock(in_channels=in_channels,
                      out_channels=1 * filters,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias,
                      norm=None,
                      relu=relu),

            ConvBlock(in_channels=1 * filters,
                      out_channels=2 * filters,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias,
                      norm=norm,
                      relu=relu),

            ConvBlock(in_channels=2 * filters,
                      out_channels=4 * filters,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias,
                      norm=norm,
                      relu=relu),

            ConvBlock(in_channels=4 * filters,
                      out_channels=8 * filters,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias,
                      norm=norm,
                      relu=relu),

            ConvBlock(in_channels=8 * filters,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=bias,
                      norm=None,
                      relu=None)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = torch.sigmoid(x)

        return x
