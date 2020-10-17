import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 norm=None,
                 relu=0.0,
                 reflection_padding=None):
        super(ConvBlock, self).__init__()
        layers = []

        if reflection_padding is not None:
            layers += [nn.ReflectionPad2d(padding=padding)]
            layers += [nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=bias
            )]
        else:
            layers += [nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )]

        if norm is not None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if relu is not None:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FractionalConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 norm=None,
                 relu=0.0,
                 reflection_padding=None):
        super(FractionalConvBlock, self).__init__()
        layers = []

        if reflection_padding is not None:
            layers += [nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=padding,
                bias=bias
            )]
        else:
            layers += [nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )]

        if norm is not None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if relu is not None:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 norm='bnorm',
                 relu=0.0):
        super(ResBlock, self).__init__()
        layers = []

        layers += [ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            norm=norm,
            relu=relu
        )]

        layers += [ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            norm=norm,
            relu=None
        )]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)
