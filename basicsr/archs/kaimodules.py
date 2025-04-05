import torch
import torch.nn as nn
from typing import List
import math
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch import Tensor


def init_weights(init_type=None, gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type is None:
                m.reset_parameters()
            else:
                raise ValueError(f'invalid initialization method: {init_type}.')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


def token2image(X):
    B, N, C = X.shape
    img_size = int(math.sqrt(N))
    assert img_size * img_size == N
    return X.permute(0, 2, 1).reshape(B, C, img_size, img_size)


def image2token(X):
    B, C, H, W = X.shape
    return X.reshape(B, C, H*W).permute(0, 2, 1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal', sn: bool = False):
        super().__init__()
        if sn:
            self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'norm {norm} is not valid.')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'activation {activation} is not valid.')
        self.conv.apply(init_weights(init_type))

    def forward(self, X:Tensor):
        X = self.conv(X)
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)
        return X


class TransposedConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal', sn: bool = False):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size, stride, padding, norm, activation, init_type, sn)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, X: Tensor, X_lateral: Tensor = None):
        X = self.up(X)
        if X_lateral is not None:
            X = self.conv(torch.cat([X, X_lateral], dim=1))
        else:
            X = self.conv(X)
        return X


def Upsample(in_channels: int, out_channels: int, legacy_v: int = 4):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear' if legacy_v in [3, 4] else 'nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )


def Downsample(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)


class PartialConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros((out_channels, )))
        self.mask_conv_weight = torch.ones(1, 1, kernel_size, kernel_size)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'norm {norm} is not valid.')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'activation {activation} is not valid.')
        self.conv.apply(init_weights(init_type))

    def forward(self, X: Tensor, mask: Tensor):
        """ Note that 0 in mask denote invalid pixels (holes).

        Args:
            X: Tensor[bs, C, H, W]
            mask: Tensor[bs, 1, H, W]

        """
        if mask is None:
            mask = torch.ones_like(X[:, :1, :, :])
        self.mask_conv_weight = self.mask_conv_weight.to(device=mask.device)
        with torch.no_grad():
            mask_conv = F.conv2d(mask, self.mask_conv_weight, stride=self.stride, padding=self.padding)
        invalid_pos = mask_conv == 0

        scale = self.kernel_size * self.kernel_size / (mask_conv + 1e-8)
        scale.masked_fill_(invalid_pos, 0.)  # type: ignore

        X = self.conv(X * mask)
        X = X * scale + self.bias.view(1, -1, 1, 1)
        X.masked_fill_(invalid_pos, 0.)
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)

        new_mask = torch.ones_like(mask_conv)
        new_mask.masked_fill_(invalid_pos, 0.)

        return X, new_mask


class GatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal'):
        super().__init__()
        self.gate = ConvNormAct(in_channels, out_channels, 3, stride=1, padding=1, norm=norm, activation='sigmoid')
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size, stride, padding, norm, activation, init_type)

    def forward(self, X: Tensor):
        return self.conv(X) * self.gate(X)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal', sn: bool = False,
                 partial: bool = False, gated: bool = False):
        super().__init__()
        assert not (partial and gated)
        self.partial = partial
        if partial:
            self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, activation=activation, init_type=init_type)
            self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation, init_type=init_type)
        elif gated:
            self.conv1 = GatedConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, activation=activation, init_type=init_type)
            self.conv2 = GatedConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation, init_type=init_type)
        else:
            self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, activation=activation, init_type=init_type, sn=sn)
            self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation, init_type=init_type, sn=sn)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.apply(init_weights(init_type))

    def forward(self, X: Tensor, mask: Tensor = None):
        shortcut = self.shortcut(X)
        if not self.partial:
            X = self.conv1(X)
            X = self.conv2(X)
            return X + shortcut
        else:
            X, mask = self.conv1(X, mask)
            X, mask = self.conv2(X, mask)
            return X + shortcut, mask


class PatchResizing2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool = False, up: bool = False, partial: bool = False):
        super().__init__()
        assert not (down and up), f'down and up cannot be both True'
        Conv = PartialConv2d if partial else ConvNormAct
        if down:
            self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, activation='gelu')
        else:
            self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='gelu')
        if up:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        else:
            self.up = None

    def forward(self, x, mask=None):
        if self.up is not None:
            x = self.up(x)
            if mask is not None:
                mask = self.up(mask)
        if isinstance(self.conv, PartialConv2d):
            x, mask = self.conv(x, mask)
            return x, mask
        else:
            x = self.conv(x)
            return x


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ProjectConv(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 64,
            use_se: bool = False,
            se_reduction=None,
    ):
        if use_se:
            if se_reduction is None:
                se_reduction = 2
            super(ProjectConv, self).__init__(SELayer(in_channels, se_reduction),
                                              nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                              nn.BatchNorm2d(out_channels),
                                              nn.LeakyReLU()
                                              )
        else:
            super(ProjectConv, self).__init__(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                              nn.BatchNorm2d(out_channels),
                                              nn.LeakyReLU(),
                                              )


class FRUnit(nn.Module):
    def __init__(
            self,
            channels: int,
            normalize: bool = True,
    ):
        super(FRUnit, self).__init__()
        self.normalize = normalize
        self.LS_conv = ResBlock(channels, channels, kernel_size=3, padding=1, stride=1)

    def forward(self, feature):
        ls_feature = self.LS_conv(feature)
        if self.normalize:
            ls_feature = F.normalize(ls_feature)

        return ls_feature
