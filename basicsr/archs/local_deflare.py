import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.kaimodules import ConvNormAct


class SCES(nn.Module):
    def __init__(self, dim, mask_nc):
        super().__init__()

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 64
        nhidden = dim

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(mask_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, dim, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, dim, kernel_size=3, padding=1)

    def forward(self, x, flare_mask):

        actv = self.mlp_shared(flare_mask)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = x * (1. - gamma) + beta

        return out


class SCESlayer(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super(SCESlayer, self).__init__()

        # create conv layers
        self.conv_0 = ConvNormAct(in_ch, out_ch, kernel_size=5, stride=1, padding=2, activation='gelu')
        self.spade = SCES(out_ch, mask_nc=3)

    def forward(self, x, flare):
        dx = self.conv_0(x)
        dx = self.spade(dx, flare)
        dx = self.actvn(dx)
        return dx

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)