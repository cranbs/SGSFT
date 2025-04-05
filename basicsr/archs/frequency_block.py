import torch
import torch.nn as nn


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class MSFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFB, self).__init__()
        self.fourier_unit = FreBlock(in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.conv_3 = nn.Conv2d(in_channels*2, out_channels, kernel_size=(1, 1), stride=(1, 1), padding='same')

    def forward(self, x):
        _, _, H, W = x.shape
        x_s = self.relu(self.conv_1(x))
        x_s = x + self.conv_2(x_s)

        x_freq = torch.fft.rfft2(x, norm='backward')
        x_f = self.fourier_unit(x_freq)
        x_f = torch.fft.irfft2(x_f, s=(H, W), norm='backward')

        out = self.conv_3(torch.cat((x_s, x_f), dim=1))

        return out


