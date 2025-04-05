from typing import List, Mapping, Any
from basicsr.utils.registry import ARCH_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from basicsr.archs.kaimodules import ConvNormAct, PatchResizing2d, Upsample,Downsample
from basicsr.archs.kaiswin import BasicLayer2d
from basicsr.archs.NAF_arch import NAFNet
from basicsr.archs.local_deflare import SCESlayer


class Encoder(nn.Module):
    def __init__(self,
                 img_size: int = 512,
                 dim: int = 64,
                 n_conv_stages: int = 1,
                 dim_mults: List[int] = (1, 2, 4),
                 depths: List[int] = (2, 2, 2),
                 window_size: int = 8,
                 legacy_v: int = 4,
                 ):
        super().__init__()
        assert len(dim_mults) == len(depths)
        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)]
        self.n_head = dim_mults
        res = img_size

        # convolution stages
        self.conv_down_blocks = nn.ModuleList([])
        for i in range(n_conv_stages):
            if legacy_v != 3:
                self.conv_down_blocks.append(nn.Sequential(
                    Downsample(dim, dim),
                    nn.GELU(),
                ))
            else:
                self.conv_down_blocks.append(nn.Sequential(
                    nn.GELU(),
                    Downsample(dim, dim),
                ))
            res = res // 2

        self.down_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            self.down_blocks.append(nn.ModuleList([
                BasicLayer2d(
                    dim=self.dims[i],
                    input_resolution=(res, res),
                    depth=depths[i],
                    num_heads=self.n_head[i],
                    window_size=window_size,
                    partial=True,
                ),
                PatchResizing2d(
                    in_channels=self.dims[i],
                    out_channels=self.dims[i+1],
                    down=True,
                ) if i < self.n_stages - 1 else nn.Identity(),
            ]))
            if i < self.n_stages -1:
                res = res // 2

    def forward(self, X: Tensor, masks: List):
        # X = self.first_conv(X)

        for blk in self.conv_down_blocks:
            X = blk(X)

        skips = []
        for (blk, down), mask in zip(self.down_blocks, masks):
            X, mask = blk(X, mask.float())
            skips.append(X)
            X = down(X)

        return X, skips


class Bottleneck(nn.Module):
    def __init__(
            self,
            img_size: int = 512,
            dim: int = 64,
            n_conv_stages: int = 0,
            dim_mults: List[int] = (1, 2, 4),
            depth: int = 2,
            window_size: int = 8,
    ):
        super().__init__()
        n_stages = len(dim_mults)
        res = img_size // (2 ** (n_stages - 1 + n_conv_stages))
        self.bottleneck = BasicLayer2d(
            dim=dim * dim_mults[-1],
            input_resolution=(res, res),
            depth=depth,
            num_heads=dim_mults[-1],
            window_size=window_size,
        )

    def forward(self, X: Tensor):
        return self.bottleneck(X)


class Decoder(nn.Module):
    def __init__(
            self,
            img_size: int = 512,                # size of input image.
            dim: int = 64,                      # channels after the first convolution.
            n_conv_stages: int = 1,             # number of convolution stages.
                                                # The input will be downsampled by 2 in each stage.
            dim_mults: List[int] = (1, 2, 4),   # a list of channel multiples in transformer stages.
                                                # The length is the number of transformer stages.
            depths: List[int] = (2, 2, 2),      # number of blocks in each transformer stage.
                                                # The length should be the same as dim_mults.
            window_size: int = 8,               # window size in attention layers.
            legacy_v: int = 4,                  # legacy_v
    ):
        super().__init__()
        assert len(dim_mults) == len(depths)
        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)]
        self.n_heads = dim_mults
        res = img_size // (2 ** n_conv_stages)

        # transformer stages
        self.up_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            self.up_blocks.append(nn.ModuleList([
                BasicLayer2d(
                    dim=self.dims[i] * 2,
                    input_resolution=(res, res),
                    depth=depths[i],
                    num_heads=self.n_heads[i],
                    window_size=window_size,
                    partial=True,
                ),
                PatchResizing2d(
                    in_channels=self.dims[i] * 2,
                    out_channels=self.dims[i-1],
                    up=True,
                ) if i > 0 else nn.Identity(),
            ]))
            res = res // 2

        # convolution stages
        self.conv_up_blocks = nn.ModuleList([])
        for i in range(n_conv_stages):
            self.conv_up_blocks.append(nn.Sequential(
                Upsample(dim * 2, dim * 2, legacy_v=legacy_v),
                nn.GELU(),
            ))

        # last convolution
        self.last_conv = ConvNormAct(dim * 2, 3, kernel_size=1, stride=1, padding=0, activation='tanh')

    def forward(self, X: Tensor, skips: List[Tensor], masks: List[Tensor]):
        for (blk, up), skip, mask in zip(reversed(self.up_blocks), reversed(skips), reversed(masks)):
            X, mask = blk(torch.cat((X, skip), dim=1), mask.float())
            X = up(X)
        for blk in self.conv_up_blocks:
            X = blk(X)
        X = self.last_conv(X)
        return X


@ARCH_REGISTRY.register()
class FRFormerNet(nn.Module):
    def __init__(
            self,
            img_size: int = 512,
            dim: int = 64,
            n_conv_stages: int = 1,
            dim_mults: List[int] = (1, 2, 4),
            encoder_depths: List[int] = (6, 4, 2),
            decoder_depths: List[int] = (2, 2, 2),
            window_size: int = 8,
            bottleneck_window_size: int = 8,
            bottleneck_depth: int = 2,
            legacy_v: int = 4,
            output_ch: int = 3,
    ):
        super().__init__()
        assert len(dim_mults) == len(encoder_depths) == len(decoder_depths)
        self.img_size = img_size
        self.n_conv_stages = n_conv_stages
        self.n_stages = len(dim_mults)
        self.NAF = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                          enc_blk_nums=[1, 1, 1, 16], dec_blk_nums=[1, 1, 1, 1])

        self.first_conv = SCESlayer(out_ch=dim)
        self.encoder = Encoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=encoder_depths,
            window_size=window_size,
            legacy_v=legacy_v,
        )
        self.bottleneck = Bottleneck(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depth=bottleneck_depth,
            window_size=bottleneck_window_size,
        )
        self.decoder = Decoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=decoder_depths,
            window_size=window_size,
            legacy_v=legacy_v,
        )

    def forward(self, X: Tensor):
        pred_flare = self.NAF(X)
        pred_mask = 1. - torch.mean(pred_flare, dim=1, keepdim=True)
        pred_mask.clamp_(0, 1)

        pred_masks = []
        for i in range(self.n_stages):
            pred_masks.append(F.interpolate(pred_mask, size=self.img_size // (2*2**i)))

        X = self.first_conv(X, pred_flare)
        X, skips = self.encoder(X, pred_masks)  # X:nc*256*128*128  skip:[nc*64*512*512, nc*128*256*256, nc*256*128*128]
        X = self.bottleneck(X)

        out = self.decoder(X, skips, pred_masks)
        return out, pred_flare

    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
    #     self.bottleneck.load_state_dict(state_dict['bottleneck'], strict=strict)
    #     self.decoder.load_state_dict(state_dict['decoder'], strict=strict)

    def my_state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            bottleneck=self.bottleneck.state_dict(),
            decoder=self.decoder.state_dict(),
        )


if __name__ == "__main__":
    FRFormer = FRFormerNet(
        dim=64,
        encoder_depths=[6, 4, 2],
        decoder_depths=[2, 2, 2],
        bottleneck_window_size=8,
    )
    dummy_input = torch.randn(1, 3, 512, 512)
    FRFormer(dummy_input)
    print(sum(p.numel() for p in FRFormer.parameters()))
    print(sum(p.numel() for p in FRFormer.encoder.parameters()))
    print(sum(p.numel() for p in FRFormer.bottleneck.parameters()))
    print(sum(p.numel() for p in FRFormer.decoder.parameters()))
    print('=' * 30)

