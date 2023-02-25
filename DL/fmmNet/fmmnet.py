import os, sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .bfp import BFP
import fmmNet.config as cf
from .utils import channel_shuffle, bilinear_kernel

upsample_mode = cf.upsample_mode
align_corners = cf.align_corners

class convBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, norm=True, act=True, **kwargs) -> None:
        super(convBlock, self).__init__()
        self.act = nn.PReLU() if act else None
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs)
        self.norm = nn.BatchNorm2d(out_channel, eps=0.001) if norm else None

    def forward(self, input, **kwargs):
        out = self.conv(input)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mapping=False, groups=1) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = convBlock(in_channel, 2 * in_channel, 1, groups=groups)
        self.conv2 = convBlock(2 * in_channel, 2 * in_channel, 3, stride=1, padding=1, groups=groups)
        self.conv3 = convBlock(2 * in_channel, in_channel, 1, act=False, groups=groups)
        self.mapping = mapping
        self.act = nn.PReLU()
        if mapping or in_channel != out_channel:
            self.mapping = True
            self.expand = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, groups=groups),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, input, **kwagrs):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + input
        if self.mapping:
            out = self.expand(out)
        return self.act(out)

class fusionBlock(nn.Module):
    order = {
        'upsample':1,
        'transpose':2,
        'pixel':3
    }
    def __init__(self, in_channel, branchs=None, use_mode=0, **kwargs) -> None:
        super(fusionBlock, self).__init__()
        self.act = nn.PReLU()
        self.names = []
        self.use_mode = use_mode
        out_channel = len(branchs) * in_channel
        for branch in branchs:
            i = self.order[branch]
            name = f'branch{i}'
            self.names.append(i)
            self.add_module(name, self.make_layers(branch, in_channel, in_channel, **kwargs))
        self.names.sort()
        self.sse = self.make_layers('sse', in_channel, out_channel, **kwargs)
        # self.fusion = nn.Conv2d(out_channel, in_channel, 1, stride=1, padding=0)
        self.fusion = convBlock(out_channel, out_channel, 3, stride=1, padding=1)
        self.out = nn.Conv2d(out_channel, in_channel, 1, stride=1, padding=0)

    def make_layers(self, mode, in_channel, out_channel, **kwargs):
        assert mode in ("upsample", "transpose", "sse", "pixel")
        layers = None
        if mode == "upsample":
            layers = [
                nn.Upsample(scale_factor=2, mode=upsample_mode[self.use_mode], align_corners=align_corners[self.use_mode])
            ]
        elif mode == "transpose":
            layers = [
                nn.ConvTranspose2d(in_channel, in_channel, 4, stride=2, padding=1, bias=False)
            ]
            layers[0].weight.data = bilinear_kernel(in_channel, in_channel, 4)
        elif mode == "pixel":
            layers = [
                nn.Conv2d(in_channel, in_channel * 2**2, 3, stride=1, padding=1, **kwargs),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        else:
            layers = [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channel, out_channel, 1, stride=1, **kwargs),
                nn.Sigmoid()
            ]

        return nn.Sequential(*layers)

    def forward(self, input):
        cached = []
        groups = len(self.names)
        for i in self.names:
            layer = getattr(self, f'branch{i}')
            cached.append(layer(input))
        out = torch.cat(cached, dim=1)
        se = self.sse(input)
        # out = out * se
        # out = channel_shuffle(out, groups)
        ########## new ###########
        out = channel_shuffle(out, groups)
        out = self.fusion(out)
        se = self.sse(input)
        out = out * se
        #new method
        #out = self.fusion(out)
        out = self.out(out)
        return self.act(out)

class Upsample(nn.Module):
    def __init__(self, in_channel, upsample_type, branchs=None, use_mode=0) -> None:
        """
        upsample_type: upsample policy
        can be upsample, tanspose pixel or fusion
        default: upsample
        """
        super(Upsample, self).__init__()
        assert upsample_type in ('fusion', 'upsample', 'transpose', 'pixel')
        self.upsample_type = upsample_type
        if upsample_type == 'upsample':
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode[use_mode], align_corners=align_corners[use_mode])
        elif upsample_type == 'fusion':
            self.upsample = fusionBlock(in_channel, branchs, use_mode=use_mode)
        elif upsample_type == 'pixel':
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * 2**2, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_channel, in_channel, 4, stride=2, padding=1, bias=False)
            self.upsample.weight.data = bilinear_kernel(in_channel, in_channel, 4)

    def forward(self, input, **kwargs):
        res = self.upsample(input)
        return res

class fmmNet(nn.Module):
    def __init__(self, 
        in_channel: int, 
        out_channel: int, 
        number_out: int,
        upsample_type: str = 'fusion', 
        branchs: Tuple[str, str, str] = None,
        refine_type='nonlocal',
        start_stage=1,
        interpolate_mode='nearest',
        groups=1
    ) -> None:
        """
        """
        if upsample_type == 'fusion':
            assert branchs is not None
        super(fmmNet, self).__init__()
        self.refine_type = refine_type
        self.inner_layer= nn.ModuleList()
        self.refine = nn.ModuleList()
        self.between = nn.ModuleList()
        self.names = []
        use_mode = upsample_mode.index(interpolate_mode)
        print(upsample_mode[use_mode])
        stages = number_out + start_stage
        self.stem = nn.Sequential(
            ResBlock(in_channel, in_channel),
            ResBlock(in_channel, in_channel),
            ResBlock(in_channel, in_channel),
            ResBlock(in_channel, in_channel),
            ResBlock(in_channel, in_channel),
            convBlock(in_channel, in_channel, 3, act=False, stride=1, padding=1)
        )
        self.bfp = BFP(in_channel, number_out, 1)
        self.upsamples = nn.ModuleList()
        for i in range(stages):
            self.upsamples.append(Upsample(in_channel, upsample_type=upsample_type, branchs=branchs, use_mode=use_mode))
            if i >= start_stage:
                self.between.append(convBlock(in_channel, in_channel, 3,norm=False, act=False, stride=1, padding=1))
                self.refine.append(nn.Conv2d(in_channel, out_channel[i - start_stage], 1))
       
    def forward_plain(self, x:torch.Tensor):
        shortcut = x
        x = self.stem(x)
        x = x + shortcut
        feats = []
        for i, layer in enumerate(self.upsamples):
            x = layer(x)
            feats.append(x)

        # feats from low resolution to high resolution
        for i in range(len(feats)):
            feats[i] = self.between[i](feats[i])
            if i > 0:
                feats[i] += F.interpolate(feats[i - 1], scale_factor=2, mode='nearest')
        res = feats
        for i in range(len(res)):
            res[i] = self.refine[i](res[i])
        return res

    def forward_nonlocal(self, x:torch.Tensor):
        shortcut = x
        x = self.stem(x)
        x = x + shortcut
        feats = []
        for i, layer in enumerate(self.upsamples):
            x = layer(x)
            feats.append(x)
        # feats from low resolution to high resolution
        for i in range(len(feats)):
            feats[i] = self.between[i](feats[i])
            if i > 0:
                feats[i] += F.interpolate(feats[i - 1], scale_factor=2, mode='nearest')
        res = self.bfp(feats)
        for i in range(len(res)):
            res[i] = self.refine[i](res[i])

        return res

    def forward(self, x:torch.Tensor):
        if self.refine_type == 'plain':
            res = self.forward_plain(x)
        elif self.refine_type == 'bfp':
            res = self.forward_nonlocal(x)
        res.reverse()
        return res

def test(env=None):
    device = 'cuda:0'
    x = torch.randn(1, 256, 7, 7).cuda(device=device)
    n, c, h, w = x.size()
    if env is None:
        block = fmmNet(256, [2048, 1024, 512, 256], 4, upsample_type='fusion', branchs=('pixel', 'transpose','upsample'),refine_type='plain', start_stage=0).cuda(device=device)
    else:
        block = fmmNet(**env).cuda(device)
    # block = fusionBlock(256).cuda(device=device)
    target_size = (n, c, 2 * h, 2 * w)
    res = block(x)
    for i in res:
        print(i.size())

if __name__ == '__main__':
    test()
