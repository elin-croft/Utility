import os, sys

import numpy as np
import torch
from torch._C import GraphExecutorState
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

def channel_shuffle(x:torch.Tensor, groups):
    b, c, h, w = x.data.size()
    number_per_group = c // groups

    #reshape
    x = x.view(b, groups, number_per_group, h, w)

    x = x.transpose(1,2).contiguous()

    #reshape back
    x = x.view(b, -1, h, w)
    return x

class downSample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size ,stride=2, groups=1):
        super(downSample, self).__init__()
        self.branch1 = self.make_layers('pooling', in_channel, out_channel, 1, stride=1, padding=0, groups=1)
        self.branch2 = self.make_layers('conv', in_channel, out_channel, kernel_size, stride=2, padding=1, groups=1)
        self.branch3 = self.make_layers('sse', in_channel, out_channel, 1, stride=1, padding=0, groups=1)

    def make_layers(self, key, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1):
        assert key in ('pooling', 'conv', 'sse')
        layers = None
        if key == 'pooling':
            layers = [
                nn.AvgPool2d(2),
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.BatchNorm2d(out_channel)
            ]
        elif key == 'conv':
            layers = [
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.BatchNorm2d(out_channel)
            ]
        else:
            layers = [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.Sigmoid()
            ]
        return nn.Sequential(*layers) if not layers is None else None

    def forward(self, x):
        pooling = self.branch1(x)
        conv = self.branch2(x)
        sse = self.branch3(x)
        out = sse * (pooling + conv)
        out = F.silu(out, inplace=True)
        return out


class fusionBlack(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size ,stride=2, padding=1, groups=1, sse=True):
        super(fusionBlack, self).__init__()
        self.groups = groups
        self.sse = sse
        self.branch1 = self.make_layers('pooling', in_channel, out_channel, 1, groups=2)
        self.branch2 = self.make_layers('conv', in_channel, out_channel, kernel_size, 
                            stride=stride, padding=padding, groups=groups)
        self.branch3 = self.make_layers('sse', in_channel, out_channel, 1, groups=groups)
    
    def make_layers(self, key, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1):
        assert key in ('pooling', 'conv', 'sse')
        layers = None
        if key == 'pooling':
            layers = [
                nn.AvgPool2d(2),
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.BatchNorm2d(out_channel)
            ]
        elif key == 'conv':
            layers = [
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.BatchNorm2d(out_channel)
            ]
        else:
            layers = [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.Sigmoid()
            ]
        return nn.Sequential(*layers) if not layers is None else None

    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = channel_shuffle(out, self.groups)
        pooling = self.branch1(out)
        conv = self.branch2(out)
        tmp = pooling + conv
        if self.sse:
            sse = self.branch3(out)
            tmp = sse * tmp
        out = tmp
        out = F.silu(out, inplace=True)
        return out

class parBlock(nn.Module):
    def __init__(self, in_channel, out_channel, groups=1):
        super(parBlock, self).__init__()
        self.groups = groups
        self.normIndex = 0
        self.branch1 = self.make_layers('conv', in_channel, out_channel, 1, groups=groups)
        self.branch2 = self.make_layers('conv', in_channel, out_channel, 3, padding=1,groups=groups)
        self.branch3 = self.make_layers('sse', in_channel, out_channel, 1, groups=groups)

    def make_layers(self, key, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1):
        assert key in ('conv', 'sse')
        layers = None
        if key == 'conv':
            layers = [
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.BatchNorm2d(out_channel)
            ]
        else:
            layers = [
                nn.BatchNorm2d(in_channel),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
                nn.Sigmoid()
            ]
        return nn.Sequential(*layers) if not layers is None else None

    def forward(self, x):
        conv1 = self.branch1(x)
        conv3 = self.branch2(x)
        for index, layer in enumerate(self.branch3):
            se = layer(x)
            if index == self.normIndex:
                norm = se
        se = norm * se
        out = conv1 + conv3 + se
        out = F.silu(out, inplace=True)
        return out

class parNet(nn.Module):
    def __init__(self):
        super(parNet, self).__init__()
def test():
    block = fusionBlack(28, 28, 3, groups=4)
    x = torch.randn(1, 14, 128, 128)
    block(x, x)
if __name__ == '__main__':
    test()