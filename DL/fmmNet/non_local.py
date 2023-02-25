import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Nonlocal(nn.Module):
    def __init__(
        self,
        in_channel,
        reduction=2,
        mode='embedded_gaussian',
        **kwargs
    ):
        super(Nonlocal, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.mode = mode
        self.inner_channel = max(1, in_channel // self.reduction)
        self.g = nn.Conv2d(self.in_channel, self.inner_channel, 1, stride=1, padding=0)
        self.theta = None
        self.phi = None
        self.out = nn.Conv2d(self.inner_channel, self.in_channel, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        if mode != 'gaussian':
            self.theta = nn.Conv2d(self.in_channel, self.inner_channel, 1, stride=1, padding=0)
            self.phi = nn.Conv2d(self.in_channel, self.inner_channel, 1, stride=1, padding=0)

    def gaussian(self, theta_x:torch.Tensor, phi_x:torch.Tensor):
        """
        theta_x: n, c, h, w
        phi_x: n, c, h, w
        """
        n = theta_x.size(0)
        theta_x = theta_x.view(n, self.inner_channel, -1).permute(0, 2, 1) # n, h * w, c
        phi_x = phi_x.view(n, self.inner_channel, -1)
        inner_res = torch.matmul(theta_x, phi_x)
        inner_res = F.softmax(inner_res, dim=-1)
        return inner_res

    def forward(self, x):
        n, c, h, w = x.size()
        g_x = self.g(x).view(n, self.inner_channel, -1)
        g_x = g_x.permute(0, 2, 1) # n, h * w, c
        if self.mode != 'gaussian':
            theta_x = self.theta(x)
            phi_x = self.phi(x)
            if 'gaussian' in self.mode:
                inner_res = self.gaussian(theta_x, phi_x)
        out = torch.matmul(inner_res, g_x)
        out = out.permute(0, 2, 1).contiguous().reshape(n, self.inner_channel, h, w)
        out = self.out(out)
        out = self.bn(out)
        out = self.relu(out)
        return out + x

if __name__ == '__main__':
    x = torch.randn(1, 256, 28, 28)
    net = Nonlocal(256)
    net(x)