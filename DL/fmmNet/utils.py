import math
import importlib

import numpy as np
import torch

def channel_shuffle(x:torch.Tensor, groups):
    n, c, h, w = x.size()
    number_per_group = c // groups

    # reshape
    x = x.view(n, groups, number_per_group, h, w)
    x = x.transpose(1, 2).contiguous()

    # reshape back
    x = x.view(n, -1, h, w)
    return x

def bilinear_kernel(in_channels, out_channels, kernel_size):
   factor = (kernel_size + 1) // 2
   if kernel_size % 2 == 1:
       center = factor - 1
   else:
       center = factor - 0.5
   og = np.ogrid[:kernel_size, :kernel_size]
   filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
   weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
   weight[range(in_channels), range(out_channels), :, :] = filt
   return torch.from_numpy(weight)

def calculat_psnr(x, y):
    return 10.0 * torch.log10(31.0**2 / torch.mean((x - y)**2))

def file2dict(path: str):
    moduleName = '.'.join(path.split('.')[0].split('/'))
    module = importlib.import_module(moduleName)
    return module.model