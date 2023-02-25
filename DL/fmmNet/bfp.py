import os, sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .non_local import Nonlocal

class BFP(nn.Module):
    def __init__(
        self,
        in_channel,
        number_level: int,
        refine_level: int,
        refine_type: str = 'nonlocal'
    ) -> None:
        super(BFP, self).__init__()
        assert refine_type in ('conv', 'nonlocal')
        self.number_level = number_level
        self.refine_level = refine_level
        self.refine_type = refine_type
        if self.refine_type == 'nonlocal':
            self.refine = Nonlocal(in_channel)
        else:
            self.refine = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
    
    def forward(self, inputs:List[torch.Tensor]):
        assert len(inputs) == self.number_level
        feats = []
        middle_size = inputs[self.refine_level].size()[2:]
        for i in range(self.number_level):
            if i < self.refine_level:
                middle = F.interpolate(inputs[i], size=middle_size, mode='nearest')
            else:
                middle = F.adaptive_max_pool2d(inputs[i], output_size=middle_size)
            feats.append(middle)
        
        blf = sum(feats) / self.number_level
        balanced = self.refine(blf)

        res = []
        for i in range(self.number_level):
            if i < self.refine_level:
                out = F.adaptive_max_pool2d(balanced, output_size=inputs[i].size()[2:])
            else:
                out = F.interpolate(balanced, size=inputs[i].size()[2:], mode='nearest')
            res.append(inputs[i] + out)
        return res

def test():
    import math
    inputs = [torch.randn(1, 256, int(14 * math.pow(2, i)), int(14 * math.pow(2, i))) for i in range(4)]
    maxe = -1e-8
    for i in inputs:
        maxe = max(torch.max(i).item(), maxe)
    print(maxe)

    #model = BFP(256, 4, 2)
    #model(inputs)

if __name__ == '__main__':
    test()
