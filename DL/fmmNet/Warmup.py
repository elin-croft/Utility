import os, sys
import math

class WarmUp:
    def __init__(self, warmup=True) -> None:
        self.warmup = warmup

    def __call__(self, optimizer, base_lr, current_step, type='linear', step=500, eps=1e-8):
            if self.warmup:
                if type == 'linear':
                    factor = min(1.0, (current_step + 1 + eps) / step)
                elif type == 'log':
                    factor = min(1.0, math.log(current_step + 1 + eps, step))
        
                for param in optimizer.param_groups:
                    param['lr'] = base_lr * factor
    
            if current_step > step:
                self.warmup = False
