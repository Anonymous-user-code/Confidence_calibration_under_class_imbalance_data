'''
Paper: Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import relplot as rp
    
@META_ARCH_REGISTRY.register()
class SmoothingECE(nn.Module):
    def __init__(self, cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "smECE"

    def forward(self,confidences,hits):
        return rp.smECE(confidences,hits)