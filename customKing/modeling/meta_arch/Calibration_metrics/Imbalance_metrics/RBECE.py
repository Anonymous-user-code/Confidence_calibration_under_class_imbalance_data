'''
Paper: "Region-dependent temperature scaling for certainty calibration and application to class-imbalanced token classification"
'''

import numpy as np
import pandas as pd
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn as nn
import torch


@META_ARCH_REGISTRY.register()
class RBECE(nn.Module):
    '''
    Contraharmonic Expected Calibration Error
    '''
    def __init__(self, cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "RBECE"
        self.cfg = cfg

    def forward(self,y_pred, y_true):
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        rbece = 0.
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred >= bin_lower) * (y_pred <= bin_upper)
            prop_in_bin = in_bin.astype(np.float32).mean()
            if in_bin.astype(np.float32).sum() > 40:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                rbece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return rbece