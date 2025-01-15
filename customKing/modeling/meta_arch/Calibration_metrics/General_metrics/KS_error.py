'''
Paper: "CALIBRATION OF NEURAL NETWORKS USING SPLINES".
Another name for this method: ECCE_MAD, See the paper "Metrics of Calibration for Probabilistic Predictions".
'''

import numpy as np
import torch
import torch.nn.functional as F
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn as nn

def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a

def ks_error(scores, labels):
    # KS stands for Kolmogorov-Smirnov

    # Change to numpy, then this will work
    scores = ensure_numpy (scores)
    labels = ensure_numpy (labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy   = np.cumsum(labels) / nsamples

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    return KS_error_max

@META_ARCH_REGISTRY.register()
class KS_error(nn.Module):
    def __init__(self,cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "KS-error"

    def forward(self,scores, labels):
        return ks_error(scores, labels)