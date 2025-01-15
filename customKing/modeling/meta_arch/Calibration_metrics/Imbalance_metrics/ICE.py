'''
Paper: "Calibration methods in imbalanced binary classification"
Reference code: https://github.com/theoguilbert/imbalanced_calib
'''

import numpy as np
import pandas as pd
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn as nn

def entropy(x):
    return -(x*np.log(x) + (1-x)*np.log(1-x)) / np.log(2)

@META_ARCH_REGISTRY.register()
class ICE_strong(nn.Module):
    def __init__(self,cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "ICE_strong"
    def forward(self, pred_probas, y_true):   
        if pred_probas.shape[0] == 0:
            return np.nan
        else:
            bins = np.linspace(0, 1, self.n_bins + 1)
            dico_ece_bins = {}
            frac_pos = y_true.sum() / y_true.shape[0]
            sum_weights = 0
            for bin_nb in range(len(bins)-1):
                bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
                if bin_end == 1:
                    bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                    
                probas_in_bin, y_corresponding = pred_probas[(pred_probas >= bin_start) * (pred_probas < bin_end)], y_true[(pred_probas >= bin_start) * (pred_probas < bin_end)]
                ece_bin = abs(probas_in_bin.mean() - y_corresponding.mean()) * ((len(probas_in_bin) ** (2*frac_pos)) + (len(probas_in_bin) ** (0.5 + frac_pos))) / 2
                sum_weights += ((len(probas_in_bin) ** (2*frac_pos)) + (len(probas_in_bin) ** (0.5 + frac_pos))) / 2
                dico_ece_bins[bin_nb] = ece_bin
            return sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights


@META_ARCH_REGISTRY.register()
class ICE_soft(nn.Module):
    def __init__(self,cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "ICE_soft"

    def forward(self,pred_probas,y_true):    
        if pred_probas.shape[0] == 0:
            return np.nan
        else:
            bins = np.linspace(0, 1, self.n_bins + 1)
            dico_ece_bins = {}
            frac_pos = y_true.sum() / y_true.shape[0]
            sum_weights = 0
            for bin_nb in range(len(bins)-1):
                bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
                if bin_end == 1:
                    bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                    
                probas_in_bin, y_corresponding = pred_probas[(pred_probas >= bin_start) * (pred_probas < bin_end)], y_true[(pred_probas >= bin_start) * (pred_probas < bin_end)]
                ece_bin = abs(probas_in_bin.mean() - y_corresponding.mean()) * (len(probas_in_bin) ** (0.5 + frac_pos))
                sum_weights += (len(probas_in_bin) ** (0.5 + frac_pos))
                dico_ece_bins[bin_nb] = ece_bin

            return sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights


@META_ARCH_REGISTRY.register()
class ICE_smooth(nn.Module):
    def __init__(self,cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "ICE_smooth"

    def forward(self,pred_probas,y_true):        
        if pred_probas.shape[0] == 0:
            return np.nan
        else:
            bins = np.linspace(0, 1, self.n_bins + 1)
            dico_ece_bins = {}
            frac_pos = y_true.sum() / y_true.shape[0]
            sum_weights = 0
            for bin_nb in range(len(bins)-1):
                bin_start, bin_end = bins[bin_nb], bins[bin_nb + 1]
                if bin_end == 1:
                    bin_end = 1.01 # To ensure selecting the proba = 1 in the following line
                    
                probas_in_bin, y_corresponding = pred_probas[(pred_probas >= bin_start) * (pred_probas < bin_end)], y_true[(pred_probas >= bin_start) * (pred_probas < bin_end)]
                ece_bin = abs(probas_in_bin.mean() - y_corresponding.mean()) * (len(probas_in_bin) ** entropy(frac_pos))
                
                sum_weights += (len(probas_in_bin) ** entropy(frac_pos))
                dico_ece_bins[bin_nb] = ece_bin

            ice_smooth = sum(pd.Series(dico_ece_bins.values()).dropna()) / sum_weights
            return ice_smooth