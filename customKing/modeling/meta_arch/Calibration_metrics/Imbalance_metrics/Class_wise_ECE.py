'''
Paper: "Class-wise Calibration: A Case Study on COVID-19 Hate Speech"
'''


import numpy as np
import pandas as pd
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn as nn
import torch.nn.functional as F
import torch

def logit_to_confidence(z,y):
    softmaxes = F.softmax(z, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    hits = predictions.eq(y)
    confidences,resort_index = torch.sort(confidences)
    hits = hits[resort_index]
    labels = y[resort_index]
    confidences = confidences.cpu().detach().numpy()
    hits = hits.cpu().detach().numpy().astype(int)
    labels = labels.cpu().detach().numpy().astype(int)
    return confidences,hits,labels

@META_ARCH_REGISTRY.register()
class CECE(nn.Module):
    '''
    Contraharmonic Expected Calibration Error
    '''
    def __init__(self, cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = False
        self.plot_name = "CECE"
        self.cfg = cfg

    def forward(self,confidences,hits,labels):
        cece_numerator = 0.
        cece_denominator = 0.
        for i in range(self.cfg.MODEL.NUM_CLASS):
            index = labels == i
            class_confidences = confidences[index]
            class_hits = hits[index]

            mass_in_bin = len(class_confidences)//self.n_bins
            ece = 0.
            for i in range(self.n_bins):
                Ps = class_confidences[i*mass_in_bin:(i+1)*mass_in_bin].mean()
                acc_in_bin = class_hits[i*mass_in_bin:(i+1)*mass_in_bin].mean()
                ece += np.abs(Ps - acc_in_bin)/self.n_bins

            cece_numerator = cece_numerator + ece**2
            cece_denominator = cece_denominator + ece
        cece = cece_numerator/cece_denominator
        return cece
    
@META_ARCH_REGISTRY.register()
class MSECE(nn.Module):
    '''
    Macro Subset Expected Calibration Error
    '''
    def __init__(self, cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = False
        self.plot_name = "MSECE"
        self.cfg = cfg

    def forward(self,confidences,hits,labels):
        cece_numerator = 0.
        for i in range(self.cfg.MODEL.NUM_CLASS):
            index = labels == i
            class_confidences = confidences[index]
            class_hits = hits[index]

            mass_in_bin = len(class_confidences)//self.n_bins
            ece = 0.
            for i in range(self.n_bins):
                Ps = class_confidences[i*mass_in_bin:(i+1)*mass_in_bin].mean()
                acc_in_bin = class_hits[i*mass_in_bin:(i+1)*mass_in_bin].mean()
                ece += np.abs(Ps - acc_in_bin)/self.n_bins

            cece_numerator = cece_numerator +  ece

        cece = cece_numerator/self.cfg.MODEL.NUM_CLASS
        return cece
    
@META_ARCH_REGISTRY.register()
class WSECE(nn.Module):
    '''
    Weighted Subset Expected Calibration Error
    '''
    def __init__(self, cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = False
        self.plot_name = "WSECE"
        self.cfg = cfg

    def forward(self,confidences,hits,labels):
        cece_numerator = 0.
        for i in range(self.cfg.MODEL.NUM_CLASS):
            index = labels == i
            class_confidences = confidences[index]
            class_hits = hits[index]

            mass_in_bin = len(class_confidences)//self.n_bins
            ece = 0.
            for i in range(self.n_bins):
                Ps = class_confidences[i*mass_in_bin:(i+1)*mass_in_bin].mean()
                acc_in_bin = class_hits[i*mass_in_bin:(i+1)*mass_in_bin].mean()
                ece += np.abs(Ps - acc_in_bin)/self.n_bins

            cece_numerator = cece_numerator +  ece*len(class_confidences)

        cece = cece_numerator/len(confidences)
        return cece

            
        
