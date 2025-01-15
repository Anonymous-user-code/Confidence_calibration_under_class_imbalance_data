'''
Reference paper:
《Class-distribution-aware calibration for long-tailed visual recognition》
《Estimating model performance under domain shifts with class-specific confidence scores》
《On the role of dataset quality and heterogeneity in model confidence》
'''

import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.solver.build import build_lossFun
import torch.nn.functional as F

class ECE_LOSS(nn.Module):
    def __init__(self,n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(self,y_pred, y_true):
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes = F.softmax(y_pred, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(y_true)

        ece = torch.zeros(1, device=y_pred.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class Temperature_scale(nn.Module):
    def __init__(self,cfg,use_ece=False):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(cfg.MODEL.NUM_ClASSES,dtype=torch.float64)*1.5)

        if use_ece:
            self.lossfun = ECE_LOSS(n_bins=15)
        else:
            self.lossfun = build_lossFun(cfg)
        self.need_calibration_train = True
        self.require_iterative_training = True
        self.cfg = cfg
        self.grads = []
        self.track_T = 0

    def forward(self,Simple_vector,label_list):
        #温度放缩
        Simple_vector = Simple_vector/self.temperature

        if self.training:
            #计算损失
            loss = self.lossfun(Simple_vector,label_list)
        
            softmaxed = None
            return Simple_vector,loss,softmaxed
        else:
            return Simple_vector
        
    def compute_gradient_contribution(self,Simple_vector,label_list,track_T = 0):
        track_T = self.track_T
        gradsIter = []
        for i in range(self.cfg.MODEL.NUM_ClASSES):
            index = (label_list==i)
            Samples = Simple_vector[index]
            labels = label_list[index]
            Samples = Samples/self.temperature
            loss = self.lossfun(Samples,labels)
            loss.backward()
            gradsIter.append(abs(self.temperature.grad[track_T].clone().detach().item()))

        # 在参数上注册这个函数
        self.grads.append(gradsIter)

    
@META_ARCH_REGISTRY.register()
def Class_specific_temperature_scale_cross_entropy(cfg):
    return Temperature_scale(cfg)

@META_ARCH_REGISTRY.register()
def Class_specific_temperature_scale_with_ece(cfg):
    return Temperature_scale(cfg,use_ece=True)