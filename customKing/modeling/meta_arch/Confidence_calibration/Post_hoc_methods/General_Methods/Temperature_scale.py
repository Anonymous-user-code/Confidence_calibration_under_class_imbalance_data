'''
Reference paper:《On calibration of modern neural networks》
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
        self.temperature = nn.Parameter(torch.ones(1,dtype=torch.float64)*1.5)
        if use_ece:
            self.lossfun = ECE_LOSS(n_bins=15)
        else:
            self.lossfun = build_lossFun(cfg)
        self.need_calibration_train = True
        self.require_iterative_training = True

    def forward(self,Simple_vector,label_list):
        #温度放缩
        Simple_vector = Simple_vector/self.temperature

        if self.training:
            #计算损失
            loss = self.lossfun(Simple_vector,label_list)
            return loss
        else:
            prob = F.softmax(Simple_vector, dim=1)
            confidences, predictions = torch.max(prob, 1)
            hits = predictions.eq(label_list)
            confidences,resort_index = torch.sort(confidences)
            hits = hits[resort_index]
            cali_confidence = confidences.cpu().detach().numpy()
            hits = hits.cpu().detach().numpy()
            return cali_confidence,hits,True
    
@META_ARCH_REGISTRY.register()
def temperature_scale_cross_entropy(cfg):
    return Temperature_scale(cfg)

@META_ARCH_REGISTRY.register()
def temperature_scale_with_ece(cfg):
    return Temperature_scale(cfg,use_ece=True)

if __name__=="__main__":
    model = Temperature_scale()
    input = torch.tensor([0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
    after_calibration = model(input)
    print(after_calibration)
