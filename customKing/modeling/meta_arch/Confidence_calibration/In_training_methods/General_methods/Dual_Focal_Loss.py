'''
Implementation of Dual Focal Loss.
Reference:
[1]  Tao, Linwei, Minjing Dong, and Chang Xu. "Dual Focal Loss for Calibration." 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from customKing.modeling.meta_arch.Image_classification.Resnext import Resnet110
from customKing.modeling.meta_arch.Image_classification.Resnext import Wide_resnet34_2
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY


class DualFocalLoss(nn.Module):
    def __init__(self, cfg,network):
        super(DualFocalLoss, self).__init__()
        self.gamma = 5.0   #See Table 13 in the paper
        self.classifier = network
        self.need_epoch = False
        self.output_uncertainty = False

    def forward(self, input, target):
        logits = self.classifier._forward_impl(input)
        
        if self.training:
            target = target.view(-1,1)
            logp_k = F.log_softmax(logits, dim=1)
            softmax_logits = logp_k.exp()
            logp_k = logp_k.gather(1, target)
            logp_k = logp_k.view(-1)
            p_k = logp_k.exp()  # p_k: probility at target label
            p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
            p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()
            loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k
            return logits,loss.mean()
        else:
            return logits

@META_ARCH_REGISTRY.register()
def Resnet110_DualFocalLoss(cfg):
    network = Resnet110(cfg)
    return DualFocalLoss(cfg,network)

@META_ARCH_REGISTRY.register()
def Wide_resnet34_2_DualFocalLoss(cfg):
    network = Wide_resnet34_2(cfg)
    return DualFocalLoss(cfg,network)