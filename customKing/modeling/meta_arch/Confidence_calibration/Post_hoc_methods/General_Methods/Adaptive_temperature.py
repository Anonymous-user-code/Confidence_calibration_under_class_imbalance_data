'''
Reference paper:《Adaptive temperature scaling for robust calibration of deep neural networks》
'''

import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.solver.build import build_lossFun
import math
import torch.nn.functional as F

class Adaptive_temperature_scale(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.w_L = nn.Parameter(torch.ones(cfg.MODEL.NUM_CLASS,dtype=torch.float32))
        self.w_H = nn.Parameter(torch.tensor([1.],dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([1.],dtype=torch.float32))
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.lossfun = torch.nn.CrossEntropyLoss()
        self.need_calibration_train = True
        self.require_iterative_training = True

    def forward(self,Simple_vector,label_list):
        LTS = torch.matmul(Simple_vector,self.w_L)
        x = self.softmax(Simple_vector)
        H_hat_list = []
        for xi in x:
            H_hat = 0
            for i in xi:
                H_hat = H_hat + i* torch.log(i)
            H_hat_list.append(H_hat)
        H_hat_list = torch.stack(H_hat_list)
        HTS = self.w_H * (H_hat_list/math.log(self.cfg.MODEL.NUM_CLASS)) + self.b
        a = LTS + HTS
        T = torch.log(1+torch.exp(a)).unsqueeze(1).expand(len(a),self.cfg.MODEL.NUM_CLASS)
        T = T.clamp(torch.finfo(torch.float32).eps,torch.finfo(torch.float32).max)

        Simple_vector = Simple_vector/T

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
def adaptive_temperature_scale(cfg):
    return Adaptive_temperature_scale(cfg)