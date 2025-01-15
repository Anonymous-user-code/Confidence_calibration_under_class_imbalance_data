'''
Reference paper:《Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration》
'''

import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
import torch.nn.functional as F

class Dirichlet_Fun():
    def __init__(self,cfg,model) -> None:
        self.W = model.lin.weight
        self.b = model.lin.bias
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.cfg = cfg

    def lossComput(self,X,Y):
        loss = self.CrossEntropyLoss(X,Y)
        W_sum = 0.
        k = self.cfg.MODEL.NUM_CLASS
        for i in range(k):
            for j in range(k):
                if i != j:
                    W_sum = W_sum + self.W[i][j]**2
        bias_sum = sum([bj**2 for bj in self.b])
        lamda = 0.001
        miu = 0.001
        #loss = loss + lamda/(k*(k-1))*W_sum + miu/k*bias_sum
        loss = loss + k*W_sum + miu*bias_sum
        return loss

class Dirichlet_calibration(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.MODEL.NUM_CLASS,cfg.MODEL.NUM_CLASS)
        self.lin.to(torch.float64)
        self.cfg = cfg
        self.lossfun = Dirichlet_Fun(cfg,self)
        self._init_weight()
        self.need_calibration_train = True
        self.require_iterative_training = True

    def _init_weight(self):
        nn.init.eye_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self,Simple_vector,label_list):
        Simple_vector = softmax(Simple_vector,dim=1,dtype=torch.float64)
        ln = torch.log(Simple_vector)
        Simple_vector = self.lin(ln)

        if self.training:
            #计算损失
            loss = self.lossfun.lossComput(Simple_vector,label_list)
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
def dirichlet_calibration(cfg):
    return Dirichlet_calibration(cfg)

if __name__=="__main__":
    model = Dirichlet_calibration()
    input = torch.tensor([0.91,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
    after_calibration = model(input)
    print(after_calibration)