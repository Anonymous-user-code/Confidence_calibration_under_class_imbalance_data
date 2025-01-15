'''
Reference paper: 《Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration》
'''

import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.solver.build import build_lossFun


class Parameterized_temperature_scale(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg.MODEL.NUM_CLASS,5).to(torch.float64)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(5,5).to(torch.float64)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(5,1).to(torch.float64)
        self.lossfun = build_lossFun(cfg)
        self.need_calibration_train = True
        self.require_iterative_training = True

    def forward(self,Simple_vector,label_list):
        #温度放缩
        x = self.relu1(self.lin1(Simple_vector))
        x = self.relu2(self.lin2(x))
        x = torch.abs(self.lin3(x))
        #x = x.clamp(torch.finfo(torch.float64).eps,torch.tensor(3.))
        Simple_vector = Simple_vector/x

        #L2正则化
        sum = (self.lin1.weight**2).sum() + (self.lin2.weight**2).sum() + (self.lin1.bias**2).sum() + (self.lin2.bias**2).sum()

        #计算损失
        loss = self.lossfun(Simple_vector,label_list) + 0.01 * sum
        
        softmaxed = None
        return Simple_vector,loss,softmaxed
    
@META_ARCH_REGISTRY.register()
def parameterized_temperature_scale(cfg):
    return Parameterized_temperature_scale(cfg)

