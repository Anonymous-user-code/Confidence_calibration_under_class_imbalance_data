'''
Reference paper:《Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning》
Official code: https://github.com/zhang64-llnl/Mix-n-Match-Calibration
'''

import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from torch.nn.functional import softmax
import json
import torch.nn.functional as F

class Mix_n_Match(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.temperature = None
        self.cfg = cfg
        self.w = nn.Parameter(torch.tensor([1.,0.,0.]))
        torch.clamp(self.w,0,1)
        
        self.lossfun = torch.nn.MSELoss()
        self.need_calibration_train = True    
        self.require_iterative_training = True
    
    def get_T(self,file_path):
        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line))
        for i in data:
            if 'temperature' in i.keys():
                t = i["temperature"]
        return torch.tensor(t).cuda()

    def forward(self,Simple_vector,label_list):
        one_hot_labels = torch.nn.functional.one_hot(label_list, num_classes=self.cfg.MODEL.NUM_CLASS).double()

        if self.temperature == None:
            file_list = self.cfg.MODEL.OUTPUT_DIR.split("/")
            file_list[-2] = "temperature_scale_with_ece"
            file_path = "/".join(file_list) + "Final_logging.json"
            self.temperature = self.get_T(file_path)
            self.temperature.requires_grad_ = False
        
        self.w.data = self.w.clamp(0.,1.)

        p1 = softmax(Simple_vector,dim=1,dtype=torch.float64)
        Simple_vector = Simple_vector / self.temperature
        p0 = softmax(Simple_vector,dim=1,dtype=torch.float64)
        p2 = torch.ones_like(p0) / self.cfg.MODEL.NUM_CLASS
        p = self.w[0] * p0 + self.w[1] * p1 + self.w[2] * p2
        p = p/torch.sum(p,dim=1,keepdim=True)

        if self.training:
            loss = self.lossfun(p,one_hot_labels) + torch.abs(torch.sum(self.w)-1)
            return loss
        else:
            #prob = F.softmax(p, dim=1)
            confidences, predictions = torch.max(p, 1)
            hits = predictions.eq(label_list)
            confidences,resort_index = torch.sort(confidences)
            hits = hits[resort_index]
            cali_confidence = confidences.cpu().detach().numpy()
            hits = hits.cpu().detach().numpy()
            return cali_confidence,hits,True
    
@META_ARCH_REGISTRY.register()
def mix_n_match(cfg):
    return Mix_n_Match(cfg)