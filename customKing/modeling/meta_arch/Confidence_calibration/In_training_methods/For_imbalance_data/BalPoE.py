'''
Reference paper: "Balanced Product of Calibrated Experts for Long-Tailed Recognition"
Reference code: "https://github.com/emasa/BalPoE-CalibratedLT"
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.modeling.meta_arch.Image_classification.Resnext import Resnet110
from customKing.modeling.meta_arch.Image_classification.DenseNet import densenet161

# MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, cfg):
        super(MLPClassifier, self).__init__()
        assert len(cfg.DATASETS.INPUTE_SIZE) == 1, "Only supports one-dimensional data!"
        self.fc1 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], cfg.DATASETS.INPUTE_SIZE[0]) 
        self.activation = nn.Sigmoid()  
        self.fc2 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], 2*cfg.DATASETS.INPUTE_SIZE[0]) 

    def _forward_impl(self,x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        predict = self.fc3(out)
        return predict

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        return out

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
    
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class MLP_s(nn.Module):
    def __init__(self,cfg,num_experts=3,reweight_temperature=0.2):
        super(MLP_s,self).__init__()

        self.in_planes = 16
        self.num_classes = cfg.MODEL.NUM_CLASS
        self.num_experts = num_experts
        self.eta = reweight_temperature
        self.MLPs = nn.ModuleList([MLPClassifier(cfg) for _ in range(num_experts)])
        self.linears = nn.ModuleList([NormedLinear(2*cfg.DATASETS.INPUTE_SIZE[0],self.num_classes) for _ in range(num_experts)])
        self.use_experts = list(range(num_experts))

        self.apply(_weights_init)

    def forward(self,x):
        outs = []
        for i in self.use_experts:
            xi = self.MLPs[i](x)
            xi = self.linears[i](xi)
            outs.append(xi)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        multi_logits = torch.stack(outs, dim=0)

        return final_out,multi_logits
    
class ResNet110_s(nn.Module):
    def __init__(self,cfg,num_experts=3,reweight_temperature=0.2):
        super(ResNet110_s,self).__init__()

        self.in_planes = 16
        self.num_classes = cfg.MODEL.NUM_CLASS
        self.num_experts = num_experts
        self.eta = reweight_temperature
        self.backbones = nn.ModuleList([Resnet110(cfg) for _ in range(num_experts)])
        self.linears = nn.ModuleList([NormedLinear(64,self.num_classes) for _ in range(num_experts)])
        self.use_experts = list(range(num_experts))
        self.apply(_weights_init)
    
    def forward(self,x):
        outs = []

        for i in self.use_experts:
            xi = self.backbones[i].forward_backbone(x)
            xi = self.linears[i](xi)
            outs.append(xi)

        final_out = torch.stack(outs, dim=1).mean(dim=1)
        multi_logits = torch.stack(outs, dim=0)
            
        return final_out,multi_logits
    
class DenseNet161_s(nn.Module):
    def __init__(self,cfg,num_experts=3,reweight_temperature=0.2):
        super(DenseNet161_s,self).__init__()

        self.in_planes = 16
        self.num_classes = cfg.MODEL.NUM_CLASS
        self.num_experts = num_experts
        self.eta = reweight_temperature
        self.backbones = nn.ModuleList([densenet161(cfg) for _ in range(num_experts)])
        self.linears = nn.ModuleList([NormedLinear(densenet161(cfg).feat_dim,self.num_classes) for _ in range(num_experts)])
        self.use_experts = list(range(num_experts))
        self.apply(_weights_init)
    
    def forward(self,x):
        outs = []

        for i in self.use_experts:
            xi = self.backbones[i].forward_backbone(x)
            xi = self.linears[i](xi)
            outs.append(xi)

        final_out = torch.stack(outs, dim=1).mean(dim=1)
        multi_logits = torch.stack(outs, dim=0)
            
        return final_out,multi_logits

    
eps = 1e-7 

class BSExpertLoss(nn.Module):
    r"""
    References:
    """

    def __init__(self, cfg, cls_num_list=None, tau_list=(0, 1, 2), eps=1e-9, **kwargs):
        super().__init__()
        self.base_loss = F.cross_entropy

        self.register_buffer('bsce_weight', torch.tensor(cls_num_list).float())
        self.register_buffer('tau_list', torch.tensor(tau_list).float())
        self.num_experts = len(tau_list)
        self.eps = eps

        assert self.num_experts >= 1

    def forward(self, multi_logits, targets):

        assert len(multi_logits.shape) == 3
        assert multi_logits.shape[0] == self.num_experts

        expert_losses = dict()
        loss = 0.0

        for idx in range(self.num_experts):
            adjusted_expert_logits = multi_logits[idx] + self.get_bias_from_index(idx)
            expert_losses[f'loss_e_{idx}'] = expert_loss = self.base_loss(adjusted_expert_logits, targets)
            loss = loss + expert_loss

        loss = loss / self.num_experts
        return loss

    def get_default_bias(self, tau=1):
        prior = self.bsce_weight
        prior = prior / prior.sum()
        log_prior = torch.log(prior + self.eps)
        return tau * log_prior

    def get_bias_from_index(self, e_idx):
        tau = self.tau_list[e_idx]
        return self.get_default_bias(tau)
    

class BalPoE(nn.Module):
    def __init__(self,cfg,cls_num_list,classifier) -> None:
        super().__init__()
        self.classifier = classifier
        self.lossFun = BSExpertLoss(cfg,cls_num_list)
        self.need_epoch = False
        self.output_uncertainty = False
        self.Two_stage = False

    def forward(self,x,y):
        final_out,multi_logits = self.classifier(x)

        if self.training:
            loss = self.lossFun(multi_logits, y)
            return final_out,loss
        else:
            return final_out

@META_ARCH_REGISTRY.register()
def MLP_BalPoE(cfg,cls_num_list):
    classifier = MLP_s(cfg)
    return BalPoE(cfg,cls_num_list,classifier)

@META_ARCH_REGISTRY.register()
def Resnet110_BalPoE(cfg,cls_num_list):
    classifier = ResNet110_s(cfg)
    return BalPoE(cfg,cls_num_list,classifier)

@META_ARCH_REGISTRY.register()
def DenseNet161_BalPoE(cfg,cls_num_list):
    classifier = DenseNet161_s(cfg)
    return BalPoE(cfg,cls_num_list,classifier)