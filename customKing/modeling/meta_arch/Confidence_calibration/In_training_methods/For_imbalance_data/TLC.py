'''
Reference paper: "Trustworthy Long-Tailed Classification"
Reference code: "https://github.com/lblaoke/TLC/tree/master"
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.modeling.meta_arch.oneD_classification.MLPClassfier import MLPClassifier
import torch.nn.init as init
from torch.nn import Parameter
from customKing.modeling.meta_arch.Image_classification.Resnext import Resnet110
from customKing.modeling.meta_arch.Image_classification.DenseNet import densenet161


class MLP_s(nn.Module):
    def __init__(self,cfg,num_experts=2,reweight_temperature=0.2):
        super(MLP_s,self).__init__()

        self.in_planes = 16
        self.num_classes = cfg.MODEL.NUM_CLASS
        self.num_experts = num_experts
        self.eta = reweight_temperature
        self.backbones = nn.ModuleList([MLPClassifier(cfg) for _ in range(num_experts)])
        self.linears = nn.ModuleList([NormedLinear(2*cfg.DATASETS.INPUTE_SIZE[0],self.num_classes) for _ in range(num_experts)])
        self.use_experts = list(range(num_experts))
        self.apply(_weights_init)
    
    def forward(self,x):
        outs = []
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.bool,device=x.device)]

        for i in self.use_experts:
            xi = self.backbones[i]._get_features(x)
            xi = self.linears[i](xi)
            outs.append(xi)

            # evidential
            alpha = torch.exp(xi)+1
            S = alpha.sum(dim=1,keepdim=True)
            b = (alpha-1)/S
            u = self.num_classes/S.squeeze(-1)

            # update w
            if b0 is None:
                C = 0
            else:
                bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
                C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
            b0 = b
            self.w.append(self.w[-1]*u/(1-C))

        # dynamic reweighting
        exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        exp_w = [wi/wi.sum() for wi in exp_w]
        exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        reweighted_outs = [outs[i]*exp_w[i] for i in self.use_experts]
        return sum(reweighted_outs)
    
class ResNet110_s(nn.Module):
    def __init__(self,cfg,num_experts=2,reweight_temperature=0.2):
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
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.bool,device=x.device)]

        for i in self.use_experts:
            xi = self.backbones[i].forward_backbone(x)
            xi = self.linears[i](xi)
            outs.append(xi)

            # evidential
            alpha = torch.exp(xi)+1
            S = alpha.sum(dim=1,keepdim=True)
            b = (alpha-1)/S
            u = self.num_classes/S.squeeze(-1)

            # update w
            if b0 is None:
                C = 0
            else:
                bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
                C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
            b0 = b
            self.w.append(self.w[-1]*u/(1-C))

        # dynamic reweighting
        exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        exp_w = [wi/wi.sum() for wi in exp_w]
        exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        reweighted_outs = [outs[i]*exp_w[i] for i in self.use_experts]
        return sum(reweighted_outs)
    
class DenseNet161_s(nn.Module):
    def __init__(self,cfg,num_experts=2,reweight_temperature=0.2):
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
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.bool,device=x.device)]

        for i in self.use_experts:
            xi = self.backbones[i].forward_backbone(x)
            xi = self.linears[i](xi)
            outs.append(xi)

            # evidential
            alpha = torch.exp(xi)+1
            S = alpha.sum(dim=1,keepdim=True)
            b = (alpha-1)/S
            u = self.num_classes/S.squeeze(-1)

            # update w
            if b0 is None:
                C = 0
            else:
                bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
                C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
            b0 = b
            self.w.append(self.w[-1]*u/(1-C))

        # dynamic reweighting
        exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        exp_w = [wi/wi.sum() for wi in exp_w]
        exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        reweighted_outs = [outs[i]*exp_w[i] for i in self.use_experts]
        return sum(reweighted_outs)

# Resnet classifier
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
class NormedLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return F.normalize(x,dim=1).mm(F.normalize(self.weight,dim=0))


class TLCLoss(nn.Module):
    def __init__(self,cfg,network,cls_num_list=None,max_m=0.5,reweight_epoch=-1,reweight_factor=0.05,annealing=500,tau=0.54):
        super(TLCLoss,self).__init__()
        self.cfg = cfg
        self.classifier = network

        self.reweight_epoch = reweight_epoch
        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = torch.tensor(m_list,dtype=torch.float,requires_grad=False)
        self.m_list = m_list

        if reweight_epoch!=-1:
            idx = 1
            betas = [0,0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights_enabled = None
            self.per_cls_weights_base = self.per_cls_weights_enabled
            self.per_cls_weights_diversity = None
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)

        # save diversity per_cls_weights
        self.T = (reweight_epoch+annealing)/reweight_factor
        self.tau = tau
        self.need_epoch = True
        self.output_uncertainty = True
        self.Two_stage = False


    def get_final_output(self,x,y):
        index = torch.zeros_like(x,dtype=torch.uint8,device=x.device)
        index.scatter_(1,y.data.view(-1,1),1)
        index_float = index.float()
        self.m_list = self.m_list.to(x.device)
        batch_m = torch.matmul(self.m_list[None,:],index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x-30*batch_m
        index = index.bool()
        return torch.exp(torch.where(index,x_m,x))

    def forward(self,x,y,epoch=None):
        x = self.classifier(x)
        extra_info = {
            "num_expert"    : len(self.classifier.logits)   ,
            "logits"        : self.classifier.logits        ,
            'w'             : self.classifier.w
        }
        
        if self.training:
            loss = 0
            for i in range(extra_info["num_expert"]):
                alpha = self.get_final_output(extra_info["logits"][i],y)
                S = alpha.sum(dim=1,keepdim=True)
                l = F.nll_loss(torch.log(alpha)-torch.log(S),y,weight=self.per_cls_weights_base,reduction="none")

                # KL
                yi = F.one_hot(y,num_classes=alpha.shape[1])

                # adjusted parameters of D(p|alpha)
                alpha_tilde = yi+(1-yi)*(alpha+1)
                S_tilde = alpha_tilde.sum(dim=1,keepdim=True)
                kl = torch.lgamma(S_tilde)-torch.lgamma(torch.tensor(alpha_tilde.shape[1]))-torch.lgamma(alpha_tilde).sum(dim=1,keepdim=True) \
                    +((alpha_tilde-1)*(torch.digamma(alpha_tilde)-torch.digamma(S_tilde))).sum(dim=1,keepdim=True)
                l += epoch/self.T*kl.squeeze(-1)

                # diversity
                if self.per_cls_weights_diversity is not None:
                    diversity_temperature = self.per_cls_weights_diversity.view((1,-1))
                    temperature_mean = diversity_temperature.mean().item()
                else:
                    diversity_temperature = 1
                    temperature_mean = 1
                output_dist = F.log_softmax(extra_info["logits"][i]/diversity_temperature,dim=1)
                with torch.no_grad():
                    mean_output_dist = F.softmax(x/diversity_temperature,dim=1)
                l -= 0.01*temperature_mean*temperature_mean*F.kl_div(output_dist,mean_output_dist,reduction="none").sum(dim=1)

                # dynamic engagement
                w = extra_info['w'][i]/extra_info['w'][i].max()
                w = torch.where(w>self.tau,True,False)
                l = (w*l).sum()/w.sum()
                loss += l.mean()

            return x,loss
        else:
            return x,self.classifier.w[-1]

@META_ARCH_REGISTRY.register()
def MLP_TLCLoss(cfg,cls_num_list):
    network = MLP_s(cfg)
    return TLCLoss(cfg,network,cls_num_list)

@META_ARCH_REGISTRY.register()
def ResNet110_TLCLoss(cfg,cls_num_list):
    network = ResNet110_s(cfg)
    return TLCLoss(cfg,network,cls_num_list)

@META_ARCH_REGISTRY.register()
def DenseNet161_TLCLoss(cfg,cls_num_list):
    network = DenseNet161_s(cfg)
    return TLCLoss(cfg,network,cls_num_list)