'''
Reference paper: 《Disentangling Label Distribution for Long-tailed Visual Recognition》
'''

import numpy as np
import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn.functional as F 
from customKing.modeling.meta_arch.Image_classification.Resnext import Resnet110
from customKing.modeling.meta_arch.oneD_classification import MLPClassfier
from customKing.modeling.meta_arch.Image_classification.DenseNet import densenet161

class DotProduct_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, use_route=True, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.use_route = use_route

    def forward(self, x):
        x = self.fc(x)
        if self.use_route:
            return x, x
        else:
            return x, None

class LADELoss(nn.Module):
    def __init__(self, num_classes, img_num_per_cls_in_train, remine_lambda=0.1):
        super().__init__()
        self.img_num_per_cls = img_num_per_cls_in_train
        self.train_prior = self.img_num_per_cls / self.img_num_per_cls.sum()

        self.balanced_prior = torch.tensor(1. / num_classes).float().cuda()
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())).cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target):
        """
        y_pred: N x C
        target: N
        """
        self.balanced_prior = self.balanced_prior.to(y_pred.device)
        self.cls_weight = self.cls_weight.to(y_pred.device)

        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.train_prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss

class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes, img_num_per_cls_in_train):
        super().__init__()
        self.img_num_per_cls = img_num_per_cls_in_train
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss


class LADE(nn.Module):
    def __init__(self,cfg,img_num_per_cls_in_train, img_num_per_cls_in_test, classifier, alpha = 0.1):
        super().__init__()
        self.classifier = classifier
        img_num_per_cls_in_train =  torch.tensor(img_num_per_cls_in_train).to(cfg.MODEL.DEVICE)
        img_num_per_cls_in_test =  torch.tensor(img_num_per_cls_in_test).to(cfg.MODEL.DEVICE)
        self.class_head = DotProduct_Classifier(cfg.MODEL.NUM_CLASS,self.classifier.feat_dim)
        self.L_lade_ce = PriorCELoss(cfg.MODEL.NUM_CLASS, img_num_per_cls_in_train)
        self.L_lade = LADELoss(cfg.MODEL.NUM_CLASS, img_num_per_cls_in_train)
        self.alpha = alpha
        self.train_prior = img_num_per_cls_in_train / img_num_per_cls_in_train.sum()
        self.test_prior = img_num_per_cls_in_test / img_num_per_cls_in_test.sum()

        self.need_epoch = False
        self.output_uncertainty = False
        self.Two_stage = False
    
    def forward(self, x, y):
        
        features = self.classifier.forward_backbone(x)
        logits, route_logits = self.class_head(features)
        
        if self.training:
            loss = self.L_lade_ce(logits,y) + self.alpha* self.L_lade(route_logits,y)
            predicts = logits + torch.log(self.train_prior)
            return predicts,loss
        else:
            predicts = logits + torch.log(self.test_prior)
            return predicts
        

@META_ARCH_REGISTRY.register()
def MLP_LADE(cfg,img_num_per_cls_in_train,img_num_per_cls_in_test):
    MLP = MLPClassfier(cfg)
    return LADE(cfg,img_num_per_cls_in_train,img_num_per_cls_in_test,MLP)

@META_ARCH_REGISTRY.register()
def ResNet110_LADE(cfg,img_num_per_cls_in_train,img_num_per_cls_in_test):
    ResNet110 = Resnet110(cfg)
    return LADE(cfg,img_num_per_cls_in_train,img_num_per_cls_in_test,ResNet110)

@META_ARCH_REGISTRY.register()
def DenseNet161_LADE(cfg,img_num_per_cls_in_train,img_num_per_cls_in_test):
    densenet = densenet161(cfg)
    return LADE(cfg,img_num_per_cls_in_train,img_num_per_cls_in_test,densenet)