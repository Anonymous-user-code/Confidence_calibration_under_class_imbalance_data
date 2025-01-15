import torch
import torch.nn as nn
import logging
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.modeling.meta_arch.oneD_classification.MLPClassfier import MLPClassifier
from customKing.modeling.meta_arch.Image_classification.Resnext import Resnet110
from customKing.modeling.meta_arch.Image_classification.DenseNet import densenet161

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).to(output.device)
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class ClassficationAndMDCA(nn.Module):
    def __init__(self, cfg, network, alpha=0.1, beta=1.0, gamma=1.0):
        super(ClassficationAndMDCA, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classification_loss = FocalLoss(gamma=self.gamma)
        self.MDCA = MDCA()
        self.classifier = network

        self.need_epoch = False
        self.output_uncertainty = False
        self.Two_stage = False


    def forward(self, x, targets):
        logits = self.classifier._forward_impl(x)

        if self.training:
            loss_cls = self.classification_loss(logits, targets)
            loss_cal = self.MDCA(logits, targets)
            loss = loss_cls + self.beta * loss_cal
            return logits,loss
        else:
            return logits
    
@META_ARCH_REGISTRY.register()
def MLP_MDCA(cfg):
    MLP = MLPClassifier(cfg)
    return ClassficationAndMDCA(cfg,MLP)

@META_ARCH_REGISTRY.register()
def ResNet110_MDCA(cfg):
    ResNet110 = Resnet110(cfg)
    return ClassficationAndMDCA(cfg,ResNet110)

@META_ARCH_REGISTRY.register()
def DenseNet161_MDCA(cfg):
    DenseNet = densenet161(cfg)
    return ClassficationAndMDCA(cfg,DenseNet)