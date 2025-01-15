'''
Reference paper: "Multi-Head Multi-Loss Model Calibration"
Reference code: "https://github.com/agaldran/mhml_calibration/tree/main"
'''

import torch
import torch.nn as nn
import sys
import numpy as np
import random
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
from customKing.modeling.meta_arch.Image_classification.Resnext import Resnet110
from customKing.modeling.meta_arch.oneD_classification.MLPClassfier import MLPClassifier
from customKing.modeling.meta_arch.Image_classification.DenseNet import densenet161

# # MLP classifier
# class MLPClassifier(nn.Module):
#     def __init__(self, cfg):
#         super(MLPClassifier, self).__init__()
#         assert len(cfg.DATASETS.INPUTE_SIZE) == 1, "Only supports one-dimensional data!"
#         self.fc1 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], cfg.DATASETS.INPUTE_SIZE[0]) 
#         self.activation = nn.Sigmoid()  
#         self.fc2 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], 2*cfg.DATASETS.INPUTE_SIZE[0]) 

#     def _forward_impl(self,x):
#         out = self.activation(self.fc1(x))
#         out = self.activation(self.fc2(out))
#         predict = self.fc3(out)
#         return predict

#     def forward(self, x):
#         out = self.activation(self.fc1(x))
#         out = self.activation(self.fc2(out))
#         return out

class MultiHeadModel(nn.Module):
    def __init__(self,cfg, model_name, num_heads):
        super(MultiHeadModel, self).__init__()
        self.model_name = model_name
        self.num_classes = cfg.MODEL.NUM_CLASS
        self.num_heads = num_heads
        self.lossFun = torch.nn.functional.cross_entropy
        self.need_epoch = False
        self.output_uncertainty = False
        self.Two_stage = False

        if model_name=='MLP':
            self.classifier = MLPClassifier(cfg)
            self.heads = nn.ModuleList([nn.Linear(2*cfg.DATASET.INPUTE_SIZE[0], self.num_classes) for _ in range(self.num_heads)])
        elif model_name=='resnet110':
            self.classifier = Resnet110(cfg)
            self.heads = nn.ModuleList([nn.Linear(64, self.num_classes) for _ in range(self.num_heads)])
        elif model_name=='Densenet161':
            self.classifier = densenet161(cfg)
            self.heads = nn.ModuleList([nn.Linear(self.classifier.feat_dim, self.num_classes) for _ in range(self.num_heads)])
        else: sys.exit('model not defined')

        

        self.ce_weights = []
        classes = np.arange(self.num_classes)
        random.shuffle(classes)
        more_weighted_classes_per_head = np.array_split(classes, num_heads)
        for c in more_weighted_classes_per_head:
            # weights for this head are 2 for classes in c and 1/2 for classes that are not in c
            w = [2 if i in c else 1/2 for i in np.arange(self.num_classes)]
            w = [num_heads if i in c else 1/num_heads for i in np.arange(self.num_classes)]
            self.ce_weights.append(w)

    def forward(self, x,labels):
        x = self.classifier.forward_backbone(x)
        head_predictions = [self.heads[i](x) for i in range(self.num_heads)]
        out = torch.stack(head_predictions, dim=-2)  # batch_size x num_heads x num_classes
        if self.training:

            weights = [torch.tensor(w, dtype=torch.float32).to(x.device) for w in self.ce_weights]
            loss_class = torch.tensor(0., dtype=torch.float32).to(x.device)
            for i in range(self.num_heads):
                loss_class = loss_class + self.lossFun(out[:, i, :], labels, weights[i])
            overall_loss = self.lossFun(out.mean(dim=1), labels)
            loss = (loss_class+overall_loss)/(self.num_heads+1)
            return torch.mean(out, dim=-2),loss  # batch_size x num_heads x num_classes
        else:
            # mean over heads, return just the logits and the user takes care of softmaxing, req. for Temp. Scaling
            return torch.mean(out, dim=-2)
        
@META_ARCH_REGISTRY.register()
def MLP_MHML_2(cfg):
    return MultiHeadModel(cfg,"MLP",2)

@META_ARCH_REGISTRY.register()
def MLP_MHML_4(cfg):
    return MultiHeadModel(cfg,"MLP",4)

@META_ARCH_REGISTRY.register()
def ResNet110_MHML_2(cfg):
    return MultiHeadModel(cfg,"resnet110",2)

@META_ARCH_REGISTRY.register()
def ResNet110_MHML_4(cfg):
    return MultiHeadModel(cfg,"resnet110",4)

@META_ARCH_REGISTRY.register()
def DenseNet161_MHML_2(cfg):
    return MultiHeadModel(cfg,"Densenet161",2)

@META_ARCH_REGISTRY.register()
def DenseNet161_MHML_4(cfg):
    return MultiHeadModel(cfg,"Densenet161",4)