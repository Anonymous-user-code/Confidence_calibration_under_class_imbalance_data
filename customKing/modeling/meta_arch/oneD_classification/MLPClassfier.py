import torch.nn as nn
from ..build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class MLPClassifier(nn.Module):
    def __init__(self, cfg):
        super(MLPClassifier, self).__init__()
        assert len(cfg.DATASETS.INPUTE_SIZE) == 1, "Only supports one-dimensional data!"
        self.fc1 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], cfg.DATASETS.INPUTE_SIZE[0]) 
        self.activation = nn.Sigmoid()  
        self.fc2 = nn.Linear(cfg.DATASETS.INPUTE_SIZE[0], 2*cfg.DATASETS.INPUTE_SIZE[0]) 
        self.fc3 = nn.Linear(2*cfg.DATASETS.INPUTE_SIZE[0], cfg.MODEL.NUM_CLASS)
        self.lossFun = nn.CrossEntropyLoss() 
        self.feat_dim = 2*cfg.DATASETS.INPUTE_SIZE[0]

    def _forward_impl(self,x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        predict = self.fc3(out)
        return predict

    def _get_features(self,x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        return out
    
    def forward_backbone(self,x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        return out
    
    def forward_head(self,x):
        predict = self.fc3(x)
        return predict

    def forward(self, x,label):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        predict = self.fc3(out)

        if self.training:
            loss = self.lossFun(predict,label)
            return predict,loss
        else:
            return predict
    
