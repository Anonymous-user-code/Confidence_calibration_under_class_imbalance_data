'''
Reference paper: "Scaling of Class-wise Training Losses for Post-hoc Calibration"
Reference code: "https://github.com/SeungjinJung/SCTL"
'''
import torch
import numpy as np
from scipy import optimize
import torch.nn as nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn.functional as F

def logit_to_confidence(z,y):
    softmaxes = F.softmax(z, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    confidences,resort_index = torch.sort(confidences)
    labels = y[resort_index]
    predictions = predictions[resort_index]
    return confidences,predictions,labels

@META_ARCH_REGISTRY.register()
class CLS(nn.Module):
    def __init__(self, cfg):
        super(CLS, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.n_class = cfg.MODEL.NUM_CLASS
        self.norm = "ND"
        self.step = 0
        self.alpha = 1
        self.beta = 1.5
        self.t = nn.Parameter(torch.ones(1))

        self.need_calibration_train = True
        self.require_iterative_training = True
        self.cfg = cfg

    def forward(self, input, target):

        if isinstance(input,np.ndarray):
            input = torch.from_numpy(input).to(self.t.device)
            target = torch.from_numpy(target).to(self.t.device)

        input = input / self.t

        if self.training:
            losses = torch.zeros(self.n_class)
            for i in range(self.n_class):
                indice = target.eq(i)
                tx = input[indice]
                ty = target[indice]
                if len(tx) == 0:
                    losses[i] = torch.tensor(0.)
                else:
                    losses[i] = self.loss(tx, ty)

            loss = losses.clone().detach()

            if self.norm == 'ND':
                norm = (loss-loss.mean())/loss.std() 
            elif self.norm == 'MM': 
                norm = (loss-loss.min())/(loss.max()-loss.min())
            elif self.norm == 'CM':
                norm = (loss-loss.mean())/(loss.max()-loss.min())

            if self.step == 0:
                self.first = loss.tolist()

            # Optimize alpha and beta
            elif self.step == 1:
                self.optim = False
                self.optimize_scailing_estimator(norm.tolist(), loss.tolist())

            self.step += 1
                    
            # scale loss
            losses *= self.scailing_estimator(norm)
            return losses.mean()
        else:
            confidences,predictions,labels = logit_to_confidence(input,target)
            return confidences,predictions,labels
    
    
    def scailing_estimator(self, x):
        return self.beta/(1+np.exp(-x/self.alpha)) - self.beta/2 + 1
    
    def optimize_scailing_estimator(self, norm, loss):
        def func(x, *args):
            return np.sqrt(((np.array(args[2]) - (x[1]/(1+np.exp(-np.array(args[0])/x[0])) -x[1]/2) * (np.array(args[1])-np.array(args[2])) - np.array(args[1]).mean()) ** 2).sum())
        
        opt = optimize.minimize(func, (self.alpha, self.beta), args=(norm, loss, self.first, self.n_class), method='SLSQP', 
                                bounds=((0.1, np.log(self.n_class)/2),(1.5, 2.0)), options={'disp':False})

        self.alpha, self.beta = opt.x