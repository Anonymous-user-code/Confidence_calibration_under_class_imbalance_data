'''
Reference paper: "Calibration methods in imbalanced binary classification"
Reference code: https://github.com/theoguilbert/imbalanced_calib/blob/main/calibration_model.py
'''

import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import numpy as np


def adjust_posterior_prob_to_new_prior(beta, p_s):
    p_adjusted = (beta * p_s) / (beta * p_s - p_s + 1)
    return p_adjusted

def cap_proba(p):
    if p > 1:
        return 1
    elif p < 0:
        return 0
    else:
        return p

@META_ARCH_REGISTRY.register()
class AUB_Platt(nn.Module):
    '''
    Adjust undersampling baise + Platt
    Note: only used for binary classification of downsampling training
    '''
    def __init__(self,cfg,beta):
        '''
        beta = train_N_pos/train_N_neg
        '''
        super().__init__()
        self.cfg = cfg
        self.beta = beta
        self.lambdaa = None
        self.need_calibration_train = True
        self.require_iterative_training = False

    def forward(self,Simple_vector,label_list):
        prob = F.softmax(Simple_vector, dim=1)
        confidences, predictions = torch.max(prob, 1)
        confidences,resort_index = torch.sort(confidences)
        predictions = predictions[resort_index]
        labels = label_list[resort_index]
        confidences = confidences.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        if self.training:
            '''
            Using the validation set to learn the hyperparameter lambda
            '''
            preds_adjusted_val = adjust_posterior_prob_to_new_prior(self.beta, confidences)
            all_loss = {}
            for lambdaa in np.linspace(0, 1, 21):
                calibration_method = LogisticRegression(penalty=None)
                confidences_w = preds_adjusted_val*lambdaa + confidences*(1-lambdaa)
                confidences_w = np.expand_dims(confidences_w, axis=1)
                calibration_method.fit(confidences_w, labels)
                loss = mean_squared_error(labels, calibration_method.predict_proba(np.array(preds_adjusted_val*lambdaa + confidences*(1-lambdaa)).reshape(-1, 1))[:, 1])
                all_loss[lambdaa] = loss
            self.lambdaa = min(all_loss.keys(), key= lambda key:all_loss[key])

            Platt_method = LogisticRegression(penalty=None)
            confidences_w = preds_adjusted_val*lambdaa + confidences*(1-lambdaa)
            confidences_w = np.expand_dims(confidences_w, axis=1)
            Platt_method.fit(confidences_w, labels)
        
        assert self.lambdaa != None, "Please use the validation set to learn the lambda parameter first!"
        proba_adj = adjust_posterior_prob_to_new_prior(self.beta, confidences)
        confidences_w = proba_adj*self.lambdaa + confidences*(1-self.lambdaa)
        confidences_w = np.expand_dims(confidences_w, axis=1)
        proba_calibrated = Platt_method.predict_proba(confidences_w)
        return proba_calibrated,labels,True
        