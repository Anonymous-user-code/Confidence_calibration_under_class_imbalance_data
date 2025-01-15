'''
Reference paper: "Transfer Knowledge from Head to Tail: Uncertainty Calibration under Long-tailed Distribution"
Reference code: "https://github.com/JiahaoChen1/Calibration"
'''

import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal
import torch.nn as nn
from scipy import optimize
import torch
import torch.nn.functional as F
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY

def Wasserstein(mu1, sigma1, mu2, sigma2):
    p1 = np.sum(np.power((mu1 - mu2),2))
    p2 = np.sum(np.power(np.power(sigma1,1/2) - np.power(sigma2, 1/2),2))
    return p1 + p2

def label2onehot(logits, labels):
    label_onehot = torch.zeros_like(torch.tensor(logits))
    label_onehot.scatter_(1, torch.tensor(labels).long().view(-1, 1), 1)
    return label_onehot.numpy()

def ll_t_da(t, *args):
    logits, labels, w = args
    logits = logits / t
    n = np.sum(np.clip(np.exp(logits), -1e20, 1e20), 1)
    p = np.clip(np.clip(np.exp(logits), -1e20, 1e20) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(labels * np.log(p) * w[:, None]) / N
    return ce

def logit_to_confidence(z,y):
    softmaxes = F.softmax(z, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    confidences,resort_index = torch.sort(confidences)
    labels = y[resort_index]
    predictions = predictions[resort_index]
    return confidences,predictions,labels

class TKHT_base(nn.Module):
    def __init__(self,cfg,num_head_cls) -> None:
        super().__init__()

        self.head_mask = np.array([-100. for i in range(cfg.MODEL.NUM_CLASS)])
        self.head_mask[:num_head_cls] = 1
        self.alpha = 0.9
        self.t = None
        self.need_calibration_train = True
        self.require_iterative_training = False
        self.cfg = cfg

    def forward(self,val_logits,val_labels,val_feature=None,train_features=None,train_labels=None):
        if self.training:
            self.w = np.zeros((val_logits.shape[0]))
            dst_means, dst_covs = [], []
            target_norms = []
            for i in range(val_logits.shape[-1]):
                t_datas = train_features[train_labels == i, :]
                dst_means.append(np.mean(t_datas, axis=0))
                dst_covs.append(np.var(t_datas, axis=0))
                target_norms.append(multivariate_normal(mean=dst_means[-1], cov=dst_covs[-1], allow_singular=True))
            self.wasser_matrix = np.zeros((val_logits.shape[-1], val_logits.shape[-1]))
            for i in range(val_logits.shape[-1]):
                for j in range(val_logits.shape[-1]):
                    if i == j:
                        self.wasser_matrix[i, j] = -1e9
                    elif self.head_mask[j] == -100:
                        self.wasser_matrix[i, j] = -1e9
                    else:
                        self.wasser_matrix[i, j] = -(Wasserstein(dst_means[i], dst_covs[i], dst_means[j], dst_covs[j])) / (train_features.shape[-1] ** (1/2))
                self.wasser_matrix[i] = softmax(self.wasser_matrix[i])
            for i in range(val_logits.shape[0]):
                gt_cls = np.int32(val_labels[i])
                if self.head_mask[gt_cls] == 1.:
                    self.w[i] = 1.
                else:
                    shift_mean = np.sum(np.array(dst_means) * self.wasser_matrix[gt_cls][:, None], axis=0) * (1 - self.alpha) + dst_means[gt_cls] * self.alpha
                    shift_cov = (np.sum(np.sqrt(np.array(dst_covs)) * self.wasser_matrix[gt_cls][:, None], axis=0) * (1 - self.alpha) + np.sqrt(dst_covs[gt_cls]) * self.alpha)** 2
                    self.w[i] = np.exp(multivariate_normal(mean=shift_mean, cov=shift_cov, allow_singular=True).logpdf(val_feature[i]) -\
                                target_norms[gt_cls].logpdf(val_feature[i]))
                    self.w[i] = np.clip(self.w[i], 0.3, 5)
        
            val_labels = label2onehot(val_logits,val_labels)
            bnds = ((0.05, 5.0),)
            t = optimize.minimize(
                ll_t_da,
                1.0,
                args=(val_logits, val_labels, self.w),
                method='L-BFGS-B',
                bounds=bnds, tol=1e-12,
                options={'disp': False})
            self.t = t.x
            self.t = torch.from_numpy(self.t)
        else:
            assert self.t != None,"Need to calibrate with validation set first!"
            val_logits = torch.from_numpy(val_logits)
            val_labels = torch.from_numpy(val_labels)
            logits = val_logits / self.t
            confidences,predictions,labels = logit_to_confidence(logits,val_labels)
            return confidences,predictions,labels


@META_ARCH_REGISTRY.register()
def TKHT_2Head(cfg):
    num_head_cls = 2
    return TKHT_base(cfg,num_head_cls=num_head_cls)