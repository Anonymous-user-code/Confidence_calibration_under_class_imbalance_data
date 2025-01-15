
import torch
from torch import nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

class Isotonic_regression(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.need_calibration_train = True
        self.require_iterative_training = False
        self.model = IsotonicRegression(out_of_bounds='clip')

    def forward(self,Simple_vector,label_list):
        prob = F.softmax(Simple_vector, dim=1)
        confidences, predictions = torch.max(prob, 1)
        hits = predictions.eq(label_list)
        confidences,resort_index = torch.sort(confidences)
        hits = hits[resort_index]
        confidences = confidences.cpu().detach().numpy()
        hits = hits.cpu().detach().numpy()
        if self.training:
            self.model.fit(confidences,hits)
        else:
            cali_confidence = self.model.predict(confidences)
            return cali_confidence,hits,True
        
@META_ARCH_REGISTRY.register()
def isotonicRegression(cfg):
    return Isotonic_regression(cfg)


