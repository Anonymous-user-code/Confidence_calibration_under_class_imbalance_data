'''
Reference paper:《Intra Order-Preserving Functions for Calibretion of Multi-Class Neural Networks》
Official code: https://github.com/AmirooR/IntraOrderPreservingCalibration
'''

import torch
import torch.nn as nn
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn.functional as F
from customKing.solver.build import build_lossFun

class OrderPreservingModel(nn.Module):
  def __init__(self, cfg, invariant=False, residual=False, m_activation=F.softplus):
    super(OrderPreservingModel, self).__init__()
    self.base_model = nn.Sequential(nn.Linear(cfg.MODEL.NUM_CLASS,int(cfg.MODEL.NUM_CLASS*1.5)),
                                    nn.ReLU(),
                                    nn.Linear(int(cfg.MODEL.NUM_CLASS*1.5),cfg.MODEL.NUM_CLASS),
                                    )
    for module in self.base_model.modules():
        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(module.weight.data.type(torch.float64))
            module.bias = nn.Parameter(module.bias.data.type(torch.float64))
    
    self.num_classes = cfg.MODEL.NUM_CLASS #it is used in msodir_loss
    self.invariant = invariant
    self.m_activation = m_activation
    self.residual = residual
    #初始化损失函数
    self.lossfun = build_lossFun(cfg)    
    self.need_calibration_train = True
    self.require_iterative_training = True

  def compute_u(self, sorted_logits):
    diffs = sorted_logits[:,:-1] - sorted_logits[:,1:]
    diffs = torch.cat((diffs, torch.ones((diffs.shape[0],1),
                                          dtype=diffs.dtype,
                                          device=diffs.device)), dim=1)
    assert(torch.all(diffs >= 0)), 'diffs should be positive: {}'.format(diffs)
    return diffs.flip([1])

  def forward(self, logits,label_list):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    _, unsorted_indices = torch.sort(sorted_indices, descending=False)
    #[B, C]
    u = self.compute_u(sorted_logits)
    inp = sorted_logits if self.invariant else logits
    m = self.base_model(inp)
    m[:,1:] = self.m_activation(m[:,1:].clone())
    m[:,0] = 0
    um = torch.cumsum(u*m,1).flip([1])
    out = torch.gather(um,1,unsorted_indices)
    if self.residual:
      out = out + logits

    #计算损失函数
    lossValue = self.lossfun(out,label_list)

    softmaxed = None
    return out,lossValue,None

@META_ARCH_REGISTRY.register()
def intra_order_preserving_model(cfg):
    return OrderPreservingModel(cfg)