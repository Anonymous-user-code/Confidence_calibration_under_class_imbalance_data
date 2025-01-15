# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from customKing.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
import inspect


def build_model(cfg,cls_num_list_train,cls_num_list_test):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    # if cfg.MODEL.MODE == "VQVAE" and cfg.MODEL.TRAIN_PIXEL == True:
    #     meta_arch = cfg.MODEL.PIXEL_MODEL
    # else:
    #     meta_arch = cfg.MODEL.META_ARCHITECTURE
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model_object = META_ARCH_REGISTRY.get(meta_arch)
    signature = inspect.signature(model_object)
    names = []
    for name, param in signature.parameters.items():
        names.append(name)
    if len(names) == 1:
        model = model_object(cfg)
    elif len(names) == 2:
        model = model_object(cfg,cls_num_list_train)
    elif len(names) == 3:
        model = model_object(cfg,cls_num_list_train,cls_num_list_test)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

def build_metric(cfg,ece_method):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    model = META_ARCH_REGISTRY.get(ece_method)(cfg)
    return model
