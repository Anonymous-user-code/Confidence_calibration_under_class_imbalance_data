'''
Not yet finished
'''

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import logging
from mmcv.runner import auto_fp16, force_fp32, load_checkpoint
from torch import distributed as dist
import torch.nn as nn
import torchvision
import cv2

class PlotTensor:
    """Plot torch tensor as matplotlib figure.

    Args:
        apply_inv (bool): Whether to apply inverse normalization.
    """

    def __init__(self, apply_inv=True) -> None:
        trans_cifar = [
            torchvision.transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.201]),
            torchvision.transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])]
        trans_in = [
            torchvision.transforms.Normalize(
                mean=[ 0., 0., 0. ], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[ 1., 1., 1. ])]
        if apply_inv:
            self.invTrans_cifar = torchvision.transforms.Compose(trans_cifar)
            self.invTrans_in = torchvision.transforms.Compose(trans_in)

    def plot(self,
             img, nrow=4, title_name=None, save_name=None,
             make_grid=True, dpi=None, cmap='gray', apply_inv=True, overwrite=False):
        assert save_name is not None

        if make_grid:
            # make grid and save as plt images
            assert img.size(0) % nrow == 0
            ncol = img.size(0) // nrow
            if ncol > nrow:
                ncol = nrow
                nrow = img.size(0) // ncol
            img_grid = torchvision.utils.make_grid(img, nrow=nrow, pad_value=0)

            if img.size(1) == 1:
                cmap = getattr(plt.cm, cmap, plt.cm.jet)
            else:
                cmap = None
            if apply_inv:
                if img.size(2) <= 64:
                    img_grid = self.invTrans_cifar(img_grid)
                else:
                    img_grid = self.invTrans_in(img_grid)
            img_grid = torch.clip(img_grid * 255, 0, 255).int()
            img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
            
            fig = plt.figure(figsize=(nrow * 2, ncol * 2))
            plt.imshow(img_grid, cmap=cmap)
            if title_name is not None:
                plt.title(title_name)
            if not os.path.exists(save_name) or overwrite:
                plt.savefig(save_name, dpi=dpi)
            plt.close()
        else:
            # save each image sperately
            save_name = save_name.split('.')  # split into a name and a suffix
            if apply_inv:
                if img.size(2) <= 64:
                    img = self.invTrans_cifar(img)
                else:
                    img = self.invTrans_in(img)
            img = torch.clip(img * 255, 0, 255).detach().cpu().numpy()
            img = np.transpose(img, (0, 2, 3, 1))
            title_name = '' if title_name is None else title_name.replace(' ', '')

            for i in range(img.shape[0]):
                img_ = img[i, ...]
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                cv2.imwrite("{}_{}.{}".format(save_name[0], title_name+str(i), save_name[1]), img_)

@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

        *** Warning: torch.distributed.all_gather has no gradient. ***
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def batch_shuffle_ddp(x, idx_shuffle=None, no_repeat=False):
    """Batch shuffle (no grad), for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        return: x, idx_shuffle, idx_unshuffle.
        *** no repeat (09.23 update) ***
    
    Args:
        idx_shuffle: Given shuffle index if not None.
        no_repeat: The idx_shuffle does not have any repeat index as
            the original indice [i for i in range(N)]. It's used in
            mixup methods (self-supervisedion).
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    if idx_shuffle is None:
        # generate shuffle idx
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # each idx should not be the same as the original
        if bool(no_repeat) == True:
            idx_original = torch.tensor([i for i in range(batch_size_all)]).cuda()
            idx_repeat = False
            for i in range(20):  # try 20 times
                if (idx_original == idx_shuffle).any() == True:  # find repeat
                    idx_repeat = True
                    idx_shuffle = torch.randperm(batch_size_all).cuda()
                else:
                    idx_repeat = False
                    break
            # repeat hit: prob < 1.8e-4
            if idx_repeat == True:
                fail_to_shuffle = True
                idx_shuffle = idx_original.clone()
                for i in range(3):
                    # way 1: repeat prob < 1.5e-5
                    rand_ = torch.randperm(batch_size_all).cuda()
                    idx_parition = rand_ > torch.median(rand_)
                    idx_part_0 = idx_original[idx_parition == True]
                    idx_part_1 = idx_original[idx_parition != True]
                    if idx_part_0.shape[0] == idx_part_1.shape[0]:
                        idx_shuffle[idx_parition == True] = idx_part_1
                        idx_shuffle[idx_parition != True] = idx_part_0
                        if (idx_original == idx_shuffle).any() != True:  # no repeat
                            fail_to_shuffle = False
                            break
                # fail prob -> 0
                if fail_to_shuffle == True:
                    # way 2: repeat prob = 0, but too simple!
                    idx_shift = np.random.randint(1, batch_size_all-1)
                    idx_shuffle = torch.tensor(  # shift the original idx
                        [(i+idx_shift) % batch_size_all for i in range(batch_size_all)]).cuda()
    else:
        assert idx_shuffle.size(0) == batch_size_all, \
            "idx_shuffle={}, batchsize={}".format(idx_shuffle.size(0), batch_size_all)

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_shuffle, idx_unshuffle

@torch.no_grad()
def cutmix(img,
           gt_label,
           alpha=1.0,
           lam=None,
           dist_mode=False,
           return_mask=False,
           **kwargs):
    r""" CutMix augmentation.

    "CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features (https://arxiv.org/abs/1905.04899)". In ICCV, 2019.
        https://github.com/clovaai/CutMix-PyTorch
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
        return_mask (bool): Whether to return the cutting-based mask of
            shape (N, 1, H, W). Defaults to False.
    """

    def rand_bbox(size, lam, return_mask=False):
        """ generate random box by lam """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        if not return_mask:
            return bbx1, bby1, bbx2, bby2
        else:
            mask = torch.zeros((1, 1, W, H)).cuda()
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1
            mask = mask.expand(size[0], 1, W, H)  # (N, 1, H, W)
            return bbx1, bby1, bbx2, bby2, mask

    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # normal mixup process
    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
        _, _, h, w = img.size()
        y_a = gt_label
        y_b = gt_label[rand_index]

        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox(img.size(), lam, True)
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)

        return img, (y_a, y_b, lam)

    # dist mixup with cross gpus shuffle
    else:
        if len(img.size()) == 5:  # self-supervised img [N, 2, C, H, W]
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            assert len(img.size()) == 4  # normal img [N, C, H, w]
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        _, _, h, w = img.size()

        if not return_mask:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        else:
            bbx1, bby1, bbx2, bby2, mask = rand_bbox(img.size(), lam, True)
        img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        if return_mask:
            img = (img, mask)

        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)

@torch.no_grad()
def mixup(img,
          gt_label,
          alpha=1.0,
          lam=None,
          dist_mode=False,
          **kwargs):
    r""" MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10
    
    Args:
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        gt_label (Tensor): Ground-truth labels (one-hot).
        alpha (float): To sample Beta distribution.
        lam (float): The given mixing ratio. If lam is None, sample a lam
            from Beta distribution.
        dist_mode (bool): Whether to do cross gpus index shuffling and
            return the mixup shuffle index, which support supervised
            and self-supervised methods.
    """
    if lam is None:
        lam = np.random.beta(alpha, alpha)

    # normal mixup process
    if not dist_mode:
        rand_index = torch.randperm(img.size(0)).cuda()
        if len(img.size()) == 4:  # [N, C, H, W]
            img_ = img[rand_index]
        else:
            assert img.dim() == 5  # semi-supervised img [N, 2, C, H, W]
            # * notice that the rank of two groups of img is fixed
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()

        y_a = gt_label
        y_b = gt_label[rand_index]
        img = lam * img + (1 - lam) * img_
        return img, (y_a, y_b, lam)
    
    # dist mixup with cross gpus shuffle
    else:
        if len(img.size()) == 5:  # self-supervised img [N, 2, C, H, W]
            img_ = img[:, 1, ...].contiguous()
            img = img[:, 0, ...].contiguous()
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img_, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        else:
            assert len(img.size()) == 4  # normal img [N, C, H, w]
            img_, idx_shuffle, idx_unshuffle = batch_shuffle_ddp(  # N
                img, idx_shuffle=kwargs.get("idx_shuffle_mix", None), no_repeat=True)
        img = lam * img + (1 - lam) * img_
        
        if gt_label is not None:
            y_a = gt_label
            y_b, _, _ = batch_shuffle_ddp(
                gt_label, idx_shuffle=idx_shuffle, no_repeat=True)
            return img, (y_a, y_b, lam)
        else:
            return img, (idx_shuffle, idx_unshuffle, lam)

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "openmixup".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split('.')[0])  # i.e., openmixup
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used. Some
            special loggers are:
            - "root": the root logger obtained with `get_root_logger()`.
            - "silent": no message will be printed.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif logger == 'root':
        _logger = get_root_logger()
        _logger.log(level, msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger != 'silent':
        raise TypeError(
            'logger should be either a logging.Logger object, "root", '
            '"silent" or None, but got {}'.format(logger))


class AutoMixup(nn.Module):
    """ AutoMix and SAMix

    Official implementation of
        "AutoMix: Unveiling the Power of Mixup (https://arxiv.org/abs/2103.13027)"
        "Boosting Discriminative Visual Representation Learning with Scenario-Agnostic
            Mixup (https://arxiv.org/pdf/2111.15454.pdf)"

    *** Requiring Hook: `momentum_update` is adjusted by `CosineScheduleHook` after
        train_iter; `mask_loss` is adjusted by `CustomCosineAnnealingHook`.
    
    Args:
        backbone (dict): Config dict for module of backbone ConvNet (main).
        backbone_k (dict): Config dict for module of momentum backbone ConvNet. Default: None.
        mix_block (dict): Config dict for the mixblock. Default: None.
        head_mix (dict): Config dict for module of mixup classification loss (backbone).
        head_one (dict): Config dict for module of onehot classification loss (backbone).
        head_mix (dict): Config dict for mixup classification loss (mixblock). Default: None.
        head_one (dict): Config dict for onehot classification loss (mixblock). Default: None.
        head_weights (dict): Dict of the used cls heads names and loss weights,
            which determines the cls or mixup head in used.
            Default: dict(head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1)
        alpha (int): Beta distribution '$\beta(\alpha, \alpha)$'.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        mask_layer (int): Number of the feature layer indix in the backbone.
        mask_loss (float): Loss weight for the mixup mask. Default: 0.
        mask_adjust (float): Probrobality (in [0, 1]) of adjusting the mask (q) in terms
            of lambda (q), which only affect the backbone training.
            Default: False (or 0.).
        mask_up_override (str or list, optional): Override up_mode for MixBlock when training
            the MixBlock. Unsampling mode {'nearest', 'bilinear', etc}. Build a list for various
            upsampling mode (as in MixBlock). Default: None.
        pre_one_loss (float): Loss weight for the pre-MixBlock head as onehot classification.
            Default: 0. (requires a pre_head in MixBlock)
        pre_mix_loss (float): Loss weight for the pre-MixBlock head as mixup classification.
            Default: 0. (requires a pre_head in MixBlock)
        lam_margin (int): Margin of lambda to stop using AutoMix to train backbone
            when lam is small. If lam > lam_margin: AutoMix; else: vanilla mixup.
            Default: -1 (or 0).
        switch_off (bool or float): Switch off MixBlock updating for fast training. Default to
            False (or 0).
        head_one_mix (bool): Whether to use CutMix and Mixup as the one-hot cls loss for
            Transformer backbones (further enhancing AutoMix). Default to False.
        head_ensemble (bool): Whether to ensemble results of all heads. Default to False.
        mix_shuffle_no_repeat (bool): Whether to use 'no_repeat' mode to generate
            mixup shuffle idx. We can ignore this issue in supervised learning.
            Default: False.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        pretrained_k (str, optional): Path to pre-trained weights for en_k. Default: None.
        save_by_sample (bool): Whether to save mixup samples separately.
        debug_mode (bool): Whether to save some intermediate products.
    """

    def __init__(self,
                 backbone,
                 backbone_k=None,
                 mix_block=None,
                 head_mix=None,
                 head_one=None,
                 head_mix_k=None,
                 head_one_k=None,
                 head_weights=dict(
                    decent_weight=[], accent_weight=[],
                    head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
                 alpha=1.0,
                 momentum=0.999,
                 mask_layer=2,
                 mask_loss=0.,
                 mask_adjust=0.,
                 mask_up_override=None,
                 pre_one_loss=0.,
                 pre_mix_loss=0.,
                 lam_margin=-1,
                 switch_off=0.,
                 head_one_mix=False,
                 head_ensemble=False,
                 save=False,
                 save_name='MixedSamples',
                 save_by_sample=False,
                 debug=False,
                 mix_shuffle_no_repeat=False,
                 pretrained=None,
                 pretrained_k=None,
                 init_cfg=None,
                 **kwargs):
        super(AutoMixup, self).__init__(init_cfg, **kwargs)
        # basic params
        self.alpha = float(alpha)
        self.mask_layer = int(mask_layer)
        self.momentum = float(momentum)
        self.base_momentum = float(momentum)
        self.mask_loss = float(mask_loss) if float(mask_loss) > 0 else 0
        self.mask_adjust = float(mask_adjust)
        self.pre_one_loss = float(pre_one_loss) if float(pre_one_loss) > 0 else 0
        self.pre_mix_loss = float(pre_mix_loss) if float(pre_mix_loss) > 0 else 0
        self.lam_margin = float(lam_margin) if float(lam_margin) > 0 else 0
        self.switch_off = float(switch_off) if float(switch_off) > 0 else 0
        self.head_one_mix = bool(head_one_mix)
        self.head_ensemble = bool(head_ensemble)
        self.mask_up_override = mask_up_override \
            if isinstance(mask_up_override, (str, list)) else None
        self.save = bool(save)
        self.save_name = str(save_name)
        self.ploter = PlotTensor(apply_inv=True)
        self.save_by_sample = bool(save_by_sample)
        self.debug = bool(debug)
        self.mix_shuffle_no_repeat = bool(mix_shuffle_no_repeat)
        assert 0 <= self.momentum and self.lam_margin < 1 and self.mask_adjust <= 1
        if self.mask_up_override is not None:
            if isinstance(self.mask_up_override, str):
                self.mask_up_override = [self.mask_up_override]
            for m in self.mask_up_override:
                assert m in ['nearest', 'bilinear', 'bicubic',]

        # network
        assert isinstance(mix_block, dict) and isinstance(backbone, dict)
        assert backbone_k is None or isinstance(backbone_k, dict)
        assert head_mix is None or isinstance(head_mix, dict)
        assert head_one is None or isinstance(head_one, dict)
        assert head_mix_k is None or isinstance(head_mix_k, dict)
        assert head_one_k is None or isinstance(head_one_k, dict)
        head_mix_k = head_mix if head_mix_k is None else head_mix_k
        head_one_k = head_one if head_one_k is None else head_one_k
        # mixblock
        self.mix_block = builder.build_head(mix_block)
        # backbone
        self.backbone_q = builder.build_backbone(backbone)
        if backbone_k is not None:
            self.backbone_k = builder.build_backbone(backbone_k)
            assert self.momentum >= 1. and pretrained_k is not None
        else:
            self.backbone_k = builder.build_backbone(backbone)
        self.backbone = self.backbone_k  # for feature extract
        # mixup cls head
        assert "head_mix_q" in head_weights.keys() and "head_mix_k" in head_weights.keys()
        self.head_mix_q = builder.build_head(head_mix)
        self.head_mix_k = builder.build_head(head_mix_k)
        # onehot cls head
        if "head_one_q" in head_weights.keys():
            self.head_one_q = builder.build_head(head_one)
        else:
            self.head_one_q = None
        if "head_one_k" in head_weights.keys():
            self.head_one_k = builder.build_head(head_one_k)
        else:
            self.head_one_k = None
        # for feature extract
        self.head = self.head_one_k if self.head_one_k is not None else self.head_one_q
        # onehot and mixup heads for training
        self.weight_mix_q = head_weights.get("head_mix_q", 1.)
        self.weight_mix_k = head_weights.get("head_mix_k", 1.)
        self.weight_one_q = head_weights.get("head_one_q", 1.)
        assert self.weight_mix_q > 0 and (self.weight_mix_k > 0 or backbone_k is not None)
        self.head_weights = head_weights
        self.head_weights['decent_weight'] = head_weights.get("decent_weight", list())
        self.head_weights['accent_weight'] = head_weights.get("accent_weight", list())
        self.head_weights['mask_loss'] = self.mask_loss
        self.head_weights['pre_one_loss'] = self.pre_one_loss
        self.head_weights['pre_mix_loss'] = self.pre_mix_loss
        self.cos_annealing = 1.  # decent from 1 to 0 as cosine
        
        self.init_weights(pretrained=pretrained, pretrained_k=pretrained_k)

    def init_weights(self, pretrained=None, pretrained_k=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
            pretrained_k (str, optional): Path to pre-trained weights to initialize the
                backbone_k and mixblock. Default: None.
        """
        # init mixblock
        if self.mix_block is not None:
            self.mix_block.init_weights(init_linear='normal')
        # init pretrained backbone_k and mixblock
        if pretrained_k is not None:
            print_log('load pretrained classifier k from: {}'.format(pretrained_k), logger='root')
            # load full ckpt to backbone and fc
            logger = logging.getLogger()
            load_checkpoint(self, pretrained_k, strict=False, logger=logger)
            # head_mix_k and head_one_k should share the same initalization
            if self.head_mix_k is not None and self.head_one_k is not None:
                for param_one_k, param_mix_k in zip(self.head_one_k.parameters(),
                                                    self.head_mix_k.parameters()):
                    param_mix_k.data.copy_(param_one_k.data)
                    param_mix_k.requires_grad = False  # stop grad k
        
        # init backbone, based on params in q
        if pretrained is not None:
            print_log('load encoder_q from: {}'.format(pretrained), logger='root')
        self.backbone_q.init_weights(pretrained=pretrained)
        # copy backbone param from q to k
        if pretrained_k is None and self.momentum < 1:
            for param_q, param_k in zip(self.backbone_q.parameters(),
                                        self.backbone_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False  # stop grad k
        
        # init head
        if self.head_mix_q is not None:
            self.head_mix_q.init_weights()
        if self.head_one_q is not None:
            self.head_one_q.init_weights()
        
        # copy head one param from q to k
        if (self.head_one_q is not None and self.head_one_k is not None) and \
            (pretrained_k is None and self.momentum < 1):
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(),
                                                self.head_one_k.parameters()):
                param_one_k.data.copy_(param_one_q.data)
                param_one_k.requires_grad = False  # stop grad k
        # copy head mix param from q to k
        if (self.head_mix_q is not None and self.head_mix_k is not None) and \
            (pretrained_k is None and self.momentum < 1):
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(),
                                                self.head_mix_k.parameters()):
                param_mix_k.data.copy_(param_mix_q.data)
                param_mix_k.requires_grad = False  # stop grad k

    def _update_loss_weights(self):
        """ update loss weights according to the cos_annealing scalar """
        if self.cos_annealing < 0 or self.cos_annealing > 1:
            return
        # cos annealing decent, from 1 to 0
        if len(self.head_weights["decent_weight"]) > 0:
            for attr in self.head_weights["decent_weight"]:
                setattr(self, attr, self.head_weights.get(attr, 1.) * self.cos_annealing)
        # cos annealing accent, from 0 to 1
        if len(self.head_weights["accent_weight"]) > 0:
            for attr in self.head_weights["accent_weight"]:
                setattr(self, attr, self.head_weights.get(attr, 1.) * (1-self.cos_annealing))

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the k form q by hook, including the backbone and heads """
        # we don't update q to k when momentum > 1
        if self.momentum >= 1.:
            return
        # update k's backbone and cls head from q
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                        param_q.data * (1. - self.momentum)
        
        if self.head_one_q is not None and self.head_one_k is not None:
            for param_one_q, param_one_k in zip(self.head_one_q.parameters(),
                                                self.head_one_k.parameters()):
                param_one_k.data = param_one_k.data * self.momentum + \
                                    param_one_q.data * (1 - self.momentum)

        if self.head_mix_q is not None and self.head_mix_k is not None:
            for param_mix_q, param_mix_k in zip(self.head_mix_q.parameters(),
                                                self.head_mix_k.parameters()):
                param_mix_k.data = param_mix_k.data * self.momentum + \
                                    param_mix_q.data * (1 - self.momentum)

    def _no_repeat_shuffle_idx(self, batch_size_this, ignore_failure=False):
        """ generate no repeat shuffle idx within a gpu """
        idx_shuffle = torch.randperm(batch_size_this).cuda()
        idx_original = torch.tensor([i for i in range(batch_size_this)]).cuda()
        idx_repeat = False
        for i in range(10):  # try 10 times
            if (idx_original == idx_shuffle).any() == True:
                idx_repeat = True
                idx_shuffle = torch.randperm(batch_size_this).cuda()
            else:
                idx_repeat = False
                break
        # hit: prob < 1.2e-3
        if idx_repeat == True and ignore_failure == False:
            # way 2: repeat prob = 0, but too simple!
            idx_shift = np.random.randint(1, batch_size_this-1)
            idx_shuffle = torch.tensor(  # shift the original idx
                [(i+idx_shift) % batch_size_this for i in range(batch_size_this)]).cuda()
        return idx_shuffle

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of a batch of images, (N, C, H, W).
            gt_label (Tensor): Groundtruth onehot labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if isinstance(img, list):
            img = img[0]
        batch_size = img.size(0)
        self._update_loss_weights()
        
        lam = np.random.beta(self.alpha, self.alpha, 2)  # 0: mb, 1: bb
        if self.mix_shuffle_no_repeat:
            index_bb = self._no_repeat_shuffle_idx(batch_size, ignore_failure=True)
            index_mb = self._no_repeat_shuffle_idx(batch_size, ignore_failure=False)
        else:
            index_bb = torch.randperm(batch_size).cuda()
            index_mb = torch.randperm(batch_size).cuda()

        # auto Mixup
        indices = [index_mb, index_bb]
        feature = self.backbone_k(img)[0]
        results = self.pixel_mixup(img, gt_label, lam, indices, feature)

        # save img bb, mixed sample visualization
        if self.save and self.mask_adjust > 0:
            self.plot_mix(
                results["img_mix_bb"], img, img[index_bb, :], lam[1], results["debug_plot"], "backbone")
        # save img mb
        if self.save and self.mask_adjust <= 0:
            self.plot_mix(
                results["img_mix_mb"], img, img[index_mb, :], lam[0], results["debug_plot"], "mixblock")
        
        # k (mb): the mix block training
        loss_mix_k = self.forward_k(results["img_mix_mb"], gt_label, index_mb, lam[0])
        # q (bb): the encoder training
        loss_one_q, loss_mix_q = self.forward_q(img, results["img_mix_bb"], gt_label, index_bb, lam[1])
        
        # loss summary
        losses = {
            'loss': loss_mix_q['loss'] * self.weight_mix_q,
            'acc_mix_q': loss_mix_q['acc'],
        }
        # onehot loss
        if loss_one_q is not None and self.weight_one_q > 0:
            losses['loss'] += loss_one_q['loss'] * self.weight_one_q
            losses['acc_one_q'] = loss_one_q['acc']
        # mixblock loss
        if loss_mix_k['loss'] is not None and self.weight_mix_k > 0:
            losses["loss"] += loss_mix_k['loss'] * self.weight_mix_k
            losses['acc_mix_k'] = loss_mix_k['acc']
        else:
            losses['acc_mix_k'] = loss_mix_q['acc']
        if results["mask_loss"] is not None and self.mask_loss > 0:
            losses["loss"] += results["mask_loss"]
        if results["pre_one_loss"] is not None and self.pre_one_loss > 0:
            losses["loss"] += results["pre_one_loss"]
        if loss_mix_k["pre_mix_loss"] is not None and self.pre_mix_loss > 0:
            losses["loss"] += loss_mix_k["pre_mix_loss"]

        return losses

    @force_fp32(apply_to=('im_mixed', 'im_q', 'im_k',))
    def plot_mix(self, im_mixed, im_q, im_k, lam, debug_plot=None, name="k"):
        """ visualize mixup results, supporting 'debug' mode """
        # plot mixup results
        img = torch.cat((im_q[:4], im_k[:4], im_mixed[:4]), dim=0)
        title_name = 'lambda {}={}'.format(name, lam)
        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=4, title_name=title_name, save_name=self.save_name,
            make_grid=not self.save_by_sample)

        # debug: plot intermediate results, fp32
        if self.debug:
            assert isinstance(debug_plot, dict)
            for key,value in debug_plot.items():
                _, h, w = value.size()
                img = value[:4].view(h, 4 * w).type(torch.float32).detach().cpu().numpy()
                fig = plt.figure()
                plt.imshow(img)
                # plt.title('debug {}, lambda k={}'.format(str(key), lam))
                _debug_path = self.save_name.split(".png")[0] + "_{}.png".format(str(key))
                if not os.path.exists(_debug_path):
                    plt.savefig(_debug_path, bbox_inches='tight')
                plt.close()

    @auto_fp16(apply_to=('x', 'mixed_x', ))
    def forward_q(self, x, mixed_x, y, index, lam):
        """
        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            mixed_x (Tensor): Mixup images of x, (N, C, H, W).
            y (Tensor): Groundtruth onehot labels, coresponding to x.
            index (List): Input list of shuffle index (tensor) for mixup.
            lam (List): Input list of lambda (scalar).

        Returns:
            dict[str, Tensor]: loss_one_q and loss_mix_q are losses from q.
        """
        # onehot q
        loss_one_q = None
        if self.head_one_q is not None and self.weight_one_q > 0:
            if self.head_one_mix:  # mixup in head_one
                if np.random.random() > 0.5:
                    x, y_one_mix = mixup(x.clone(), y, 0.8, dist_mode=False)
                else:
                    x, y_one_mix = cutmix(x.clone(), y, 1.0, dist_mode=False)
            out_one_q = self.backbone_q(x)[-1]
            pred_one_q = self.head_one_q([out_one_q])
            # loss
            if self.head_one_mix:
                loss_one_q = self.head_one_q.loss(pred_one_q, y_one_mix)
            else:
                loss_one_q = self.head_one_q.loss(pred_one_q, y)
            if torch.isnan(loss_one_q['loss']):
                print_log("Warming NAN in loss_one_q. Please use FP32!", logger='root')
                loss_one_q = None
        
        # mixup q
        loss_mix_q = None
        if self.weight_mix_q > 0:
            out_mix_q = self.backbone_q(mixed_x)[-1]
            pred_mix_q = self.head_mix_q([out_mix_q])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_q[0] = pred_mix_q[0].type(torch.float32)
            # mixup loss
            y_mix_q = (y, y[index], lam)
            loss_mix_q = self.head_mix_q.loss(pred_mix_q, y_mix_q)
            if torch.isnan(loss_mix_q['loss']):
                print_log("Warming NAN in loss_mix_q. Please use FP32!", logger='root')
                loss_mix_q = dict(loss=None)
        
        return loss_one_q, loss_mix_q

    @auto_fp16(apply_to=('mixed_x', ))
    def forward_k(self, mixed_x, y, index, lam):
        """ forward k with the mixup sample """
        loss_mix_k = dict(loss=None, pre_mix_loss=None)
        # switch off mixblock training
        if self.switch_off > 0:
            if 0 < self.cos_annealing <= 1:
                if np.random.rand() > self.switch_off * self.cos_annealing:
                    return loss_mix_k
        
        # training mixblock from k
        if self.weight_mix_k > 0:
            # mixed_x forward
            out_mix_k = self.backbone_k(mixed_x)
            pred_mix_k = self.head_mix_k([out_mix_k[-1]])
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_k[0] = pred_mix_k[0].type(torch.float32)
            # k mixup loss
            y_mix_k = (y, y[index], lam)
            loss_mix_k = self.head_mix_k.loss(pred_mix_k, y_mix_k)
            if torch.isnan(loss_mix_k['loss']):
                print_log("Warming NAN in loss_mix_k. Please use FP32!", logger='root')
                loss_mix_k["loss"] = None

        # mixup loss, short cut of pre-mixblock
        if self.pre_mix_loss > 0:
            out_mb = out_mix_k[0]
            # pre FFN
            if self.mix_block.pre_attn is not None:
                out_mb = self.mix_block.pre_attn(out_mb)  # non-local
            if self.mix_block.pre_conv is not None:
                out_mb = self.mix_block.pre_conv([out_mb])  # neck
            # pre mixblock mixup loss
            pred_mix_mb = self.mix_block.pre_head(out_mb)
            # force fp32 in mixup loss (causing NAN in fp16 training with a large batch size)
            pred_mix_mb[0] = pred_mix_mb[0].type(torch.float32)
            loss_mix_k["pre_mix_loss"] = \
                self.mix_block.pre_head.loss(pred_mix_mb, y_mix_k)["loss"] * self.pre_mix_loss
            if torch.isnan(loss_mix_k["pre_mix_loss"]):
                print_log("Warming NAN in pre_mix_loss.", logger='root')
                loss_mix_k["pre_mix_loss"] = None
        else:
            loss_mix_k["pre_mix_loss"] = None
        
        return loss_mix_k

    def pixel_mixup(self, x, y, lam, index, feature):
        """ pixel-wise input space mixup

        Args:
            x (Tensor): Input of a batch of images, (N, C, H, W).
            y (Tensor): A batch of gt_labels, (N, 1).
            lam (List): Input list of lambda (scalar).
            index (List): Input list of shuffle index (tensor) for mixup.
            feature (Tensor): The feature map of x, (N, C, H', W').

        Returns: dict includes following
            mixed_x_bb, mixed_x_mb: Mixup samples for bb (training the backbone)
                and mb (training the mixblock).
            mask_loss (Tensor): Output loss of mixup masks.
            pre_one_loss (Tensor): Output onehot cls loss of pre-mixblock.
        """
        results = dict()
        # lam info
        lam_mb = lam[0]  # lam is a scalar
        lam_bb = lam[1]

        # mask upsampling factor
        if x.shape[3] > 64:  # normal version of resnet
            scale_factor = 2**(2 + self.mask_layer)
        else:  # CIFAR version
            scale_factor = 2**self.mask_layer
        
        # get mixup mask
        mask_mb = self.mix_block(feature, lam_mb, index[0],
            scale_factor=scale_factor, debug=self.debug, unsampling_override=self.mask_up_override)
        mask_bb = self.mix_block(feature, lam_bb, index[1],
            scale_factor=scale_factor, debug=False, unsampling_override=None)
        if self.debug:
            results["debug_plot"] = mask_mb["debug_plot"]
        else:
            results["debug_plot"] = None

        # pre mixblock loss
        results["pre_one_loss"] = None
        if self.pre_one_loss > 0.:
            pred_one = self.mix_block.pre_head([mask_mb["x_lam"]])
            y_one = (y, y, 1)
            results["pre_one_loss"] = \
                self.mix_block.pre_head.loss(pred_one, y_one)["loss"] * self.pre_one_loss
            if torch.isnan(results["pre_one_loss"]):
                print_log("Warming NAN in pre_one_loss.", logger='root')
                results["pre_one_loss"] = None
        
        mask_mb = mask_mb["mask"]
        mask_bb = mask_bb["mask"].clone().detach()

        # adjust mask_bb with lambd
        if self.mask_adjust > np.random.rand():  # [0,1)
            epsilon = 1e-8
            _mask = mask_bb[:, 0, :, :].squeeze()  # [N, H, W], _mask for lam
            _mask = _mask.clamp(min=epsilon, max=1-epsilon)
            _mean = _mask.mean(dim=[1, 2]).squeeze()  # [N, 1, 1] -> [N]
            idx_larg = _mean[:] > lam[0] + epsilon  # index of mean > lam_bb
            idx_less = _mean[:] < lam[0] - epsilon  # index of mean < lam_bb
            # if mean > lam_bb
            mask_bb[idx_larg==True, 0, :, :] = \
                _mask[idx_larg==True, :, :] * (lam[0] / _mean[idx_larg==True].view(-1, 1, 1))
            mask_bb[idx_larg==True, 1, :, :] = 1 - mask_bb[idx_larg==True, 0, :, :]
            # elif mean < lam_bb
            mask_bb[idx_less==True, 1, :, :] = \
                (1 - _mask[idx_less==True, :, :]) * ((1 - lam[0]) / (1 - _mean[idx_less==True].view(-1, 1, 1)))
            mask_bb[idx_less==True, 0, :, :] = 1 - mask_bb[idx_less==True, 1, :, :]
        # lam_margin for backbone training
        if self.lam_margin >= lam_bb or self.lam_margin >= 1-lam_bb:
            mask_bb[:, 0, :, :] = lam_bb
            mask_bb[:, 1, :, :] = 1 - lam_bb
        
        # loss of mixup mask
        results["mask_loss"] = None
        if self.mask_loss > 0.:
            results["mask_loss"] = self.mix_block.mask_loss(mask_mb, lam_mb)["loss"]
            if results["mask_loss"] is not None:
                results["mask_loss"] *= self.mask_loss
        
        # mix, apply mask on x and x_
        # img_mix_mb = x * (1 - mask_mb) + x[index[0], :] * mask_mb
        assert mask_mb.shape[1] == 2
        assert mask_mb.shape[2:] == x.shape[2:], f"Invalid mask shape={mask_mb.shape}"
        results["img_mix_mb"] = \
            x * mask_mb[:, 0, :, :].unsqueeze(1) + x[index[0], :] * mask_mb[:, 1, :, :].unsqueeze(1)
        
        # img_mix_bb = x * (1 - mask_bb) + x[index[1], :] * mask_bb
        results["img_mix_bb"] = \
            x * mask_bb[:, 0, :, :].unsqueeze(1) + x[index[1], :] * mask_bb[:, 1, :, :].unsqueeze(1)
        
        return results

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        keys = list()  # 'acc_mix_k', 'acc_one_k', 'acc_mix_q', 'acc_one_q'
        pred = list()
        # backbone
        last_k = self.backbone_k(img)[-1]
        last_q = self.backbone_q(img)[-1]
        # head k
        if self.weight_mix_k > 0:
            pred.append(self.head_mix_k([last_k]))
            keys.append('acc_mix_k')
            if self.head_one_k is not None:
                pred.append(self.head_one_k([last_k]))
                keys.append('acc_one_k')
        # head q
        pred.append(self.head_mix_q([last_q]))
        keys.append('acc_mix_q')
        if self.head_one_q is not None:
            pred.append(self.head_one_q([last_q]))
            keys.append('acc_one_q')
        # head ensemble
        if self.head_ensemble:
            pred.append([torch.stack(
                [pred[i][0] ** 2 for i in range(len(pred))]).mean(dim=0)])
            keys.append('acc_avg')

        out_tensors = [p[0].cpu() for p in pred]  # NxC
        return dict(zip(keys, out_tensors))

    def augment_test(self, img, **kwargs):
        """Test function with test time augmentation."""
        keys = list()  # 'acc_mix_k', 'acc_one_k', 'acc_mix_q', 'acc_one_q'
        pred = list()
        # backbone
        img = img[0]
        last_k = self.backbone_k(img)[-1]
        last_q = self.backbone_q(img)[-1]
        # head k
        if self.weight_mix_k > 0:
            pred.append(self.head_mix_k([last_k]))
            keys.append('acc_mix_k')
            if self.head_one_k is not None:
                pred.append(self.head_one_k([last_k]))
                keys.append('acc_one_k')
        # head q
        pred.append(self.head_mix_q([last_q]))
        keys.append('acc_mix_q')
        if self.head_one_q is not None:
            pred.append(self.head_one_q([last_q]))
            keys.append('acc_one_q')
        # head ensemble
        if self.head_ensemble:
            pred.append([torch.stack(
                [pred[i][0] ** 2 for i in range(len(pred))]).mean(dim=0)])
            keys.append('acc_avg')

        out_tensors = [p[0].cpu() for p in pred]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_test(self, img, **kwargs):
        """Forward computation during testing.

        Args:
            img (List[Tensor] or Tensor): the outer list indicates the
                test-time augmentations and inner Tensor should have a
                shape of (N, C, H, W).

        Returns:
            dict[key, Tensor]: A dictionary of head names (key) and predictions.
        """
        if isinstance(img, list):
            return self.augment_test(img)
        else:
            return self.simple_test(img)

    def forward_inference(self, img, **kwargs):
        """Forward output for inference.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.

        Returns:
            tuple[Tensor]: final model outputs.
        """
        x = self.backbone(img)[-1]
        preds = self.head([x], post_process=True)
        return preds[0]

    def forward_vis(self, img, gt_label, **kwargs):
        """" visualization by jupyter notebook """
        batch_size = img.size()[0]
        lam = kwargs.get('lam', [0.5, 0.5])
        index = [i+1 for i in np.arange(batch_size)]
        index[-1] = 0

        # forward mixblock
        indices = [index, index]
        feature = self.backbone_k(img)[0]
        results = self.pixel_mixup(img, gt_label, lam, indices, feature)
        return {'mix_bb': results["img_mix_bb"], 'mix_mb': results["img_mix_mb"]}
