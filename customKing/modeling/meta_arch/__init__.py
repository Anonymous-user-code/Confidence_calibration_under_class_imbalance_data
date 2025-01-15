# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from turtle import forward
from .build import META_ARCH_REGISTRY, build_model  # isort:skip


# Classification task
#Image classification
from .Image_classification.Resnext import Resnet20,Resnet110,Resnet18,Resnet34,Resnet50,Resnet101,Resnet152,ResNeXt29_8x64d,ResNeXt29_16x64d,ResNeXt50,ResNeXt101,Wide_resnet34_2,Wide_resnet50_2,Wide_resnet101_2
from .Image_classification.Wide_Resnet import Wide_ResNet28_10,Wide_ResNet34_2
from .Image_classification.ResNet_SD import resnet50_StoDepth_lineardecay,resnet34_StoDepth_lineardecay,resnet18_StoDepth_lineardecay,resnet101_StoDepth_lineardecay,resnet152_StoDepth_lineardecay
from .Image_classification.ResNet_NTS import resnet50_nts
from .Image_classification.DenseNet import densenet_k12_D40,densenet_k12_D100,densenet_k24_D100,densenet_BC_k12_D100,densenet_BC_k24_D250,densenet_BC_k40_D190,densenet121,densenet161,densenet169,densenet201
from .Image_classification.CoAtNets import coatnet_0,coatnet_1,coatnet_2,coatnet_3,coatnet_4
from .Image_classification.VisionTransformer import Vit_small,Vit_b_16,Vit_l_32,Vit_b_32,Vit_l_16
from .Image_classification.SwinTransformer import swin_b,swin_l,swin_s,swin_t
from .Image_classification.PNASNet5 import PNASNet5
#1D classification
from .oneD_classification.MLPClassfier import MLPClassifier

#General Calibration
#Calibration metrics
from .Calibration_metrics.General_metrics.naive_ECE import ECE_with_equal_mass,ECE_with_equal_width
from .Calibration_metrics.General_metrics.KS_error import KS_error
from .Calibration_metrics.General_metrics.Debaised_ECE import Debaised_ECE
from .Calibration_metrics.General_metrics.ECE_sweep import ECE_sweep_em
from .Calibration_metrics.General_metrics.Smoothing_ECE import SmoothingECE
#post-hoc
from .Confidence_calibration.Post_hoc_methods.General_Methods.Temperature_scale import temperature_scale_cross_entropy,temperature_scale_with_ece
from .Confidence_calibration.Post_hoc_methods.General_Methods.IsotonicRegression import isotonicRegression
from .Confidence_calibration.Post_hoc_methods.General_Methods.Class_specific_temperature_scaling import  Class_specific_temperature_scale_with_ece,Class_specific_temperature_scale_cross_entropy
from .Confidence_calibration.Post_hoc_methods.General_Methods.Dirichlet_calibration import dirichlet_calibration
from .Confidence_calibration.Post_hoc_methods.General_Methods.Matrix_scale import matrix_scale
from .Confidence_calibration.Post_hoc_methods.General_Methods.Mix_n_match import mix_n_match
from .Confidence_calibration.Post_hoc_methods.General_Methods.Spline_fitting import Spline_Calibration
from .Confidence_calibration.Post_hoc_methods.General_Methods.Intra_order_preserving import intra_order_preserving_model
from .Confidence_calibration.Post_hoc_methods.General_Methods.Parameterized_temperature import parameterized_temperature_scale
from .Confidence_calibration.Post_hoc_methods.General_Methods.Adaptive_temperature import adaptive_temperature_scale
#In-train
from .Confidence_calibration.In_training_methods.General_methods.Dual_Focal_Loss import Resnet110_DualFocalLoss,Wide_resnet34_2_DualFocalLoss


# Calibration under class imbalance data
#Calibration metrics
from .Calibration_metrics.Imbalance_metrics.ICE import ICE_strong,ICE_soft,ICE_smooth
from .Calibration_metrics.Imbalance_metrics.Class_wise_ECE import CECE,MSECE,WSECE
from .Calibration_metrics.Imbalance_metrics.RBECE import RBECE
# Data level calibration method
from .Confidence_calibration.Data_level_methods.For_imbalance_data.AUB_platt import AUB_Platt
from .Confidence_calibration.Data_level_methods.For_imbalance_data.UniMix import ResNet110_UniMix_BeyianLoss,DenseNet_UniMix_BeyianLoss

#post-hoc
from .Confidence_calibration.Post_hoc_methods.For_imbalance_data.TKHT import TKHT_2Head
from .Confidence_calibration.Post_hoc_methods.For_imbalance_data.CLS import CLS
from .Confidence_calibration.Post_hoc_methods.For_imbalance_data import RCIR
from .Confidence_calibration.Post_hoc_methods.For_imbalance_data import I_Max_sCW

#In-train
from .Confidence_calibration.In_training_methods.For_imbalance_data.TLC import MLP_TLCLoss,ResNet110_TLCLoss,DenseNet161_TLCLoss
from .Confidence_calibration.In_training_methods.For_imbalance_data.MHML import MLP_MHML_2,MLP_MHML_4,ResNet110_MHML_2,ResNet110_MHML_4,DenseNet161_MHML_2,DenseNet161_MHML_4
from .Confidence_calibration.In_training_methods.For_imbalance_data.BalPoE import MLP_BalPoE,Resnet110_BalPoE,DenseNet161_BalPoE
from .Confidence_calibration.In_training_methods.For_imbalance_data.MiSLAS import MLP_MiSLAS,ResNet110_MiSLAS,DenseNet161_MiSLAS
from .Confidence_calibration.In_training_methods.For_imbalance_data.LADE import MLP_LADE,ResNet110_LADE,DenseNet161_LADE
from .Confidence_calibration.In_training_methods.For_imbalance_data.MDCA import MLP_MDCA,ResNet110_MDCA,DenseNet161_MDCA




__all__ = list(globals().keys())
