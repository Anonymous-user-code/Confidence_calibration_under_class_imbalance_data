import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
import inspect
from customKing.modeling.meta_arch.build import build_metric
import torch.nn as nn

def numpy_to_tensor(cfg,z,y):
    z = torch.from_numpy(z)
    y = torch.from_numpy(y)
    if len(y.shape) == 2:
        y = y.squeeze(dim=1)
    z = z.to(cfg.MODEL.DEVICE)
    y = y.to(cfg.MODEL.DEVICE).long()
    return z,y

def logit_to_confidence(z,y):
    softmaxes = F.softmax(z, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    confidences,resort_index = torch.sort(confidences)
    labels = y[resort_index]
    predictions = predictions[resort_index]
    return confidences,predictions,labels

def compute_test_acc(cfg, Dataset, model):
    model.eval()
    result_list = []
    label_list = []
    confidence_list = []
    logit_list = []

    if isinstance(model, nn.DataParallel):
        output_uncertainty = model.module.output_uncertainty
    else:
        output_uncertainty = model.output_uncertainty

    with torch.no_grad():
        for batch_img,batch_label in tqdm(Dataset,dynamic_ncols=True):
            batch_label = batch_label.to(cfg.MODEL.DEVICE).clone().detach().float()
            batch_img = batch_img.to(cfg.MODEL.DEVICE).clone().detach().float()  
            if output_uncertainty:
                inference_result,uncertainty = model(batch_img,batch_label)
                logit_list.append(inference_result)
                inference_result = F.softmax(inference_result, dim=1)
                _, result = torch.max(inference_result.data, 1)
                result_list.append(result)
                label_list.append(batch_label)
                confidence_list.append(uncertainty)
            else:
                inference_result = model(batch_img,batch_label)
                logit_list.append(inference_result)
                inference_result = F.softmax(inference_result, dim=1)
                confidence, result = torch.max(inference_result.data, 1)
                result_list.append(result)
                label_list.append(batch_label)
                confidence_list.append(confidence)

    logits = torch.cat(logit_list,dim=0)
    results = torch.cat(result_list,dim=0)
    labels = torch.cat(label_list,dim=0)
    confidences = torch.cat(confidence_list,dim=0)
    hits = results.eq(labels)

    #------------compute Acc------------#
    acc = sum(hits)/len(hits)
    with open(cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
        f.write(json.dumps({"testAcc": acc.item()})+"\n")

    return logits,confidences,labels,hits,acc

def compute_metrics(cfg, Dataset, model=None, Final_record = False):
    if model is not None:
        if cfg.DATASET.IS_LOGITS_DATA:
            if len(Dataset) == 2:
                logits, labels = Dataset
            elif len(Dataset) == 3:
                feature,logits, labels = Dataset
            model.eval()
            cali_confidence,predictions,labels = model(logits,labels)   # tensor
            hits = predictions.eq(labels)
            cali_confidence = cali_confidence.cpu().detach().numpy()
            hits = hits.cpu().detach().numpy().astype(int)
            labels = labels.cpu().detach().numpy().astype(int)
            model.train()
        else:
            logits,cali_confidence,labels,hits,acc = compute_test_acc(cfg, Dataset, model)
            cali_confidence = cali_confidence.cpu().detach().numpy()
            hits = hits.cpu().detach().numpy().astype(int)
            labels = labels.cpu().detach().numpy().astype(int)

        metric_values = {}
        for metric in cfg.EVALUATE.METHOD_list:
            metric_method = build_metric(cfg,metric)
            signature = inspect.signature(metric_method.forward)
            names = []
            for name, param in signature.parameters.items():
                names.append(name)
            if len(names)==2:
                metric_value = metric_method(cali_confidence, hits)
            else:
                metric_value = metric_method(cali_confidence,hits,labels)
            metric_values[metric] = metric_value
    else:
        if len(Dataset) == 2:
            logits, labels = Dataset
        elif len(Dataset) == 3:
            feature,logits, labels = Dataset

        logits,labels = numpy_to_tensor(cfg,logits,labels)
        confidence,predictions,labels = logit_to_confidence(logits,labels)
        hits = predictions.eq(labels)
        confidence = confidence.cpu().detach().numpy()
        hits = hits.cpu().detach().numpy().astype(int)
        labels = labels.cpu().detach().numpy().astype(int)
        metric_values = {}
        for metric in cfg.EVALUATE.METHOD_list:
            metric_method = build_metric(cfg,metric)
            signature = inspect.signature(metric_method.forward)
            names = []
            for name, param in signature.parameters.items():
                names.append(name)
            if len(names)==2:
                metric_value = metric_method(confidence, hits)
            else:
                metric_value = metric_method(confidence,hits,labels)
            metric_values[metric] = metric_value

    if Final_record:
        with open(cfg.MODEL.OUTPUT_DIR+r"Final_logging.json",mode="a") as f:
            f.write(json.dumps(metric_values)+"\n")
            if hasattr(model,"temperature"):
                f.write(json.dumps({"temperature":model.temperature.item()}))
    else:
        with open(cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
            if model is None:
                f.write("Uncalibration:"+"\n")
            f.write(json.dumps(metric_values)+"\n")