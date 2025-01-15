import logging
import torch
import time
import numpy as np
from customKing.solver.build import build_optimizer,build_lr_scheduler
import json
from compute_metrics import compute_test_acc, compute_metrics
import inspect
import torch.nn as nn
import os
import shutil

def numpy_to_tensor(cfg,z,y):
    z = torch.from_numpy(z)
    y = torch.from_numpy(y)
    if len(y.shape) == 2:
        y = y.squeeze(dim=1)
    z = z.to(cfg.MODEL.DEVICE)
    y = y.to(cfg.MODEL.DEVICE).long()
    return z,y

class doTrain():
    def __init__(self,cfg):
        self.iteration = cfg.SOLVER.START_ITER
        self.cfg = cfg

    def do_train(self,train_data,valid_data,test_data,model,init_epoch = 0,init_stage = 1):
        logging.basicConfig(level=logging.INFO)
        
        if self.cfg.CALIBRATION.NEED_CLASSIFIER_TRAIN:
            # The process of training classifier
            assert self.cfg.DATASET.IS_LOGITS_DATA == False, "Must be original data!"
            if model.Two_stage:
                stages = [1,2]
            else:
                stages = [1]
            
            if self.cfg.MODEL.PRE_WEIGHT != "" and self.cfg.MODEL.PRE_WEIGHT != None:
                init_iteration = (init_epoch+1)*len(train_data)
                len_step = len(self.cfg.SOLVER.STEPS)
                for i in range(len_step):
                    if init_iteration < self.cfg.SOLVER.STEPS[i]:
                        self.cfg.SOLVER.STEPS[i] = self.cfg.SOLVER.STEPS[i] - init_iteration
                        break
                    else:
                        self.cfg.SOLVER.STEPS[i] = None
                        self.cfg.SOLVER.BASE_LR = self.cfg.SOLVER.BASE_LR/10
                self.cfg.SOLVER.STEPS = list(filter(lambda x: x is not None, self.cfg.SOLVER.STEPS))
                self.iteration = self.iteration + init_iteration

            optimizer = torch.optim.SGD(model.parameters(),lr=self.cfg.SOLVER.BASE_LR,momentum=self.cfg.SOLVER.MOMENTUM,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,nesterov=self.cfg.SOLVER.NESTEROV)
            Need_epoch = model.need_epoch
            
            scheduler = build_lr_scheduler(self.cfg.SOLVER, optimizer) 
            if self.cfg.SOLVER.PARALLEL_TRAIN:
                model = nn.DataParallel(model,device_ids=self.cfg.SOLVER.DEVICE_LIST)

            for stage in stages:

                if init_stage != 1 and stage ==1:
                    continue

                for epoch in range(init_epoch+1, self.cfg.SOLVER.MAX_EPOCH):
                    if self.iteration > self.cfg.SOLVER.MAX_ITER:
                        break
                    log_iteration = len(train_data) if (len(train_data) < 20) else 20
                    for batch_img,batch_label in train_data:
                        model.train()
                        self.iteration = self.iteration + 1

                        batch_img = batch_img.to(self.cfg.MODEL.DEVICE).clone().detach().float()    
                        batch_label = batch_label.to(self.cfg.MODEL.DEVICE).float().long() 
                        
                        if len(stages)==2:
                            output = model(batch_img,batch_label,stage)
                        else:
                            if Need_epoch:
                                output = model(batch_img,batch_label,epoch)
                            else:
                                output = model(batch_img,batch_label)

                        if self.cfg.SOLVER.PARALLEL_TRAIN:
                            if isinstance(output,tuple):
                                predict = output[0]
                                losses = output[1]
                                loss = losses.mean()
                            else:
                                loss = output.mean()
                        else:
                            if isinstance(output,torch.Tensor):
                                loss = output
                            elif len(output) == 2:
                                predict,loss = output
                        optimizer.zero_grad()
                        loss.backward()   
                        optimizer.step() 

                        #---------Calculate the training accuracy (that is, the accuracy within a batch)-----#
                        if "predict" in locals():
                            _, predicted = torch.max(predict.data, 1)
                            correct = predicted.eq(batch_label).cpu().sum()
                            train_acc = correct / batch_img.shape[0]
                            train_acc = train_acc.item()
                        else:
                            train_acc = None

                        scheduler.step()

                        if self.iteration % log_iteration ==0:
                            with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                                if len(stages)==2:
                                    f.write(json.dumps({"Stage": stage, "Epoch":epoch, "Iteration":self.iteration, "train_acc":train_acc, "loss_value":loss.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")
                                else:
                                    f.write(json.dumps({"Epoch":epoch, "Iteration":self.iteration, "train_acc":train_acc, "loss_value":loss.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")

                    compute_test_acc(self.cfg, test_data, model)

                    root_path = os.path.expanduser(os.getenv("CUSTOM_KING_WEGHTS"))
                    folder_path = model_path = os.path.join(root_path,r"Confidence_calibration_under_imbalance_data/"+self.cfg.MODEL.OUTPUT_DIR)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    model_path = os.path.join(folder_path,self.cfg.MODEL.META_ARCHITECTURE+f"_{stage}_{epoch}.pth")
                    if self.cfg.SOLVER.PARALLEL_TRAIN:
                        torch.save(model.module.state_dict(), model_path)
                    else:
                        torch.save(model.state_dict(), model_path)

        if self.cfg.CALIBRATION.NEED_CALIBRATION_TRAIN:
            assert self.cfg.DATASET.IS_LOGITS_DATA == True, "Must be logits data!"
            if self.cfg.CALIBRATION.ITERATIVE_TRAINING:
                optimizer = torch.optim.SGD(model.parameters(),lr=self.cfg.SOLVER.BASE_LR,momentum=self.cfg.SOLVER.MOMENTUM,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,nesterov=self.cfg.SOLVER.NESTEROV)
                scheduler = build_lr_scheduler(self.cfg.SOLVER, optimizer) 
                for epoch in range(self.cfg.SOLVER.MAX_EPOCH):
                    if len(valid_data) == 2:
                        z_val, y_val = valid_data
                    else:
                        feature_val,z_val, y_val = valid_data
                    z_val,y_val = numpy_to_tensor(self.cfg,z_val,y_val)
                    loss_value= model(z_val,y_val)
                    optimizer.zero_grad()    
                    loss_value.backward()    
                    optimizer.step()

                    with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                        f.write(json.dumps({"Epoch":epoch,"loss_value":loss_value.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")

                    #-----------Compute Metrics After Calibration-----------#
                    compute_metrics(self.cfg,test_data,model=model)
                    
                    scheduler.step()
            else:
                if len(valid_data) == 2:
                    z_val, y_val = valid_data
                    z_val,y_val = numpy_to_tensor(self.cfg,z_val,y_val)
                    model.train()
                    model(z_val,y_val)
                elif len(valid_data) == 3:
                    feature_val,z_val, y_val = valid_data
                    model.train()
                    signature = inspect.signature(model.forward)
                    names = []
                    for name, param in signature.parameters.items():
                        names.append(name)
                    if len(names) == 5:
                        feature_train,z_train,y_train = train_data
                        model(z_val,y_val,feature_val,feature_train,y_train)
                    elif len(names) == 4:
                        feature_train,z_train,y_train = train_data
                        model(z_val,y_val,z_train,y_train)
                    elif len(names) == 2:
                        model(z_val,y_val)