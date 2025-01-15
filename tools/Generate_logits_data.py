import torch
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_model

import time
import json
import torch
import logging
from customKing.modeling.meta_arch.build import build_model
from customKing.solver.build import build_optimizer,build_lr_scheduler
from customKing.data import get_dataset_dicts,build_loader 
from customKing.engine import default_writers
import customKing.utils.comm as comm
from customKing.utils.events import EventStorage
import time
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import json
from torch.nn import DataParallel
import os

import pickle

class doTest():
    def __init__(self,cfg) -> None:
        self.cfg = cfg

    def do_test(self,model,test_data):
        result_list = []
        label_list = []
        model.eval()
        with torch.no_grad():
            for batch_img,batch_label in test_data:
                batch_img = batch_img.to(self.cfg.MODEL.DEVICE).clone().detach().float()  
                inference_result = model(batch_img,batch_label)
                _, result = torch.max(inference_result.data, 1)
                result_list = result_list + result.tolist()
                label_list = label_list + batch_label.tolist()
        correct = 0
        for i in range(len(label_list)):
            if label_list[i] == result_list[i]:
                correct = correct + 1
        test_acc = correct / len(label_list)
        logging.info(f"test_acc: {test_acc}")
        with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
            f.write(json.dumps({"test_acc": test_acc})+"\n")

class doTrain():
    def __init__(self,cfg):
        self.iteration = cfg.SOLVER.START_ITER
        self.cfg = cfg
        self.best_model = None
        self.testClass = doTest(cfg)

    def Inference_Output(self,trainData,validData,testData,model):
        model.eval()
        if "/" in self.cfg.MODEL.OUTPUT_DIR:
            filename = self.cfg.MODEL.OUTPUT_DIR.split("/")
        elif "\\" in self.cfg.MODEL.OUTPUT_DIR:
            filename = self.cfg.MODEL.OUTPUT_DIR.split("\\")

        torch.multiprocessing.set_sharing_strategy('file_system')

        with torch.no_grad():
            train_logits = []
            train_labels = []
            train_features = []
            for batch_img,batch_label in trainData:
                batch_img = batch_img.to(self.cfg.MODEL.DEVICE).clone().detach().float()
                batch_feature = model.forward_backbone(batch_img).to(torch.float64)
                batch_logit = model(batch_img,batch_label).to(torch.float64)
                train_features.append(batch_feature)
                train_logits.append(batch_logit)
                train_labels.append(batch_label)
            train_features = torch.cat(train_features,dim=0).cpu().detach().numpy()
            train_logits = torch.cat(train_logits,dim=0).cpu().detach().numpy()
            train_labels = torch.cat(train_labels,dim=0).cpu().detach().numpy()

            valid_logits = []
            valid_labels = []
            valid_features = []
            for batch_img,batch_label in validData:
                batch_img = batch_img.to(self.cfg.MODEL.DEVICE).clone().detach().float()
                batch_feature = model.forward_backbone(batch_img).to(torch.float64)
                batch_logit = model(batch_img,batch_label).to(torch.float64)
                valid_features.append(batch_feature)
                valid_logits.append(batch_logit)
                valid_labels.append(batch_label)
            valid_features = torch.cat(valid_features,dim=0).cpu().detach().numpy()
            valid_logits = torch.cat(valid_logits,dim=0).cpu().detach().numpy()
            valid_labels = torch.cat(valid_labels,dim=0).cpu().detach().numpy()

            test_logits = []
            test_labels = []
            test_features = []
            for batch_img,batch_label in testData:
                batch_img = batch_img.to(self.cfg.MODEL.DEVICE).clone().detach().float()
                batch_feature = model.forward_backbone(batch_img).to(torch.float64)
                batch_logit = model(batch_img,batch_label).to(torch.float64)
                test_features.append(batch_feature)
                test_logits.append(batch_logit)
                test_labels.append(batch_label)
            test_features = torch.cat(test_features,dim=0).cpu().detach().numpy()
            test_logits = torch.cat(test_logits,dim=0).cpu().detach().numpy()
            test_labels = torch.cat(test_labels,dim=0).cpu().detach().numpy()

        data = ((train_features,train_logits,train_labels),(valid_features, valid_logits, valid_labels), (test_features, test_logits, test_labels))
        _root = os.path.expanduser(os.getenv("CUSTOM_KING_DATASETS", "datasets"))
        save_path = os.path.join(_root,"Generate_logits_datasets")
        with open(save_path+r"/"+self.cfg.DATASETS.TRAIN.split("_")[0] +"_"+ self.cfg.MODEL.META_ARCHITECTURE + ".p", 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(data, f)

    def do_train(self,train_data,valid_data,test_data,model,init_epoch = 0):
        logging.basicConfig(level=logging.INFO) 
        model.train() 

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

        optimizer = build_optimizer(self.cfg.SOLVER, model) 
        scheduler = build_lr_scheduler(self.cfg.SOLVER, optimizer) 

        if self.cfg.SOLVER.IS_PARALLEL:
            model = DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])        

        for epoch in range(init_epoch+1,self.cfg.SOLVER.MAX_EPOCH):
            time1 = time.time()
            log_iteration = len(train_data)-1 if (len(train_data) < 20) else 20
            for batch_img,batch_label in train_data:
                self.iteration = self.iteration + 1
                if self.iteration > self.cfg.SOLVER.MAX_ITER:
                    break
                batch_img = batch_img.to(self.cfg.MODEL.DEVICE).clone().detach().float()    
                batch_label = batch_label.to(self.cfg.MODEL.DEVICE).float().long()                        
                logit,losses = model(batch_img,batch_label)  
                time2 = time.time()
                optimizer.zero_grad()   
                if self.cfg.SOLVER.IS_PARALLEL:
                    loss = losses.sum()/len(losses)
                    loss.backward()
                else:
                    loss = losses
                    loss.backward()   
                optimizer.step()  
                scheduler.step()

                #---------Calculate the training accuracy (that is, the accuracy within a batch)-----#
                _, predicted = torch.max(logit.data, 1)
                correct = predicted.eq(batch_label).cpu().sum()
                train_acc = correct / batch_img.shape[0]

                #--------------Record and update the learning rate--------------#
                if self.iteration % log_iteration == 0:
                    with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                        f.write(json.dumps({"Epoch":epoch, "Iteration":self.iteration, "train_acc":train_acc.item(), "loss_value":loss.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")

                if self.iteration > self.cfg.SOLVER.MAX_ITER:
                    break
            # self.testClass.do_test(model,test_data)  #Evaluate on the test set every other epoch
            # model.train()
            root_path = os.path.expanduser(os.getenv("CUSTOM_KING_WEGHTS"))
            folder_path = model_path = os.path.join(root_path,r"Confidence_calibration_under_imbalance_data/"+self.cfg.MODEL.OUTPUT_DIR)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            model_path = os.path.join(folder_path,self.cfg.MODEL.META_ARCHITECTURE+f"_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
                
        #Record inference results on the training set (for model calibration)
        self.Inference_Output(train_data,valid_data,test_data,model)

def main():
        task_mode = "Classification"
        cfg = get_cfg(task_mode)
        Set_seed(seed=cfg.SEED)
        if not os.path.exists(cfg.MODEL.OUTPUT_DIR):
            os.makedirs(cfg.MODEL.OUTPUT_DIR)
        #save cfg
        if os.path.exists(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml')):
            os.remove(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'))
        with open(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'), 'a+') as f:
            print(cfg, file=f)

        #-----------Load Data------------------#
        train_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.TRAIN))
        cls_num_list_train = train_data.dataset.get_cls_num_list()
        valid_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.VALID))
        cls_num_list_valid = valid_data.dataset.get_cls_num_list()
        test_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.TEST))
        cls_num_list_test = test_data.dataset.get_cls_num_list()

        #-----------Load model-----------------#
        model = build_model(cfg,cls_num_list_train,cls_num_list_test)
        if cfg.MODEL.PRE_WEIGHT != "":
            model.load_state_dict(torch.load(cfg.MODEL.PREWEIGHT))
            init_epoch = os.path.basename(cfg.MODEL.PREWEIGHT).split(".")[0].split("_")[-1]
            init_epoch = int(init_epoch)
        else:
            init_epoch = 0
        logging.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        model.to(cfg.MODEL.DEVICE)

        DoTrain = doTrain(cfg)
        DoTrain.do_train(train_data,valid_data,test_data,model,init_epoch)
        print("finished training!")

        
if __name__ == "__main__":
    main()