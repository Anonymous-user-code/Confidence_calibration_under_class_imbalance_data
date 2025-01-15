import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from torchvision.datasets.folder import ImageFolder
from customKing.data import DatasetCatalog
import torch.utils.data as torchdata
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np


class ImageNet_train_valid_test(Dataset):
    def __init__(self,root,mode,transform=None,target_transform=None) -> None:
        super().__init__()

        main_folder_path = os.path.join(root,"ImageNet2012")
        meta = loadmat(os.path.join(main_folder_path,"meta.mat"))
        data = meta["synsets"]
        Winds_to_class_dict = {}
        for i in range(1000):
            Winds_to_class_dict[data[i][0][1][0]] = data[i][0][0][0][0]-1

        self.data = []
        self.label = []
        self.mode = mode

        if mode == "test":
            folder_path = os.path.join(main_folder_path,"val")
            class_to_samples = {}
            for entry in os.listdir(folder_path):
                entry_path = os.path.join(folder_path,entry)
                for sample in os.listdir(entry_path):
                    file_path = os.path.join(entry_path,sample)
                    self.data.append(file_path)
                    self.label.append(Winds_to_class_dict[entry])
        else:
            folder_path = os.path.join(main_folder_path,"train")
            class_to_samples = {}
            for entry in os.listdir(folder_path):
                class_to_samples[Winds_to_class_dict[entry]] = []
                entry_path = os.path.join(folder_path,entry)
                for sample in os.listdir(entry_path):
                    file_path = os.path.join(entry_path,sample)
                    class_to_samples[Winds_to_class_dict[entry]].append(file_path)

            if mode == "train":
                for i in range(1000):
                    class_to_samples[i] = class_to_samples[i][:9*len(class_to_samples[i])//10]
            elif mode == "valid":
                for i in range(1000):
                    class_to_samples[i] = class_to_samples[i][9*len(class_to_samples[i])//10:]

            for i in range(1000):
                for file_path in class_to_samples[i]:
                    self.data.append(file_path)
                    self.label.append(i)

            imb_type="exp"
            imb_factor = 1/256
            self.img_num_list = self.get_img_num_per_cls(1000, imb_type, imb_factor)
            self.gen_imbalanced_data(self.img_num_list)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data[index]
        label = self.label[index]
        pil_image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(pil_image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        if self.mode != "test":
            img_max = len(self.data) / cls_num
            img_num_per_cls = []
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max * imb_factor))
            else:
                img_num_per_cls.extend([int(img_max)] * cls_num)
            return img_num_per_cls
        
    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.label, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            for idx in selec_idx:
                new_data.append(self.data[idx])
                new_targets.append(self.label[idx])
        self.data = new_data
        self.label = new_targets

    def get_cls_num_list(self):
        if self.mode == "test":
            img_num_list = []
            label = np.array(self.label)
            for i in range(1000):
                img_num_list.append(sum(label == i))
            return img_num_list
        else:
            return self.img_num_list

def load_ImageNet(name,root):
    transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Pad([28]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])
    
    transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    if name == "ImageNet_train_LT":
        dataset = ImageNet_train_valid_test(root=root, mode="train", transform=transform_train) #训练数据集
    elif name == "ImageNet_valid_LT":
        dataset = ImageNet_train_valid_test(root=root, mode="valid", transform=transform_test)
    elif name == "ImageNet_train_and_valid_LT":
        dataset_train = ImageNet_train_valid_test(root=root, mode="train", transform=transform_train) #训练数据集
        dataset_valid = ImageNet_train_valid_test(root=root, mode="valid", transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="ImageNet_test":
        dataset = ImageNet_train_valid_test(root=root, mode="test", transform=transform_test)
    return dataset

def register_ImageNet_LT(name,root):
    DatasetCatalog.register(name, lambda: load_ImageNet(name,root))