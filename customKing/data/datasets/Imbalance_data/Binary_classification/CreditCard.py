from customKing.data import DatasetCatalog
import torch.utils.data as torchdata
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class CreditcardDataset(Dataset):
    def __init__(self, root: str, mode: str, balance_test: bool = True) -> None:
        super().__init__()
        # 指定CSV文件路径
        data_file_path = os.path.join(root, r'creditcard.csv')
        
        # 读取CSV文件并解析数据
        data = pd.read_csv(data_file_path)
        x_list = data.drop(columns=["Class"]).values  # 特征数据
        y_list = data["Class"].values  # 标签数据

        
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_list, y_list, test_size=0.2, stratify=y_list, random_state=42
        )
        x_list, y_list = x_train_val, y_train_val

        # 剩余数据集分为训练集和验证集
        if mode != "test":
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_list, y_list, test_size=0.25, stratify=y_list, random_state=42
            )

        # 根据 mode 加载对应的数据集
        if mode == "train":
            self.data = np.array(x_train)
            self.labels = np.array(y_train)
        elif mode == "valid":
            self.data = np.array(x_valid)
            self.labels = np.array(y_valid)
        elif mode == "test":
            self.data = np.array(x_test)
            self.labels = np.array(y_test)
        else:
            raise ValueError("mode must be one of ['train', 'valid', 'test']")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_cls_num_list(self):
        pos_labels = self.labels[self.labels==1]
        neg_labels = self.labels[self.labels==0]
        return [len(neg_labels),len(pos_labels)]

def load_Creditcard(name, root):

    if name == "Creditcard_train":
        dataset = CreditcardDataset(root=root, mode="train")
    elif name == "Creditcard_valid":
        dataset = CreditcardDataset(root=root, mode="valid")
    elif name == "Creditcard_train_and_valid":
        dataset_train = CreditcardDataset(root=root, mode="train")
        dataset_valid = CreditcardDataset(root=root, mode="valid")
        dataset = torchdata.ConcatDataset([dataset_train, dataset_valid])
    elif name == "Creditcard_test":
        dataset = CreditcardDataset(root=root, mode="test")
    return dataset

def register_Creditcard(name, root):
    DatasetCatalog.register(name, lambda: load_Creditcard(name, root))
