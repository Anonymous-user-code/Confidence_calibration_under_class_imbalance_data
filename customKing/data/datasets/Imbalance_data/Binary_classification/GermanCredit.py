from customKing.data import DatasetCatalog
import torch.utils.data as torchdata
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class credit_train_valid_test(Dataset):
    def __init__(self, root: str, mode: str) -> None:
        super().__init__()
        # 修改文件路径为 .data 文件路径
        base_folder = r"statlog+german+credit+data/german.data-numeric"
        data_file_path = os.path.join(root, base_folder)
        
        x_list = []
        y_list = []

        # 读取 .data 文件并解析数据
        with open(data_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # 假设每行数据用空格分隔
                values = line.strip().split()
                x = [float(value) for value in values[:-1]]  # 特征数据
                x_list.append(x)
                y_list.append(float(values[-1])-1)  # 标签数据
        
        # 转换为 numpy 数组，方便后续处理
        x_list = np.array(x_list)
        x_list = (x_list - x_list.min(axis=0)) / (x_list.max(axis=0) - x_list.min(axis=0))
        y_list = np.array(y_list)
        
        # 使用分层抽样进行训练集、验证集和测试集的划分
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_list, y_list, test_size=0.4, stratify=y_list, random_state=42
        )
        
        # 从临时数据中进一步分出验证集和测试集
        x_valid, x_test, y_valid, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # 根据 mode 加载对应的数据集
        if mode == "train":
            self.data = x_train
            self.labels = y_train
        elif mode == "valid":
            self.data = x_valid
            self.labels = y_valid
        elif mode == "test":
            self.data = x_test
            self.labels = y_test
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

def load_Credit(name,root):

    if name == "Credit_train":
        dataset = credit_train_valid_test(root=root, mode="train") #训练数据集

    elif name == "Credit_valid":
        dataset = credit_train_valid_test(root=root, mode="valid")
    elif name == "Credit_train_and_valid":
        dataset_train = credit_train_valid_test(root=root, mode="train") #训练数据集
        dataset_valid = credit_train_valid_test(root=root, mode="valid") #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    elif name == "Credit_test":
        dataset = credit_train_valid_test(root=root, mode="test") #训练数据集
    return dataset

def register_Credit(name,root):
    DatasetCatalog.register(name, lambda: load_Credit(name,root))