from customKing.data import DatasetCatalog
import torch.utils.data as torchdata
from torch.utils.data import Dataset
import csv
import os


class Pami_train_valid_test(Dataset):
    def __init__(self,root: str,mode: str,) -> None:
        super().__init__()
        base_folder = r"pima-indians-diabetes-database/versions/1/diabetes.csv"
        csv_file_path = os.path.join(root, base_folder)
        x_list = []
        y_list = []
        with open(csv_file_path, 'r') as csvfile:
            # 创建CSV读取器
            csvreader = csv.reader(csvfile)

            # 读取标题行
            header = next(csvreader)

            # 遍历CSV文件中的每一行
            for row in csvreader:
                x = []
                for i in range(len(row[:-1])):
                    x.append(float(row[i]))
                x_list.append(x)
                y_list.append(float(row[-1])) 
        
        if mode=="train":
            self.data = x_list[:3*len(x_list)//5]
            self.labels = y_list[:3*len(x_list)//5]
        if mode == "valid":
            self.data = x_list[3*len(x_list)//5:4*len(x_list)//5]
            self.labels = y_list[3*len(x_list)//5:4*len(x_list)//5]
        if mode == "test":
            self.data = x_list[4*len(x_list)//5:]
            self.labels = y_list[4*len(x_list)//5:]



    def __getitem__(self, index):
        x,label = self.data[index],self.labels[index]

        return x, label

    def __len__(self) -> int:
        return len(self.data)



def load_Pami(name,root):

    if name == "Pami_train":
        dataset = Pami_train_valid_test(root=root, mode="train") #训练数据集

    elif name == "Pami_valid":
        dataset = Pami_train_valid_test(root=root, mode="valid")
    elif name == "Pami_train_and_valid":
        dataset_train = Pami_train_valid_test(root=root, mode="train") #训练数据集
        dataset_valid = Pami_train_valid_test(root=root, mode="valid") #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    elif name == "Pami_test":
        dataset = Pami_train_valid_test(root=root, mode="test") #训练数据集
    return dataset

def register_Pami(name,root):
    DatasetCatalog.register(name, lambda: load_Pami(name,root))