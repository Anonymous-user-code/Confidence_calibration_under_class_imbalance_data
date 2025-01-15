
from torch.utils.data import Dataset,DataLoader
from customKing.data import DatasetCatalog
from unicodedata import name
import os
from torchvision import transforms
import torch.utils.data as torchdata
from torchvision.io import read_image
import torch
from PIL import Image

class BIRDS_Dataset(Dataset):

    def __init__(self,root,mode="train", transform = None,target_transform=None) -> None:
        super().__init__()
        self.root = root
        self.folder_path = os.path.join(self.root,r"CUB_200_2011/images")
        self.transform = transform
        self.target_transform = target_transform

        self.images,self.labels = self.load_data(self.folder_path,mode)

        
    def load_data(self,folder_path,mode):
        subfolder_list = os.listdir(folder_path)
        label_list = [int(name[:3])-1 for name in subfolder_list]

        images_list = [[] for j in range(len(subfolder_list))]

        for i in range(len(subfolder_list)):
            file_list = os.listdir(os.path.join(folder_path,subfolder_list[i]))
            for file in file_list:
                file_path = os.path.join(os.path.join(folder_path,subfolder_list[i]),file)
                img = Image.open(file_path).convert("RGB")
                images_list[label_list[i]].append(img)


        if mode == "train":
            images = []
            labels = []
            for i in range(len(images_list)):
                data_len = len(images_list[i])
                train_len = data_len*2//4
                for lt in images_list[i][:-train_len]:
                    images.append(lt)
                    labels.append(i)
        elif mode == "valid":
            images = []
            labels = []
            for i in range(len(images_list)):
                data_len = len(images_list[i])
                train_len = data_len*2//4
                valid_len = data_len//4
                for lt in images_list[i][-train_len:-valid_len]:
                    images.append(lt)
                    labels.append(i)
        elif mode == "test":
            images = []
            labels = []
            for i in range(len(images_list)):
                data_len = len(images_list[i])
                valid_len = data_len//3
                for lt in images_list[i][-valid_len:]:
                    images.append(lt)
                    labels.append(i)
        return images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img,label = self.images[index],self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label        

def load_BIRDS(name,root):
    transform_train = transforms.Compose([
            transforms.Resize((448,448)),
            # transforms.Pad([56]),
            # transforms.RandomCrop(448),
            # transforms.RandomHorizontalFlip(), #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])
    
    transform_test = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])

    if name == "BIRDS_train":
        dataset = BIRDS_Dataset(root=root, mode="train", transform=transform_train) #训练数据集

    elif name == "BIRDS_valid":
        dataset = BIRDS_Dataset(root=root, mode="valid", transform=transform_test) #训练数据集

    if name == "BIRDS_train_and_valid":
        dataset_train = BIRDS_Dataset(root=root, mode="train", transform=transform_train) #训练数据集
        dataset_valid = BIRDS_Dataset(root=root, mode="valid", transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="BIRDS_test":
        dataset = BIRDS_Dataset(root=root, mode="test", transform=transform_test)
    if name == "BIRDS_train_and_valid_and_test":
        dataset_train = BIRDS_Dataset(root=root, mode="train", transform=transform_train) #训练数据集
        dataset_valid = BIRDS_Dataset(root=root, mode="valid", transform=transform_train) #验证数据集
        dataset_test = BIRDS_Dataset(root=root, mode="test", transform=transform_train) #验证数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid,dataset_test])
    return dataset

def register_BIRDS(name,root):
    DatasetCatalog.register(name, lambda: load_BIRDS(name,root))
    

if __name__=="__main__":
    data = DatasetCatalog.get(name)
    print(data)
