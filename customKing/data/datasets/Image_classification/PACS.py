from torch.utils.data import Dataset
from customKing.data import DatasetCatalog
import torch
import hub
import numpy as np
from torchvision import transforms
import os

class PACS_train_test(Dataset):
    def __init__(self,mode = "test",transform = None,root=None) -> None:
        super().__init__()

        if root == None or not os.path.exists(os.path.join(os.path.join(root,"PACS"),mode)):
            ds_train = hub.load('hub://activeloop/pacs-train')
            self.images = []
            self.labels = []
            self.domains = []
            for data in ds_train:
                if mode == "train":
                    domain = data["domains"].numpy()
                    if domain[0] != 3:
                        image = data["images"]
                        label = data["labels"]
                        self.images.append(image.numpy().astype(np.float32))
                        self.labels.append(label.numpy()[0].astype(np.float32))
                elif mode == "test":
                    domain = data["domains"].numpy()
                    if domain[0] == 3:
                        image = data["images"]
                        label = data["labels"]
                        self.images.append(image.numpy().astype(np.float32))
                        self.labels.append(label.numpy()[0].astype(np.float32))
            self.images = np.stack(self.images,axis=0)
            self.labels = np.stack(self.labels,axis=0)
            os.makedirs(os.path.join(os.path.join(root,"PACS"),mode))
            np.save(os.path.join(os.path.join(os.path.join(root,"PACS"),mode),mode+"_image.npy"),self.images)
            np.save(os.path.join(os.path.join(os.path.join(root,"PACS"),mode),mode+"_label.npy"),self.labels)
            self.images = torch.from_numpy(self.images).permute(0,3,1,2)
        else:
            self.images = np.load(os.path.join(os.path.join(os.path.join(root,"PACS"),mode),mode+"_image.npy"))
            self.labels = np.load(os.path.join(os.path.join(os.path.join(root,"PACS"),mode),mode+"_label.npy"))
            self.images = torch.from_numpy(self.images).permute(0,3,1,2)

        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = torch.tensor(self.labels[index])
        if self.transform != None:
            image = self.transform(image)
        return image,label

    def __len__(self):
        return len(self.images)

def load_PACS(name,root):
    transform_train = transforms.Compose([
            transforms.Resize([224,224]),
            # transforms.Pad([28]),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(), #图像一半的概率翻转，一半的概率不翻转
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])
    transform_test = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    if name =="PACS_train":
        dataset = PACS_train_test(mode="train",transform = transform_train,root=root)
    if name =="PACS_test":
        dataset = PACS_train_test(mode="test",transform=transform_test,root=root)
    return dataset

def register_PACS(name,root):
    DatasetCatalog.register(name, lambda: load_PACS(name,root))