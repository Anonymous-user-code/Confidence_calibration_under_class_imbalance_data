import numpy as np
import pickle
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from PIL import Image
import os
from torchvision import transforms
import torch.utils.data as torchdata
from customKing.data import DatasetCatalog

HEIGHT = 32
WIDTH = 32
DEPTH = 3

def _read_data(files):
    """Reads CIFAR-10 format data. Always returns NHWC format.

    Returns:
        images: np tensor of size [N, H, W, C]
        labels: np tensor of size [N]
    """
    images, labels = [], []
    for file_name in files:
        with open(file_name, 'rb') as finp:
            data = pickle.load(finp, encoding='bytes')
            batch_images = data[b'data']#.astype(np.float32) / 255.0
            if 'cifar-100' in file_name:
                batch_labels = np.array(data[b'fine_labels'], dtype=np.int32)
            else:
                batch_labels = np.array(data[b'labels'], dtype=np.int32)
            images.append(batch_images)
            labels.append(batch_labels)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    images = np.reshape(images, [-1, 3, WIDTH, HEIGHT])
    images = np.transpose(images, [0, 2, 3, 1])

    return images, labels

def read_data(data_path, num_valids=5000):
    print("Reading data from {}".format(data_path))

    images, labels = {}, {}

    if 'cifar-100' in data_path:
        train_files = [
            os.path.join(data_path, 'train')
        ]
        test_file = [
            os.path.join(data_path, 'test')
        ]
    else:
        train_files = [
            os.path.join(data_path, 'data_batch_1'),
            os.path.join(data_path, 'data_batch_2'),
            os.path.join(data_path, 'data_batch_3'),
            os.path.join(data_path, 'data_batch_4'),
            os.path.join(data_path, 'data_batch_5'),
        ]
        test_file = [
            os.path.join(data_path, 'test_batch'),
        ]
    images['train'], labels['train'] = _read_data(train_files)

    if num_valids:
        images['valid'] = images['train'][-num_valids:]
        labels['valid'] = labels['train'][-num_valids:]

        images['train'] = images['train'][:-num_valids]
        labels['train'] = labels['train'][:-num_valids]
    else:
        images['valid'], labels['valid'] = None, None

    images['test'], labels['test'] = _read_data(test_file)

    return images, labels


class Cifar100_train_valid_test(Dataset):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    data_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            root: str,
            mode: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        self.root = root
        self.mode = mode  #mode is train, valid, or test
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self._load_meta()
        data,label = read_data(os.path.join(root, self.base_folder))
        self.data = data[self.mode]
        self.label = label[self.mode]

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.data_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        
    def __getitem__(self, index):
        image,label = self.data[index],self.label[index]
        img = Image.fromarray(image)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self) -> int:
        return len(self.data)

def load_Cifar100(name,root):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


    if name == "Cifar100_train":
        dataset = Cifar100_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集

    elif name == "Cifar100_valid":
        dataset = Cifar100_train_valid_test(root=root, mode="valid", download=True, transform=transform_test)
    elif name == "Cifar100_train_and_valid":
        dataset_train = Cifar100_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = Cifar100_train_valid_test(root=root, mode="valid", download=True, transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="Cifar100_test":
        dataset = Cifar100_train_valid_test(root=root, mode="test", download=True, transform=transform_test)
    if name == "Cifar10_train_and_valid_and_test":
        dataset_train = Cifar100_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = Cifar100_train_valid_test(root=root, mode="valid", download=True, transform=transform_train) #验证数据集
        dataset_test = Cifar100_train_valid_test(root=root, mode="test", download=True, transform=transform_train) #验证数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid,dataset_test])
    return dataset

def register_Cifar100(name,root):
    DatasetCatalog.register(name, lambda: load_Cifar100(name,root))