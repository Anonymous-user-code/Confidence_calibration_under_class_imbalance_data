from PIL import Image
import os
import numpy as np
from typing import Any, Callable, cast, Optional, Tuple
from torchvision.datasets.utils import download_url
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torch.utils.data as torchdata
from customKing.data import DatasetCatalog

class USPS_train_valid_test(VisionDataset):
    """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.

    Args:
        root (string): Root directory of dataset to store``USPS`` data files.
        train (bool, optional): If True, creates dataset from ``usps.bz2``,
            otherwise from ``usps.t.bz2``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 'ec16c51db3855ca6c91edd34d0e9b197'
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", '8ea070ee2aca1ac39742fdd1ef5ed118'
        ],
    }

    def __init__(
            self,
            root: str,
            mode: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(USPS_train_valid_test, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        if mode == "train" or mode == "valid":
            split = 'train'
        else:
            split = "test"
        
        url, filename, checksum = self.split_list[split]
        full_path = os.path.join(self.root, filename)

        if download and not os.path.exists(full_path):
            download_url(url, self.root, filename, md5=checksum)

        import bz2
        with bz2.open(full_path) as fp:
            raw_data = [line.decode().split() for line in fp.readlines()]
            tmp_list = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
            imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
            imgs = ((cast(np.ndarray, imgs) + 1) / 2 * 255).astype(dtype=np.uint8)
            targets = [int(d[0]) - 1 for d in raw_data]

        self.data = imgs
        self.targets = targets

        num_valids = len(self.targets)//4

        if mode == "valid":
            self.data = self.data[-num_valids:]
            self.targets = self.targets[-num_valids:]
        if mode == "train":
            self.data = self.data[:-num_valids]
            self.targets = self.targets[:-num_valids]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
    
def load_USPS(name,root):
    transform_train = transforms.Compose([
            transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.Pad([4]),
            # transforms.RandomCrop(28),
            transforms.ToTensor()
            ])
    transform_test = transforms.Compose([
            transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
            ])
    if name == "USPS_train":
        dataset = USPS_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
    elif name == "USPS_valid":
        dataset = USPS_train_valid_test(root=root, mode="valid", download=True, transform=transform_test)
    elif name == "USPS_train_and_valid":
        dataset_train = USPS_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = USPS_train_valid_test(root=root, mode="valid", download=True, transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="USPS_test":
        dataset = USPS_train_valid_test(root=root, mode="test", download=True, transform=transform_test)
    if name == "USPS_train_and_valid_and_test":
        dataset_train = USPS_train_valid_test(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = USPS_train_valid_test(root=root, mode="valid", download=True, transform=transform_train) #验证数据集
        dataset_test = USPS_train_valid_test(root=root, mode="test", download=True, transform=transform_train) #验证数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid,dataset_test])
    return dataset

def register_USPS(name,root):
    DatasetCatalog.register(name, lambda: load_USPS(name,root))

