from unicodedata import name
from customKing.data import DatasetCatalog
import os
import numpy as np
import pickle
import torchvision
from torchvision import transforms
import torch.utils.data as torchdata
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.utils import download_and_extract_archive, download_url,verify_str_arg
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import pathlib


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        mode: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.mode = mode  #mode is train, valid, or test
        if self.mode == "valid":
            split = "test"

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        if self.mode != "train":
            samples_list = [[] for j in range(196)]
            for sample in self._samples:
                samples_list[sample[1]].append(sample)

            samples = []
            for i in range(len(samples_list)):
                if self.mode == "valid":
                    valid_len = len(samples_list[i])//2
                    for sample in samples_list[i][:-valid_len]:
                        samples.append(sample)
                elif self.mode == "test":
                    valid_len = len(samples_list[i])//2
                    for sample in samples_list[i][-valid_len:]:
                        samples.append(sample)
            self._samples = samples

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False
        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


def load_CARS(name,root):
    transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Pad([28]),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(), #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])
    
    transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  #R,G,B每层的归一化用到的均值和方差
            ])

    if name == "CARS_train":
        dataset = StanfordCars(root=root, mode="train", download=True, transform=transform_train) #训练数据集
    elif name == "CARS_valid":
        dataset = StanfordCars(root=root, mode="valid", download=True, transform=transform_test)
    elif name == "CARS_train_and_valid":
        dataset_train = StanfordCars(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = StanfordCars(root=root, mode="valid", download=True, transform=transform_train) #训练数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid])
    if name =="CARS_test":
        dataset = StanfordCars(root=root, mode="test", split="test", download=True, transform=transform_test)
    if name == "CARS_train_and_valid_and_test":
        dataset_train = StanfordCars(root=root, mode="train", download=True, transform=transform_train) #训练数据集
        dataset_valid = StanfordCars(root=root, mode="valid", download=True, transform=transform_train) #验证数据集
        dataset_test = StanfordCars(root=root, mode="test", split="test", download=True, transform=transform_train) #验证数据集
        dataset = torchdata.ConcatDataset([dataset_train,dataset_valid,dataset_test])
    return dataset

def register_CARS(name,root):
    DatasetCatalog.register(name, lambda: load_CARS(name,root))