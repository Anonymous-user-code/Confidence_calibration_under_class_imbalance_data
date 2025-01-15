from torch.utils.data import Dataset
from customKing.data import DatasetCatalog
import torch
import numpy as np
from customKing.data.datasets.Image_classification.MNIST import MNIST_train_valid_test
from torchvision import transforms


class Sphere_OOD_test(Dataset):
    def __init__(self,root) -> None:
        super().__init__()

        transform_test = transforms.Compose([
            transforms.ToTensor()
            ])

        # 定义维度和样本数量
        dim = 784
        num_samples = 10000

        # 生成均匀分布在超球面上的样本
        samples = np.random.randn(num_samples, dim)
        samples /= np.linalg.norm(samples, axis=1)[:, np.newaxis]

        # 计算最大欧氏距离
        Dataset = MNIST_train_valid_test(root=root, mode="test", download=True, transform=transform_test)
        # in_samples = []
        # for image,_ in Dataset:
        #     in_samples.append(image.view(-1))

        in_samples = [data[0].view(-1) for data in Dataset]
        in_samples = np.stack(in_samples,axis=0)
        max_distance = np.max(np.linalg.norm(in_samples, axis=1))

        # 将样本放大到最大欧氏距离
        samples *= max_distance

        # 将样本重塑为28x28的图像
        self.images = torch.from_numpy(samples.reshape(-1, 28, 28)).unsqueeze(dim=1)

        #self.images = torch.normal(0.5, 1, size=(10000, 1, 28, 28))
    
    def __getitem__(self, index):
        return self.images[index],torch.tensor(0.)

    def __len__(self):
        return len(self.images)

def load_Sphere_OOD(name,root):
    if name =="Sphere_test":
        dataset = Sphere_OOD_test(root)
    return dataset

def register_Sphere_OOD(name,root):
    DatasetCatalog.register(name, lambda: load_Sphere_OOD(name,root))