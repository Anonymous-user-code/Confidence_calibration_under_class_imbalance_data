from torch.utils.data import Dataset
from customKing.data import DatasetCatalog
import torch

class Uniform_Noise_test(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.images = torch.rand(10000,1,28,28)
    
    def __getitem__(self, index):
        return self.images[index],torch.tensor(0.)

    def __len__(self):
        return len(self.images)

def load_Uniform_Noise(name):
    if name =="Uniform_Noise_test":
        dataset = Uniform_Noise_test()
    return dataset

def register_Uniform_Noise(name):
    DatasetCatalog.register(name, lambda: load_Uniform_Noise(name))