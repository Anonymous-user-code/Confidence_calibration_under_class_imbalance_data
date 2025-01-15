from .catalog import DatasetCatalog
from torch.utils.data import DataLoader,Sampler
import numpy as np

def get_dataset_dicts(name):
    assert isinstance(name, str)
    dataset = DatasetCatalog.get(name) 
    return dataset

class UnderSampler(Sampler):
    def __init__(self, labels):
        """
        labels: 标签数组，包含所有样本的标签
        batch_size: 每个批次的大小
        is_shuffle: 是否在每个epoch后打乱样本
        """
        
        # 计算正负样本的数量
        labels = np.array(labels)
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        
        self.beta = len(self.pos_indices)/len(self.neg_indices)

        # 确保正负样本数量相等
        min_size = min(len(self.pos_indices), len(self.neg_indices))
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        
        self.pos_indices = self.pos_indices[:min_size]
        self.neg_indices = self.neg_indices[:min_size]
        
    def __iter__(self):
        batch = []
        for i in range(len(self.pos_indices)):
            batch.append(self.pos_indices[i])
            batch.append(self.neg_indices[i])
        return iter(batch)
        
    def __len__(self):
        # 返回批次数量
        return len(self.pos_indices)*2

def build_loader(cfg,dataset):
    return DataLoader(dataset,batch_size=cfg.SOLVER.BATCH_SIZE,shuffle=cfg.SOLVER.SHUFFLE,num_workers=cfg.SOLVER.NUM_WORKS,drop_last=False)

def build_sampling_loader(cfg,dataset):
    if cfg.DATASET.SAMPLER == "undersampler":
        labels = [dataset[idx][1] for idx in range(len(dataset))]
        Sampler = UnderSampler(labels)
        cfg.SOLVER.SHUFFLE = False
    else:
        Sampler = None
    return DataLoader(dataset,batch_size=cfg.SOLVER.BATCH_SIZE,shuffle=cfg.SOLVER.SHUFFLE,num_workers=cfg.SOLVER.NUM_WORKS,drop_last=False,sampler=Sampler,pin_memory=cfg.SOLVER.PIN_MEMORY)
