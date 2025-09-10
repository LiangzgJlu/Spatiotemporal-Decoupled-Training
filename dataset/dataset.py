import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

FEATURE_RANGE = [[100, 0], [30, 0], [30, -30], [30, 0]]

class CarFollowingTrainTSIDataset(Dataset):
    def __init__(self, opt):
        super(CarFollowingTrainTSIDataset, self).__init__()
        self.path = opt.dataset_path
        self.hwl = opt.history_windows_length
        
        self.noise = opt.noise
        pool = np.load(self.path, allow_pickle=True)
        self.data = []
        self.target = []
        for v in pool[0]:            
            if opt.linear_normal:
                # Feature Linear Normalization
                v[:, 0] = (v[:, 0] - FEATURE_RANGE[0][1]) / (FEATURE_RANGE[0][0] - FEATURE_RANGE[0][1])
                v[:, 1] = (v[:, 1] - FEATURE_RANGE[1][1]) / (FEATURE_RANGE[1][0] - FEATURE_RANGE[1][1])
                v[:, 2] = (v[:, 2] - FEATURE_RANGE[2][1]) / (FEATURE_RANGE[2][0] - FEATURE_RANGE[2][1])
                v[:, 3] = (v[:, 3] - FEATURE_RANGE[3][1]) / (FEATURE_RANGE[3][0] - FEATURE_RANGE[3][1])
                # v[:, 2] = v[:, 3] - v[:, 1]
            self.data.append(v)
        for v in pool[1]:
            self.target.append([v])
        
        logger.info(f"input count: {len(self.data)}, target: {len(self.target)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        d = self.data[index]
        t = self.target[index]
        d, t = torch.FloatTensor(d), torch.FloatTensor(t)
        if self.noise > 1e-5:
            d = d + torch.normal(0, self.noise, size=d.shape)
            # d[:, [0, 1, 3]] = d[:, [0, 1, 3]] + torch.normal(0, self.noise, size=d[:, [0, 1, 3]].shape)
            # d[:, 2] = d[:, 3] - d[:, 1]

        return d, t
class CarFollowingTrainNTSIDataset(Dataset):
    def __init__(self, opt):
        super(CarFollowingTrainNTSIDataset, self).__init__()
        self.path = opt.dataset_path
        self.hwl = opt.history_windows_length
        self.noise = opt.noise
        dataset = np.load(self.path, allow_pickle=True)
        self.data = []
  
        for v in dataset:            
            if opt.linear_normal:
                # Feature Linear Normalization
                v[:, 0] = (v[:, 0] - FEATURE_RANGE[0][1]) / (FEATURE_RANGE[0][0] - FEATURE_RANGE[0][1])
                v[:, 1] = (v[:, 1] - FEATURE_RANGE[1][1]) / (FEATURE_RANGE[1][0] - FEATURE_RANGE[1][1])
                v[:, 2] = (v[:, 2] - FEATURE_RANGE[2][1]) / (FEATURE_RANGE[2][0] - FEATURE_RANGE[2][1])
                v[:, 3] = (v[:, 3] - FEATURE_RANGE[3][1]) / (FEATURE_RANGE[3][0] - FEATURE_RANGE[3][1])
                    
            self.data.append(v)
                
        logger.info(f"input count: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        d = self.data[index]
        return d



    


def create_dataset(opt) -> Dataset:
    if opt.tsi:
        return CarFollowingTrainTSIDataset(opt)
    else:
        return CarFollowingTrainNTSIDataset(opt)
    