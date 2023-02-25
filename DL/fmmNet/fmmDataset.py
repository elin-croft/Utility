import os, sys
from typing import Any, Callable, Optional, Tuple
import numpy as np
import tqdm

import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

class fmmDataset(Dataset):
    def __init__(
        self,
        root: str,
        mode: str,
        recurrent: bool = False,
        label_length: int = 1,
        data_size: Optional[Tuple] = None,
        target_size: Optional[Tuple] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super(fmmDataset, self).__init__()
        #assert mode in ('train', 'test', 'val')
        self.data_name = []
        self.data: Any = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        root = os.path.join(root, mode)   
        if not recurrent:
            self.getData(root)
        else:
            paths = self.getRoots(root)
            for path in paths:
                self.getData(path)
                
        self.data = torch.cat(self.data, dim=0).view(-1, *data_size) 

    def getRoots(
        self, 
        root:str
    ) -> list:
        res = []
        for root, dirs, _ in os.walk(root):
            for dir in dirs:
                res.append(os.path.join(root, dir))
        return res

    def getData(self, root) -> None:
        print("loading data from {}".format(root))
        for file in tqdm.tqdm(os.listdir(root)):
            with open(os.path.join(root, file), 'rb') as f:
                entry = pickle.load(f)
                self.data_name.append(entry['name'])
                self.data.append(entry['input_feat'])
                self.targets.append([i.squeeze() for i in entry['target_feat']])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        return a tuple that contains data and label
        """
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> Tuple:
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        elif isinstance(self.data, torch.Tensor):
            return self.data.size()
        else:
            return self.__len__()
