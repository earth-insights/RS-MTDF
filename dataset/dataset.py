from PIL import Image
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.transform_modified import resize,  crop
from torchvision.transforms import ToTensor
class LabelDataset(Dataset):
    def __init__(self, root, id_path, size, nsample=None):
        self.root = root
        self.size = size

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()
        
        if nsample is not None:
            self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        id = self.ids[item]
        img_path = id.split(' ')[0]
        mask_path = id.split(' ')[1]

        img = Image.open(img_path).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(mask_path).convert('L')) - 1)

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 255  
        img, mask = crop(img, mask, self.size, ignore_value)
        img = ToTensor()(img) 
        mask = torch.from_numpy(np.array(mask)).long()  
        return img, mask

    def __len__(self):

        return len(self.ids)
