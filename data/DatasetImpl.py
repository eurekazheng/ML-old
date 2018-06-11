import os
import numpy as np
import csv
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DiaretDataset(Dataset):

    def __init__(self, img_root, label_root, transforms=None, mode='train'):
        with open(os.path.join(label_root, mode + '.csv')) as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            imgs = [(os.path.join(img_root, mode, line[0] + '.jpeg'),
                     int(line[1])) for line in data]
        imgs = sorted(imgs, key=lambda x: x[0].split('_')[-2])
        self.mode = mode
        self.imgs = imgs
        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor()
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
