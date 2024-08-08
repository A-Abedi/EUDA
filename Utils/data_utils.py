import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def make_dataset(image_list, base_path, labels):
    if labels:
      len_ = len(image_list)
      images = [(base_path + image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(base_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(base_path + val.split()[0], int(val.split()[1])) for val in image_list]
    return images


class ImageList(Dataset):
    def __init__(self, image_list, base_path, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, base_path, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images."))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

