# coding=utf-8
from __future__ import absolute_import, division, print_function
from torchvision import transforms
import numpy as np
import torch


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel - mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
    """

    def __init__(self, mean=None, meanfile=None, img_size=None):
        if mean:
            self.mean = mean
        else:
            arr = np.load(meanfile)

            if img_size:
                start = (256 - img_size) // 2
                end = start + img_size

                arr = arr[:, start:end, start:end]

            self.mean = torch.from_numpy(arr.astype('float32') / 255.0)[[2, 1, 0], :, :]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.sub_(m)
        return tensor


def get_transform(img_size):
    transform_source = transforms.Compose([
            transforms.Resize((img_size+32, img_size+32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize(meanfile='./Data/ilsvrc_2012_mean.npy', img_size=img_size)
    ])
    transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            Normalize(meanfile='./Data/ilsvrc_2012_mean.npy', img_size=img_size)
        ])

    return transform_source, transform_test





