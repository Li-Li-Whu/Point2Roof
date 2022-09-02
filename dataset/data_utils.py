from torch.utils.data import DataLoader
#from .roofn3d_dataset import RoofN3dDataset
from dataset.roofn3d_dataset import RoofN3dDataset
import numpy as np
import random

__all__ = {
    'RoofN3dDataset': RoofN3dDataset
}

class GaussianTransform:
    def __init__(self, sigma = (0.005, 0.015), clip = 0.05, p = 0.8):
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, points):
        if np.random.rand(1) < self.p:
            lastsigma = np.random.rand(1) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
            row, Col = points.shape
            jittered_point = np.clip(lastsigma * np.random.randn(row, Col), -1 * self.clip, self.clip)
            jittered_point += points
            return jittered_point
        else:
            return points


def build_dataloader(path, batch_size, data_cfg, workers=16, logger=None, training=True):
    path += '/train.txt' if training else '/test.txt'

    if training:
        trasform = GaussianTransform(sigma=(0.005, 0.015), clip = 10, p = 0.8)
    else:
        trasform = GaussianTransform(sigma= (0.005, 0.015), clip = 10, p = 0.0)

    dataset = RoofN3dDataset(path, trasform, data_cfg, logger)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, collate_fn=dataset.collate_batch,
        shuffle=training)
    return dataloader
