import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pc_util
from torch.autograd import Function, Variable


class Conv2ds(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module('conv%d' % (i + 1), Conv2dBN(in_cn, out_cn))


class Conv2dBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


class Conv1ds(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module('conv%d' % (i + 1), Conv1dBN(in_cn, out_cn))


class Conv1dBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.conv = nn.Conv1d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


class Linears(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module('linear%d' % (i + 1), LinearBN(in_cn, out_cn))


class LinearBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.conv = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


def load_params_with_optimizer(net, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint')
    checkpoint = torch.load(filename)
    epoch = checkpoint.get('epoch', -1)
    it = checkpoint.get('it', 0.0)

    net.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        logger.info('==> Loading optimizer parameters from checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    logger.info('==> Done')

    return it, epoch



def load_params(net, filename, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    if logger is not None:
        logger.info('==> Loading parameters from checkpoint')
    checkpoint = torch.load(filename)

    net.load_state_dict(checkpoint['model_state'])
    if logger is not None:
        logger.info('==> Done')




class DBSCANCluster(Function):

    @staticmethod
    def forward(ctx, eps: float, min_pts: int, point: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param eps: float, dbscan eps
        :param min_pts: int, dbscan core point threshold
        :param point: (B, N, 3) xyz coordinates of the points
        :return:
            idx: (B, N) cluster idx
        """
        point = point.contiguous()

        B, N, _ = point.size()
        idx = torch.cuda.IntTensor(B, N).zero_() - 1

        pc_util.dbscan_wrapper(B, N, eps, min_pts, point, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_out):
        return ()


dbscan_cluster = DBSCANCluster.apply


class GetClusterPts(Function):

    @staticmethod
    def forward(ctx, point: torch.Tensor, cluster_idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param point: (B, N, 3) xyz coordinates of the points
        :param cluster_idx: (B, N) cluster idx
        :return:
            key_pts: (B, M, 3) cluster center pts, M is max_num_cluster_class
            num_cluster: (B, M) cluster num, num of pts in each cluster class
        """
        cluster_idx = cluster_idx.contiguous()

        B, N = cluster_idx.size()
        M = torch.max(cluster_idx) +1
        key_pts = torch.cuda.FloatTensor(B, M, 3).zero_()
        num_cluster = torch.cuda.IntTensor(B, M).zero_()
        pc_util.cluster_pts_wrapper(B, N, M, point, cluster_idx, key_pts, num_cluster)
        key_pts[key_pts * 1e4 == 0] = -1e1
        ctx.mark_non_differentiable(key_pts)
        ctx.mark_non_differentiable(num_cluster)
        return key_pts, num_cluster

    @staticmethod
    def backward(ctx, grad_out):
        return ()


get_cluster_pts = GetClusterPts.apply