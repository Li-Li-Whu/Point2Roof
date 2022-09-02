import glob
import tqdm
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
from model.pointnet_util import *
from model.model_utils import *

def writePoints(points, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(points)):
            point = points[i]
            file1.write(str(point[0]))
            file1.write(' ')
            file1.write(str(point[1]))
            file1.write(' ')
            file1.write(str(point[2]))
            file1.write('\n')

def writeEdges(edges, clsRoad):
    with open(clsRoad, 'w+') as file1:
        for i in range(len(edges)):
            edge = edges[i]
            file1.write(str(edge[0] + 1))
            file1.write(' ')
            file1.write(str(edge[1] + 1))
            file1.write(' ')
            file1.write('\n')

def assign_targets(points, gvs, radius):
    idx = ball_center_query(radius, points, gvs).type(torch.int64)
    batch_size = gvs.size()[0]
    idx_add = torch.arange(batch_size).to(idx.device).unsqueeze(-1).repeat(1, idx.shape[-1]) * gvs.shape[1]
    gvs = gvs.view(-1, 3)
    idx_add += idx
    target_points = gvs[idx_add.view(-1)].view(batch_size, -1, 3)
    dis = target_points - points
    dis[idx < 0] = 0
    dis /= radius
    label = torch.where(idx >= 0, torch.ones(idx.shape).to(idx.device), torch.zeros(idx.shape).to(idx.device))
    return dis, label

def test_model(model, data_loader, logger):
    dataloader_iter = iter(data_loader)
    with tqdm.trange(0, len(data_loader), desc='test', dynamic_ncols=True) as tbar:
        model.use_edge = True
        statistics = {'tp_pts': 0, 'num_label_pts': 0, 'num_pred_pts': 0, 'pts_bias': np.zeros(3, np.float),
                      'tp_edges': 0, 'num_label_edges': 0, 'num_pred_edges': 0}
        for cur_it in tbar:
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            with torch.no_grad():
                batch = model(batch)
                load_data_to_cpu(batch)
            eval_process(batch, statistics)
        bias = statistics['pts_bias'] / statistics['tp_pts']
        logger.info('pts_recall: %f' % (statistics['tp_pts'] / statistics['num_label_pts']))
        logger.info('pts_precision: %f' % (statistics['tp_pts'] / statistics['num_pred_pts']))
        logger.info('pts_bias: %f, %f, %f' % (bias[0], bias[1], bias[2]))
        logger.info('edge_recall: %f' % (statistics['tp_edges'] / statistics['num_label_edges']))
        logger.info('edge_precision: %f' % (statistics['tp_edges'] / statistics['num_pred_edges']))

def eval_process(batch, statistics):
    batch_size = batch['batch_size']
    pts_pred, pts_refined, pts_label = batch['keypoint'], batch['refined_keypoint'], batch['vectors']
    edge_pred, edge_label = batch['edge_score'], batch['edges']
    mm_pts = batch['minMaxPt']
    id = batch['frame_id']

    idx = 0
    for i in range(batch_size):
        mm_pt = mm_pts[i]
        minPt = mm_pt[0]
        maxPt = mm_pt[1]
        deltaPt = maxPt - minPt

        p_pts = pts_refined[pts_pred[:, 0] == i]
        l_pts = pts_label[i]
        l_pts = l_pts[np.sum(l_pts, -1, keepdims=False) > -2e1]
        vec_a = np.sum(p_pts ** 2, -1)
        vec_b = np.sum(l_pts ** 2, -1)
        dist_matrix = vec_a.reshape(-1, 1) + vec_b.reshape(1, -1) - 2 * np.matmul(p_pts, np.transpose(l_pts))
        dist_matrix = np.sqrt(dist_matrix + 1e-6)
        p_ind, l_ind = linear_sum_assignment(dist_matrix)
        mask = dist_matrix[p_ind, l_ind] < 0.1   # 0.1
        tp_ind, tl_ind = p_ind[mask], l_ind[mask]
        #dis = np.abs(p_pts[tp_ind] - l_pts[tl_ind])
        dis = np.abs( ((p_pts[tp_ind]*deltaPt) + minPt) - ((l_pts[tl_ind]*deltaPt) + minPt) )


        statistics['tp_pts'] += tp_ind.shape[0]
        statistics['num_label_pts'] += l_pts.shape[0]
        statistics['num_pred_pts'] += p_pts.shape[0]
        statistics['pts_bias'] += np.sum(dis, 0)

        match_edge = list(itertools.combinations(l_ind, 2))
        match_edge = np.array([tuple(sorted(e)) for e in match_edge])
        score = edge_pred[idx:idx+len(match_edge)]
        idx += len(match_edge)
        l_edge = edge_label[i]
        l_edge = l_edge[np.sum(l_edge, -1, keepdims=False) > 0]
        l_edge = [tuple(e) for e in l_edge]
        match_edge = match_edge[score > 0.5]
        tp_edges = np.sum([tuple(e) in l_edge for e in match_edge])
        statistics['tp_edges'] += tp_edges
        statistics['num_label_edges'] += len(l_edge)
        statistics['num_pred_edges'] += match_edge.shape[0]


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def load_data_to_cpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        batch_dict[key] = val.cpu().numpy()