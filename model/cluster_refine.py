import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_stack_utils import *
from .model_utils import *
from scipy.optimize import linear_sum_assignment
from utils import loss_utils
import pc_util


class ClusterRefineNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.matcher = HungarianMatcher(self.model_cfg.MatchRadius)
        sa_cfg = model_cfg.RefineSA
        mlps = sa_cfg.MLPs
        mlps = [[input_channel] + mlp for mlp in mlps]
        self.fea_refine_module = StackSAModuleMSG(
                radii=sa_cfg.Radii,
                nsamples=sa_cfg.Nsamples,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
        self.num_output_feature = sum([mlp[-1]for mlp in mlps])
        self.shared_fc = LinearBN(256, 128)
        self.drop = nn.Dropout(0.5)
        self.offset_fc = nn.Linear(128, 3)
        # self.cls_fc = nn.Linear(128, 1)
        if self.training:
            self.train_dict = {}
            # self.add_module(
            #     'cls_loss_func',
            #     loss_utils.SigmoidBCELoss()
            # )
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss()
            )
            self.loss_weight = self.model_cfg.LossWeight


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


# tips: change from batch to stack
    def forward(self, batch_dict):
        offset_pts = batch_dict['points'].clone()
        offset = batch_dict['point_pred_offset']
        pts_score = batch_dict['point_pred_score']
        score_thresh = self.model_cfg.ScoreThresh
        offset_pts[pts_score > score_thresh] += offset[pts_score > score_thresh]
        pts_cluster = offset_pts.new_ones(offset_pts.shape) * -10
        pts_cluster[pts_score > score_thresh] = offset_pts[pts_score > score_thresh]
        cluster_idx = dbscan_cluster(self.model_cfg.Cluster.eps, self.model_cfg.Cluster.min_pts, pts_cluster)
        key_pts, num_cluster = get_cluster_pts(pts_cluster, cluster_idx)
        if self.training:
             new_pts, targets, labels, matches, new_xyz_batch_cnt = self.matcher(key_pts, batch_dict['vectors'])
             offset_targets = (targets - new_pts) / self.model_cfg.MatchRadius if new_pts is not None else None
             batch_dict['matches'] = matches
             self.train_dict.update({
                 'keypoint_cls_label': labels,
                 'keypoint_offset_label': offset_targets
             })
        else:
            pts_list, new_xyz_batch_cnt = [], []
            for i, pts in enumerate(key_pts):
                pts = pts[torch.sum(pts, -1) > -2e1]
                if len(pts) == 0:
                    new_xyz_batch_cnt.append(0)
                    continue
                new_xyz_batch_cnt.append(len(pts))
                pts_list.append(pts)
            if sum(new_xyz_batch_cnt) == 0:
                new_pts, new_xyz_batch_cnt = None, None
            else:
                new_pts = torch.cat(pts_list, 0)
                new_xyz_batch_cnt = new_pts.new_tensor(new_xyz_batch_cnt, dtype=torch.int32)
        if new_pts is None:
            exit()
        batch_idx = torch.zeros(new_pts.shape[0], device=new_pts.device)
        idx = 0
        for i, cnt in enumerate(new_xyz_batch_cnt):
            if cnt == 0:
                continue
            batch_idx[idx:idx + cnt] = i
            idx += cnt

        pos_mask = new_xyz_batch_cnt > 0
        offset_pts = offset_pts[pos_mask]
        xyz = offset_pts.view(-1, 3)
        xyz_batch_cnt = offset_pts.new_ones(offset_pts.shape[0], dtype=torch.int32) * offset_pts.shape[1]
        new_xyz_batch_cnt = new_xyz_batch_cnt[pos_mask]
        point_fea = batch_dict['point_features']
        point_fea = point_fea * pts_score.detach().unsqueeze(-1)
        point_fea = point_fea[pos_mask]
        point_fea = point_fea.contiguous().view(-1, point_fea.shape[-1])
        _, refine_fea = self.fea_refine_module(xyz, xyz_batch_cnt, new_pts, new_xyz_batch_cnt, point_fea)

        x = self.drop(self.shared_fc(refine_fea))
        pred_offset = self.offset_fc(x)
        # pred_cls = self.cls_fc(x)
        if self.training:
            self.train_dict.update({
                # 'keypoint_cls_pred': pred_cls,
                'keypoint_offset_pred': pred_offset
            })
        batch_dict['keypoint'] = torch.cat([batch_idx.view(-1, 1), new_pts], -1)
        batch_dict['keypoint_features'] = refine_fea
        # batch_dict['keypoint_pred_score'] = torch.sigmoid(pred_cls).squeeze(-1)
        batch_dict['refined_keypoint'] = pred_offset * self.model_cfg.MatchRadius + new_pts
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        # pred_cls, pred_offset = self.train_dict['keypoint_cls_pred'], self.train_dict['keypoint_offset_pred']
        pred_offset = self.train_dict['keypoint_offset_pred']
        label_cls, label_offset = self.train_dict['keypoint_cls_label'], self.train_dict['keypoint_offset_label']
        # cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        reg_loss = self.get_reg_loss(pred_offset, label_offset, label_cls, self.loss_weight['reg_weight'])
        loss = reg_loss
        # loss = cls_loss + reg_loss
        loss_dict.update({
            # 'refine_cls_loss': cls_loss.item(),
            'refine_offset_loss': reg_loss.item(),
            'refine_loss': loss.item()
        })

        # pred_cls = pred_cls.squeeze(-1)
        # label_cls = label_cls.squeeze(-1)
        # pred_logit = torch.sigmoid(pred_cls)
        # pred = torch.where(pred_logit >= 0.5, pred_logit.new_ones(pred_logit.shape),
        #                    pred_logit.new_zeros(pred_logit.shape))
        # acc = torch.sum((pred == label_cls) & (label_cls == 1)).item() / torch.sum(label_cls == 1).item()
        # disp_dict.update({'pos_acc': acc})
        return loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        batch_size = int(pred.shape[0])
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * weight
        return cls_loss

    def get_reg_loss(self, pred, label, cls_label, weight):
        positives = cls_label > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_loss_src = self.reg_loss_func(pred.unsqueeze(dim=0), label.unsqueeze(dim=0), weights=reg_weights.unsqueeze(dim=0))
        reg_loss = reg_loss_src.sum()
        reg_loss = reg_loss * weight
        return reg_loss
        

class StackSAModuleMSG(nn.Module):

    def __init__(self, radii, nsamples, mlps, use_xyz, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class HungarianMatcher(nn.Module):
    def __init__(self, match_r):
        super().__init__()
        self.dist_thresh = match_r

    # tips: matcher with dist threshold
    @torch.no_grad()
    def forward(self, output, targets):
        pts_list, target_list, label_list, match_list, new_xyz_batch_cnt = [], [], [], [], []
        for i in range(output.shape[0]):
            tmp_output, tmp_targets = output[i], targets[i]
            tmp_output = tmp_output[torch.sum(tmp_output, -1) > -2e1]
            if len(tmp_output) == 0:
                new_xyz_batch_cnt.append(0)
                continue
            tmp_targets = tmp_targets[torch.sum(tmp_targets, -1) > -2e1]
            vec_a = torch.sum(tmp_output.unsqueeze(1).repeat(1, tmp_targets.shape[0], 1) ** 2, -1)
            vec_b = torch.sum(tmp_targets.unsqueeze(0).repeat(tmp_output.shape[0], 1, 1) ** 2, -1)
            dist_matrix = vec_a + vec_b - 2 * torch.mm(tmp_output, tmp_targets.permute(1, 0))
            dist_matrix = F.relu(dist_matrix)
            dist_matrix = torch.sqrt(dist_matrix)

            out_ind, tar_ind = linear_sum_assignment(dist_matrix.cpu().numpy())
            out_ind, tar_ind = dist_matrix.new_tensor(out_ind, dtype=torch.int64), dist_matrix.new_tensor(tar_ind, dtype=torch.int64)
            dist_val = dist_matrix[out_ind, tar_ind]
            out_ind = out_ind[dist_val < self.dist_thresh]
            tar_ind = tar_ind[dist_val < self.dist_thresh]

            pts_list.append(tmp_output)
            tmp_label = tmp_targets.new_zeros(tmp_output.shape[0])
            tmp_label[out_ind] = 1.
            tmp_pts_target = tmp_targets.new_zeros(tmp_output.shape)
            tmp_pts_target[out_ind] = tmp_targets[tar_ind]
            tmp_match = tmp_targets.new_ones(tmp_output.shape[0], dtype=torch.int64) * -1
            tmp_match[out_ind] = tar_ind
            label_list.append(tmp_label)
            target_list.append(tmp_pts_target)
            match_list.append(tmp_match)
            new_xyz_batch_cnt.append(tmp_output.shape[0])
        if sum(new_xyz_batch_cnt) == 0:
            return None, None, None, None, None
        return torch.cat(pts_list, 0), torch.cat(target_list, 0), torch.cat(label_list, 0), torch.cat(match_list, 0), tmp_output.new_tensor(new_xyz_batch_cnt, dtype=torch.int32)







