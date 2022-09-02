import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_stack_utils import *
from .model_utils import *
from scipy.optimize import linear_sum_assignment
from utils import loss_utils
import pc_util
import itertools


class EdgeAttentionNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.freeze = False

        self.att_layer = PairedPointAttention(input_channel)
        num_feature = self.att_layer.num_output_feature
        self.shared_fc = LinearBN(num_feature, num_feature)
        self.drop = nn.Dropout(0.5)
        self.cls_fc = nn.Linear(num_feature, 1)
        if self.training:
            self.train_dict = {}
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidBCELoss()
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

    def forward(self, batch_dict):
        batch_idx = batch_dict['keypoint'][:, 0]
        point_fea = batch_dict['keypoint_features']

        if self.training:
            matches = batch_dict['matches']
            edge_label = batch_dict['edges']
            bin_label_list = []
            for i, edge in enumerate(edge_label):
                mask = batch_idx == i
                tmp_idx = batch_idx[mask]
                if tmp_idx.shape[0] <= 1:
                    continue
                match = matches[mask]
                match_edge = list(itertools.combinations(match.cpu().numpy(), 2))
                match_edge = [tuple(sorted(e)) for e in match_edge]
                edge = [tuple(e) for e in edge.cpu().numpy()]
                label = edge_label.new_tensor([e in edge for e in match_edge])
                bin_label_list.append(label)
            self.train_dict['label'] = torch.cat(bin_label_list)

        idx = 0
        pair_idx_list = []
        pair_idx_list1, pair_idx_list2 = [], []
        for i in range(batch_dict['batch_size']):
            mask = batch_idx == i
            tmp_idx = batch_idx[mask]
            if tmp_idx.shape[0] <= 1:
                continue
            fea = point_fea[mask]
            pair_idx = itertools.combinations(range(fea.shape[0]), 2)
            pair_idx = point_fea.new_tensor(list(pair_idx))
            pair_idx_list.append(pair_idx)
            pair_idx_list1.append(pair_idx[:, 0] + idx)
            pair_idx_list2.append(pair_idx[:, 1] + idx)
            idx += tmp_idx.shape[0]
        pair_idx1 = torch.cat(pair_idx_list1).long()
        pair_idx2 = torch.cat(pair_idx_list2).long()
        pair_fea1 = point_fea[pair_idx1]
        pair_fea2 = point_fea[pair_idx2]
        edge_fea = self.att_layer(pair_fea1, pair_fea2)
        edge_pred = self.cls_fc(self.drop(self.shared_fc(edge_fea)))
        batch_dict['pair_points'] = torch.cat(pair_idx_list, 0)
        batch_dict['edge_score'] = torch.sigmoid(edge_pred).view(-1)
        if self.training:
            self.train_dict['edge_pred'] = edge_pred
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_cls = self.train_dict['edge_pred']
        label_cls = self.train_dict['label']
        cls_loss = self.get_cls_loss(pred_cls, label_cls, self.loss_weight['cls_weight'])
        loss = cls_loss
        loss_dict.update({
            'edge_cls_loss': cls_loss.item(),
            'edge_loss': loss.item()
        })

        pred_cls = pred_cls.squeeze(-1)
        label_cls = label_cls.squeeze(-1)
        pred_logit = torch.sigmoid(pred_cls)
        pred = torch.where(pred_logit >= 0.5, pred_logit.new_ones(pred_logit.shape),
                           pred_logit.new_zeros(pred_logit.shape))
        acc = torch.sum((pred == label_cls) & (label_cls == 1)).item() / torch.sum(label_cls == 1).item()
        #acc = torch.sum((pred == label_cls)).item() / len(label_cls.view(-1))
        disp_dict.update({'edge_acc': acc})
        return loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        positives = label > 0
        negatives = label == 0
        cls_weights = (negatives * 1.0 + positives * 1.0).float()
        pos_normalizer = positives.sum().float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum()

        cls_loss = cls_loss * weight
        return cls_loss


class PairedPointAttention(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.edge_att1 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.edge_att2 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.fea_fusion_layer = nn.MaxPool1d(2)

        self.num_output_feature = input_channel

    def forward(self, point_fea1, point_fea2):
        fusion_fea = point_fea1 + point_fea2
        att1 = self.edge_att1(fusion_fea)
        att2 = self.edge_att2(fusion_fea)
        att_fea1 = point_fea1 * att1
        att_fea2 = point_fea2 * att2
        fea = torch.cat([att_fea1.unsqueeze(1), att_fea2.unsqueeze(1)], 1)
        fea = self.fea_fusion_layer(fea.permute(0, 2, 1)).squeeze(-1)
        return fea











