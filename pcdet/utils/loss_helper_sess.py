""" Loss Function for Self-Ensembling Semi-Supervised 3D Object Detection

Author: Zhao Na, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from pcdet.utils import common_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance


def compute_center_consistency_loss(end_points, ema_end_points):
    center = end_points['gt_boxes'][:, :, 0:3]  # (B, num_proposal, 3)
    ema_center = ema_end_points['gt_boxes'][:, :, 0:3]  # (B, num_proposal, 3)
    flip_x = end_points['flip_x']  # (B,)
    flip_y = end_points['flip_y']  # (B,)
    rot_angle = end_points['noise_rotation']
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)
    rot_angle = np.array([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]])  # (B,3,3)
    scale_ratio = end_points['noise_scale']
    scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)  # (B,1,3)
    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]
    ema_center = common_utils.rotate_points_along_z(ema_center, rot_angle)[0]
    ema_center = torch.bmm(ema_center, rot_angle.transpose(1, 2))  # (B, num_proposal, 3)

    ema_center = ema_center * scale_ratio

    dist1, ind1, dist2, ind2 = nn_distance(center,
                                           ema_center)  # ind1 (B, num_proposal): ema_center index closest to center

    # TODO: use both dist1 and dist2 or only use dist1
    dist = dist1 + dist2
    return torch.mean(dist), ind2


def compute_class_consistency_loss(end_points, ema_end_points, map_ind):
    cls_scores = end_points['batch_cls_preds']  # (B, num_proposal, num_class)
    ema_cls_scores = ema_end_points['batch_cls_preds']  # (B, num_proposal, num_class)

    cls_log_prob = F.log_softmax(cls_scores, dim=2)  # (B, num_proposal, num_class)
    # cls_log_prob = F.softmax(cls_scores, dim=2)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2)  # (B, num_proposal, num_class)

    cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(cls_log_prob_aligned, ema_cls_prob)

    return class_consistency_loss * 2


def compute_size_consistency_loss(end_points, ema_end_points, map_ind):
    num_size_cluster = 3
    type_mean_size = {
        'Car': np.array([3.9, 1.6, 1.56]),
        'Pedestrian': np.array([0.8, 0.6, 1.73]),
        'Cyclist': np.array([1.76, 0.6, 1.73])
    }
    mean_size_arr = np.zeros((num_size_cluster, 3))
    for i in range(mean_size_arr):
        mean_size_arr[i, :] = type_mean_size[i]
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster,3)
    B, K = map_ind.shape

    scale_ratio = end_points['scale']  # (B,1,3)
    size_class = torch.argmax(end_points['batch_box_preds'], -1)  # B,num_proposal
    size_residual = torch.gather(end_points['loss_rcnn'], 2,
                                 size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))  # B,num_proposal,1,3
    size_residual.squeeze_(2)

    ema_size_class = torch.argmax(ema_end_points['batch_box_preds'], -1)  # B,num_proposal
    ema_size_residual = torch.gather(ema_end_points['loss_rcnn'], 2,
                                     ema_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1,
                                                                                       3))  # B,num_proposal,1,3
    ema_size_residual.squeeze_(2)

    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B, K, 3)
    size = size_base + size_residual

    ema_size_base = torch.index_select(mean_size_arr, 0, ema_size_class.view(-1))
    ema_size_base = ema_size_base.view(B, K, 3)
    ema_size = ema_size_base + ema_size_residual
    ema_size = ema_size * scale_ratio

    size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])

    size_consistency_loss = F.mse_loss(size_aligned, ema_size)

    return size_consistency_loss


def get_consistency_loss(end_points, end_points_ema):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        ema_end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
            }
    Returns:
        consistency_loss: pytorch scalar tensor
        end_points: dict
    """
    center_consistency_loss, map_ind = compute_center_consistency_loss(end_points, end_points_ema)
    class_consistency_loss = compute_class_consistency_loss(end_points, end_points_ema, map_ind)
    size_consistency_loss = compute_size_consistency_loss(end_points, end_points_ema, map_ind)

    consistency_loss = center_consistency_loss + class_consistency_loss + size_consistency_loss

    end_points['center_consistency_loss'] = center_consistency_loss
    end_points['class_consistency_loss'] = class_consistency_loss
    end_points['size_consistency_loss'] = size_consistency_loss
    end_points['consistency_loss'] = consistency_loss

    return consistency_loss, end_points
