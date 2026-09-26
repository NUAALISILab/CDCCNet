from typing import Any
from packaging import version

from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F

from models.mynn import initialize_weights
import numpy as np
import time


class PCRCLLoss(nn.Module):
    def __init__(self, args):
        super(PCRCLLoss, self).__init__()

        self.args = args
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_patch = 20
        self.num_classes = 2
        self.max_samples = 1000
        self.temperature = 0.07
        self.hardness_temperature = 0.5


    def reshape_map(self, map, shape):

        if map.dim() == 2:
            map = map.unsqueeze(0).unsqueeze(0).float().clone()
        elif map.dim() == 3:
            map = map.unsqueeze(1).float().clone()
        elif map.dim() == 4:
            map = map.float().clone()
        else:
            raise ValueError(f"reshape_map: unexpected tensor shape {map.shape}")

        map = torch.nn.functional.interpolate(map, shape, mode='nearest')

        if map.size(1) == 1:
            map = map.squeeze(1).long()
        else:
            map = map.long()

        return map

    def _contrastive(self, pos_q, pos_k, neg):
        num_patch, _, patch_dim = pos_q.shape

        # l_pos shape : (num_patch, 1)
        l_pos = torch.bmm(pos_q, pos_k.transpose(2, 1))
        l_pos = l_pos.view(num_patch, 1)

        # l_neg shape : (num_patch, negative_size)
        l_neg = torch.bmm(pos_q, neg.transpose(2, 1))
        l_neg = l_neg.view(num_patch, -1)

        out = torch.cat([l_pos, l_neg], dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=pos_q.device))

        return loss

    def _hard_negative_sampling(self, correct_maps, feats_q, feats_k, predicts_j, labels):

        B, HW, C = feats_q.shape

        X_pos_q = []
        X_pos_k = []
        X_neg = []

        hard_pool_ratio = 0.3

        num_hard = self.num_patch // 2
        num_random = self.num_patch - num_hard

        for ii in range(B):

            M = correct_maps[ii]

            indices = (M == 1).nonzero(as_tuple=False).squeeze(1)

            if indices.numel() == 0:
                continue

            classes_labels = torch.unique(labels[ii])
            classes_wrong = torch.unique(predicts_j[ii, indices])

            pos_indices_list = []
            neg_indices_list = []

            for cls_id in classes_wrong:
                cls_indices = ((M == 1) & (predicts_j[ii] == cls_id)).nonzero(as_tuple=False).squeeze(1)

                if cls_indices.numel() == 0:
                    continue

                if not torch.any(classes_labels == cls_id):
                    continue

                neg_cls_indices = (labels[ii] == cls_id).nonzero(as_tuple=False).squeeze(1)

                num_candidates = neg_cls_indices.numel()

                if num_candidates < self.num_patch:
                    continue

                anchor_feats = feats_q[ii, cls_indices, :]  # [M, C]
                candidate_neg_feats = feats_q[ii, neg_cls_indices, :]  # [N, C]

                with torch.no_grad():
                    neg_similarity = torch.matmul(anchor_feats.detach(), candidate_neg_feats.detach().transpose(0, 1))  # [M, N]
                    hard_pool_size = max(num_hard, int(np.ceil(num_candidates * hard_pool_ratio)))
                    hard_pool_size = min(hard_pool_size, num_candidates - num_random)

                    _, hard_positions = torch.topk(neg_similarity, k=hard_pool_size, dim=1, largest=True, sorted=False)
                    num_anchor = cls_indices.size(0)

                    hard_random_score = torch.rand(num_anchor, hard_pool_size, device=feats_q.device)
                    hard_sample_relative = torch.topk(hard_random_score, k=num_hard, dim=1, largest=True, sorted=False).indices
                    sampled_hard_positions = torch.gather(hard_positions, dim=1, index=hard_sample_relative)  # [M, num_hard]
                    non_hard_mask = torch.ones(num_anchor, num_candidates, dtype=torch.bool, device=feats_q.device)
                    non_hard_mask.scatter_(1, hard_positions, False)

                    random_score = torch.rand(num_anchor, num_candidates, device=feats_q.device)
                    random_score = random_score.masked_fill(~non_hard_mask, -1.0)
                    sampled_random_positions = torch.topk(random_score, k=num_random, dim=1, largest=True, sorted=False).indices  # [M, num_random]

                    sampled_positions = torch.cat([sampled_hard_positions, sampled_random_positions], dim=1)

                    sampled_neg_indices = neg_cls_indices[sampled_positions]  # [M, num_patch]

                pos_indices_list.append(cls_indices)
                neg_indices_list.append(sampled_neg_indices)

            if not pos_indices_list:
                continue

            pos_indices = torch.cat(pos_indices_list, dim=0)  # [N_anchor]
            neg_indices = torch.cat(neg_indices_list, dim=0)  # [N_anchor, num_patch]

            X_pos_q.append(feats_q[ii, pos_indices, :].unsqueeze(1))
            X_pos_k.append(feats_k[ii, pos_indices, :].unsqueeze(1))
            X_neg.append(feats_q[ii, neg_indices, :])

        if not X_pos_q:
            return None, None, None

        X_pos_q = torch.cat(X_pos_q, dim=0)
        X_pos_k = torch.cat(X_pos_k, dim=0)
        X_neg = torch.cat(X_neg, dim=0)

        if X_pos_q.shape[0] > B * self.max_samples:
            indices = torch.randperm(X_pos_q.size(0), device=X_pos_q.device)[:B * self.max_samples]

            X_pos_q = X_pos_q[indices]
            X_pos_k = X_pos_k[indices]
            X_neg = X_neg[indices]

        return X_pos_q, X_pos_k, X_neg


    def forward(self, feats_q, feats_k, predicts, predicts_j, labels):
        B, C, H, W = feats_q.shape

        labels = self.reshape_map(labels, (H, W))
        predicts = self.reshape_map(predicts, (H, W))
        predicts_j = self.reshape_map(predicts_j, (H, W))

        correct_maps = torch.ones_like(predicts, device=feats_q[0].device)
        correct_maps[predicts == predicts_j] = 0
        correct_maps[labels == 255] = 0
        correct_maps[predicts != labels] = 0
        correct_maps = correct_maps.flatten(1, 2)

        predicts_j = predicts_j.flatten(1, 2)
        labels = labels.flatten(1, 2)

        feats_k = feats_k.detach()

        feats_q_reshape = feats_q.permute(0, 2, 3, 1).flatten(1, 2)
        feats_k_reshape = feats_k.permute(0, 2, 3, 1).flatten(1, 2)

        patches_q, patches_k, patches_neg = self._hard_negative_sampling(correct_maps, feats_q_reshape, feats_k_reshape,
                                                                     predicts_j, labels)

        if patches_q is None:
            loss = torch.FloatTensor([0]).cuda()
            return loss

        loss = self._contrastive(patches_q, patches_k, patches_neg)

        return loss

