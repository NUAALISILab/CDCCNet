# -*- coding: utf-8 -*-
from typing import Any
from packaging import version

from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F


class STCCLLoss(nn.Module):
    def __init__(self, args):
        super(STCCLLoss, self).__init__()
        self.args = args
        self.ignore_label = 255
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.max_classes = 2
        self.max_views = 10
        self.temperature = 0.07
        self.gamma = 2.0


    def resize_label(self, labels, HW):

        if labels.dim() == 2:
            labels = labels.unsqueeze(0).unsqueeze(0).float().clone()
        elif labels.dim() == 3:
            labels = labels.unsqueeze(1).float().clone()
        elif labels.dim() == 4:
            labels = labels.float().clone()
        else:
            raise ValueError(f"resize_label: unexpected tensor shape {labels.shape}")

        labels = torch.nn.functional.interpolate(labels, HW, mode='nearest')

        if labels.size(1) == 1:
            labels = labels.squeeze(1).long()
        else:
            labels = labels.long()

        return labels

    def _hard_anchor_sampling(self, X_q, X_k, y_hat, y):
        batch_size, feat_dim = X_q.shape[0], X_q.shape[-1]
        n_view = self.max_views

        X_q_valid = []
        X_k_valid = []

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = torch.unique(this_y_hat)

            this_classes = [
                x for x in this_classes
                if x != self.ignore_label
            ]

            this_classes = [
                x for x in this_classes
                if (this_y_hat == x).nonzero().shape[0] > n_view
            ]


            if len(this_classes) < self.max_classes:
                continue

            this_classes = sorted(this_classes, key=lambda x: int(x.item()))
            X_q_img = torch.zeros((self.max_classes, n_view, feat_dim), dtype=X_q.dtype, device=X_q.device)
            X_k_img = torch.zeros((self.max_classes, n_view, feat_dim), dtype=X_k.dtype, device=X_k.device)


            for n, cls_id in enumerate(this_classes):
                if n == self.max_classes:
                    break

                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    X_q_img = None
                    X_k_img = None
                    break

                if num_hard_keep > 0:
                    perm = torch.randperm(num_hard, device=hard_indices.device)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                else:
                    hard_indices = hard_indices[:0]

                if num_easy_keep > 0:
                    perm = torch.randperm(num_easy, device=easy_indices.device)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                else:
                    easy_indices = easy_indices[:0]

                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_q_img[n, :, :] = X_q[ii, indices, :].squeeze(1)

                X_k_img[n, :, :] = X_k[ii, indices, :].squeeze(1)

            if X_q_img is None:
                continue

            X_q_valid.append(X_q_img)
            X_k_valid.append(X_k_img)

        if len(X_q_valid) == 0:
            return None, None, None

        X_q_ = torch.stack(X_q_valid, dim=0)
        X_k_ = torch.stack(X_k_valid, dim=0)

        num_classes = [self.max_classes] * len(X_q_valid)

        return X_q_, X_k_, num_classes

    def _contrastive(self, feats_q_, feats_k_):
        batch_size, num_classes, n_view, patch_dim = feats_q_.shape
        num_patches = batch_size * num_classes * n_view

        feats_q_ = feats_q_.contiguous().view(num_patches, -1, patch_dim)
        feats_k_ = feats_k_.contiguous().view(num_patches, -1, patch_dim)

        l_pos = torch.bmm(feats_q_, feats_k_.transpose(2, 1))
        l_pos = l_pos.view(num_patches, 1)

        feats_q_ = feats_q_.contiguous().view(batch_size, -1, patch_dim)
        feats_k_ = feats_k_.contiguous().view(batch_size, -1, patch_dim)
        n_patches = feats_q_.shape[1]

        l_neg_curbatch = torch.bmm(feats_q_, feats_k_.transpose(2, 1))

        diag_block = torch.zeros((batch_size, n_patches, n_patches), device=feats_q_.device, dtype=torch.uint8)
        for i in range(num_classes):
            diag_block[:, i * n_view:(i + 1) * n_view, i * n_view:(i + 1) * n_view] = 1

        l_neg_curbatch = l_neg_curbatch[~diag_block.to(torch.bool)].view(batch_size, n_patches, -1)

        l_neg = l_neg_curbatch.view(num_patches, -1)

        out = torch.cat([l_pos, l_neg], dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feats_q_.device))

        return loss


    def forward(self, feats_q, feats_k, labels=None, predict=None):

        B, C, H, W = feats_q.shape

        labels = self.resize_label(labels, (H, W))
        predict = self.resize_label(predict, (H, W))

        labels = labels.contiguous().view(B, -1)
        predict = predict.contiguous().view(B, -1)

        feats_q = feats_q.permute(0, 2, 3, 1)
        feats_q = feats_q.contiguous().view(feats_q.shape[0], -1, feats_q.shape[-1])

        feats_k = feats_k.detach()
        feats_k = feats_k.permute(0, 2, 3, 1)
        feats_k = feats_k.contiguous().view(feats_k.shape[0],-1,feats_k.shape[-1])

        feats_q_, feats_k_, num_classes = \
            self._hard_anchor_sampling(feats_q, feats_k, labels, predict)

        if feats_q_ is None:
            return feats_q.sum().reshape(1) * 0.0

        loss = self._contrastive(feats_q_, feats_k_)

        return loss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class ProjectionHead(nn.Module):

    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.proj(x))


