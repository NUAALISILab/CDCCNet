# -*- coding: utf-8 -*-
import torch
from torch import nn


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

    def _collect_valid_classes(self, y_label, n_view):
        classes = torch.unique(y_label)
        classes = [cls_id for cls_id in classes if cls_id != self.ignore_label]
        classes = [cls_id for cls_id in classes if (y_label == cls_id).nonzero().shape[0] > n_view]
        classes = sorted(classes, key=lambda cls_id: int(cls_id.item()))
        return classes

    def _sample_existing_class(self, y_label, y_prediction_ori, cls_id, n_view):
        hard_indices = ((y_label == cls_id) & (y_prediction_ori != cls_id)).nonzero()
        easy_indices = ((y_label == cls_id) & (y_prediction_ori == cls_id)).nonzero()

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
            return None

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

        return torch.cat((hard_indices, easy_indices), dim=0)

    def _fill_all_slots_randomly(self, X_aug, X_ori, image_index, n_view):
        indices = torch.arange(X_aug.shape[1], device=X_aug.device)
        perm = torch.randperm(X_aug.shape[1], device=X_aug.device)
        indices = indices[perm[:n_view * self.max_classes]]
        indices = indices.view(self.max_classes, n_view)

        X_aug_img = torch.zeros((self.max_classes, n_view, X_aug.shape[-1]), dtype=X_aug.dtype, device=X_aug.device)
        X_ori_img = torch.zeros((self.max_classes, n_view, X_ori.shape[-1]), dtype=X_ori.dtype, device=X_ori.device)
        X_aug_img[:, :, :] = X_aug[image_index, indices, :]
        X_ori_img[:, :, :] = X_ori[image_index, indices, :]
        return X_aug_img, X_ori_img

    def _fill_missing_slots(self, X_aug_img, X_ori_img, X_aug, X_ori, image_index, used_indices, num_present_classes, n_view):
        used_indices = torch.stack(used_indices).flatten(0, 1)
        num_remain = self.max_classes - num_present_classes

        all_indices = torch.arange(X_aug.shape[1], device=X_aug.device)
        left_mask = torch.zeros(X_aug.shape[1], device=X_aug.device, dtype=torch.bool)
        left_mask[used_indices] = True
        left_indices = all_indices[~left_mask]

        perm = torch.randperm(len(left_indices), device=X_aug.device)
        indices = left_indices[perm[:n_view * num_remain]]
        indices = indices.view(num_remain, n_view)

        X_aug_img[num_present_classes:, :, :] = X_aug[image_index, indices, :]
        X_ori_img[num_present_classes:, :, :] = X_ori[image_index, indices, :]

    def _hard_anchor_sampling(self, X_aug, X_ori, y_label, y_prediction_ori):
        batch_size, feat_dim = X_aug.shape[0], X_aug.shape[-1]
        n_view = self.max_views

        X_aug_valid = []
        X_ori_valid = []

        for ii in range(batch_size):
            this_y_label = y_label[ii]
            this_y_prediction_ori = y_prediction_ori[ii]
            this_classes = self._collect_valid_classes(this_y_label, n_view)

            if len(this_classes) == 0:
                X_aug_img, X_ori_img = self._fill_all_slots_randomly(X_aug, X_ori, ii, n_view)
                X_aug_valid.append(X_aug_img)
                X_ori_valid.append(X_ori_img)
                continue

            X_aug_img = torch.zeros((self.max_classes, n_view, feat_dim), dtype=X_aug.dtype, device=X_aug.device)
            X_ori_img = torch.zeros((self.max_classes, n_view, feat_dim), dtype=X_ori.dtype, device=X_ori.device)
            this_indices = []
            image_is_valid = True

            for n, cls_id in enumerate(this_classes):
                if n == self.max_classes:
                    break

                indices = self._sample_existing_class(this_y_label, this_y_prediction_ori, cls_id, n_view)
                if indices is None:
                    image_is_valid = False
                    break

                X_aug_img[n, :, :] = X_aug[ii, indices, :].squeeze(1)
                X_ori_img[n, :, :] = X_ori[ii, indices, :].squeeze(1)
                this_indices.append(indices)

            if not image_is_valid:
                continue

            if len(this_classes) < self.max_classes:
                self._fill_missing_slots(X_aug_img, X_ori_img, X_aug, X_ori, ii, this_indices, len(this_classes), n_view)

            X_aug_valid.append(X_aug_img)
            X_ori_valid.append(X_ori_img)

        if len(X_aug_valid) == 0:
            return None, None, None

        X_aug_ = torch.stack(X_aug_valid, dim=0)
        X_ori_ = torch.stack(X_ori_valid, dim=0)
        num_classes = [self.max_classes] * len(X_aug_valid)
        return X_aug_, X_ori_, num_classes

    def _build_same_slot_mask(self, batch_size, num_classes, n_view, n_patches, device):
        mask = torch.zeros((batch_size, n_patches, n_patches), device=device, dtype=torch.uint8)
        for class_slot in range(num_classes):
            start = class_slot * n_view
            end = (class_slot + 1) * n_view
            mask[:, start:end, start:end] = 1
        return mask

    def _contrastive(self, feats_aug_, feats_ori_):
        batch_size, num_classes, n_view, patch_dim = feats_aug_.shape
        num_patches = batch_size * num_classes * n_view

        feats_aug_ = feats_aug_.contiguous().view(num_patches, -1, patch_dim)
        feats_ori_ = feats_ori_.contiguous().view(num_patches, -1, patch_dim)

        l_pos = torch.bmm(feats_aug_, feats_ori_.transpose(2, 1))
        l_pos = l_pos.view(num_patches, 1)

        feats_aug_ = feats_aug_.contiguous().view(batch_size, -1, patch_dim)
        feats_ori_ = feats_ori_.contiguous().view(batch_size, -1, patch_dim)
        n_patches = feats_aug_.shape[1]

        l_neg_curbatch = torch.bmm(feats_aug_, feats_ori_.transpose(2, 1))
        same_slot_mask = self._build_same_slot_mask(batch_size, num_classes, n_view, n_patches, feats_aug_.device)
        l_neg_curbatch = l_neg_curbatch[~same_slot_mask.to(torch.bool)].view(batch_size, n_patches, -1)
        l_neg = l_neg_curbatch.view(num_patches, -1)

        out = torch.cat([l_pos, l_neg], dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feats_aug_.device))
        return loss

    def _flatten_features(self, feats):
        feats = feats.permute(0, 2, 3, 1)
        return feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

    def forward(self, feats_aug, feats_ori, labels=None, prediction=None):
        B, C, H, W = feats_aug.shape

        labels = self.resize_label(labels, (H, W))
        prediction = self.resize_label(prediction, (H, W))

        labels = labels.contiguous().view(B, -1)
        prediction = prediction.contiguous().view(B, -1)

        feats_aug = self._flatten_features(feats_aug)
        feats_ori = feats_ori.detach()
        feats_ori = self._flatten_features(feats_ori)

        feats_aug_, feats_ori_, num_classes = self._hard_anchor_sampling(feats_aug, feats_ori, labels, prediction)

        if feats_aug_ is None:
            return feats_aug.sum().reshape(1) * 0.0

        loss = self._contrastive(feats_aug_, feats_ori_)
        return loss



