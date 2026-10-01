import torch
import torch.nn.functional as F


# reference : https://github.com/shachoi/RobustNet
def compute_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

    return f_cor, B

def compute_cross_covariance_matrix(ori_map, aug_map, eye=None):
    eps = 1e-5
    assert ori_map.shape == aug_map.shape

    batch_size, channels, height, width = ori_map.shape
    spatial_size = height * width

    if eye is None:
        eye = torch.eye(channels).cuda()

    feature1 = ori_map.contiguous().view(batch_size, channels, spatial_size)
    feature2 = aug_map.contiguous().view(batch_size, channels, spatial_size)

    cross_covariance = torch.bmm(feature1, feature2.transpose(1, 2))
    cross_covariance = cross_covariance.div(spatial_size - 1)
    cross_covariance = cross_covariance + eps * eye

    return cross_covariance, batch_size


def cross_covariance_diagonal_loss(ori_feat, aug_feat):
    assert ori_feat.shape == aug_feat.shape

    cross_covariance, batch_size = compute_cross_covariance_matrix(ori_feat, aug_feat)
    loss = torch.FloatTensor([0]).cuda()

    for covariance in cross_covariance:
        diagonal = torch.diagonal(covariance.squeeze(dim=0), 0)
        target = torch.ones_like(diagonal).cuda()
        loss = loss + F.mse_loss(diagonal, target)

    return loss / batch_size