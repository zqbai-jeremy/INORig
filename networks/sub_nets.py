import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import scipy.io as sio
import os
import itertools
import shutil
import matplotlib.pyplot as plt

# External libs
from external.face3d.face3d.morphable_model import MorphabelModel
from external.face3d.face3d.morphable_model.load import load_BFM_info
from external.face3d.face3d.mesh import render
import external.face3d.face3d.mesh as mesh

# Internal libs
from core_dl.base_net import BaseNet
import core_3dv.camera_operator_gpu as cam_opt
from networks.basic_feat_extrator import RGBNet, RGBNetSmall, RGBNetSmallSmall
import data.BFM.utils as bfm_utils


def batched_gradient(features):
    """
    Compute gradient of a batch of feature maps
    :param features: a 3D tensor for a batch of feature maps, dim: (N, C, H, W)
    :return: gradient maps of input features, dim: (N, ２*C, H, W), the last row and column are padded with zeros
             (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy
    """
    H = features.size(-2)
    W = features.size(-1)
    C = features.size(1)
    N = features.size(0)
    grad_x = (features[:, :, :, 2:] - features[:, :, :, :W - 2]) / 2.0
    grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')
    grad_y = (features[:, :, 2:, :] - features[:, :, :H - 2, :]) / 2.0
    grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')
    grad = torch.cat([grad_x.view(N, C, H, W), grad_y.view(N, C, H, W)], dim=1)
    return grad


BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, n_gn=8):
        super(BasicBlockGN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.GroupNorm(n_gn, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.GroupNorm(n_gn, planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class BasicBlockNN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlockNN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.relu = nn.SELU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class RegressionNet(BaseNet):
    """
    Regress the per view params: pose
    """

    def __init__(self):
        super(RegressionNet, self).__init__()

        self.rgb_net = RGBNetSmall(input_dim=(3, 256, 256))
        self.avg_pool = nn.AvgPool2d(4)

        self.pose_fc = nn.Sequential(
            nn.Linear(512, 256, bias=False),        # FC: 512 --> 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 6, bias=True)
        )
        self.pose_fc[-1].weight.data.fill_(0.0)
        self.pose_fc[-1].bias.data.fill_(0.0)

    def train(self, mode=True):
        super(RegressionNet, self).train(mode)
        for module in self.rgb_net.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()

    def forward(self, x):
        """
        :param x: (N, V, C, H, W) images
        :return: pose: (N, V, 6)
        """
        # Extract multi-level features
        N, V, C, H, W = x.shape
        rgb_feats = self.rgb_net(x.view(N * V, C, H, W))                            # dim: [(N * V, c, h, w)]
        feat = self.avg_pool(rgb_feats[0]).view(N * V, 512)

        # Per view pose
        pose = self.pose_fc(feat)                                                   # dim: (N * V, 6)

        return pose.view(N, V, 6)


class MyGridSample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, feat):
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True).detach()
        ctx.save_for_backward(feat, grid)
        return vert_feat

    @staticmethod
    def backward(ctx, grad_output):
        feat, grid = ctx.saved_tensors

        # Gradient for grid
        N, C, H, W = feat.shape
        _, Hg, Wg, _ = grid.shape
        feat_grad = batched_gradient(feat)      # dim: (N, 2*C, H, W)
        grid_grad = F.grid_sample(feat_grad, grid, mode='bilinear', padding_mode='zeros', align_corners=True)       # dim: (N, 2*C, Hg, Wg)
        grid_grad = grid_grad.view(N, 2, C, Hg, Wg).permute(0, 3, 4, 2, 1).contiguous()         # dim: (N, Hg, Wg, C, 2)
        grad_output_perm = grad_output.permute(0, 2, 3, 1).contiguous()                         # dim: (N, Hg, Wg, C)
        grid_grad = torch.bmm(grad_output_perm.view(N * Hg * Wg, 1, C),
                              grid_grad.view(N * Hg * Wg, C, 2)).view(N, Hg, Wg, 2)
        grid_grad[:, :, :, 0] = grid_grad[:, :, :, 0] * (W - 1) / 2
        grid_grad[:, :, :, 1] = grid_grad[:, :, :, 1] * (H - 1) / 2

        # Gradient for feat
        feat_d = feat.detach()
        feat_d.requires_grad = True
        grid_d = grid.detach()
        grid_d.requires_grad = True
        with torch.enable_grad():
            vert_feat = F.grid_sample(feat_d, grid_d, mode='bilinear', padding_mode='zeros', align_corners=True)
            vert_feat.backward(grad_output.detach())
        feat_grad = feat_d.grad

        return grid_grad, feat_grad


class FPNBlock(nn.Module):
    def __init__(self, in_nch, out_nch, n_gn, has_out_conv):
        super(FPNBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv = nn.Conv2d(in_nch, out_nch, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv = BasicBlockGN(out_nch, out_nch, n_gn=n_gn) if has_out_conv else None

    def forward(self, pre_feat, lateral_feat):
        pre_feat = self.up(pre_feat)
        lateral_feat = self.lateral_conv(lateral_feat)
        merge_feat = pre_feat + lateral_feat
        out_feat = self.out_conv(merge_feat) if self.out_conv is not None else None
        return out_feat, merge_feat


class FPN(nn.Module):
    def __init__(self, in_nchs=(128, 64, 32, 16), out_nch=64, n_gn=8, n_out_levels=3):
        super(FPN, self).__init__()
        self.n_out_levels = n_out_levels
        self.coarsest_conv = nn.Conv2d(in_nchs[0], out_nch, kernel_size=1, stride=1, padding=0, bias=False)
        self.n_layers = 0
        for i in range(1, len(in_nchs)):
            if i < len(in_nchs) - n_out_levels:
                has_out_conv = False
            else:
                has_out_conv = True
            in_nch = in_nchs[i]
            setattr(self, 'layer%d' % (i - 1), FPNBlock(in_nch, out_nch, n_gn, has_out_conv))
            self.n_layers += 1

    def forward(self, rgb_feats):
        fpn_feats = []
        pre_feat = self.coarsest_conv(rgb_feats[0])
        for i in range(self.n_layers):
            layer = getattr(self, 'layer%d' % i)
            fpn_feat, pre_feat = layer(pre_feat, rgb_feats[i + 1])
            fpn_feats.append(fpn_feat)
        return fpn_feats


class StepSizeNet(nn.Module):
    def __init__(self, in_nchs=(128, 2), nch=128, out_nchs=(199, 29, 6)):
        super(StepSizeNet, self).__init__()
        self.in_nchs = in_nchs
        self.nch = nch
        self.out_nchs = out_nchs
        in_nch = np.sum(np.asarray(in_nchs))
        out_nch = np.sum(np.asarray(out_nchs))
        self.net = nn.Sequential(
            nn.Linear(in_nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, nch),
            nn.SELU(inplace=True),
            nn.Linear(nch, out_nch),
            nn.Tanh()
        )
        self.net[-2].weight.data.fill_(0.0)
        self.net[-2].bias.data.fill_(0.0)

    def forward(self, inputs, scales, biases):
        """
        Predict step size of gradient descent
        :param inputs: (abs_residual (N, C), ...)
        :param scales: (scale_apply_to_step_size, ...)
        :param biases: (bias_apply_to_step_size, ...)
        :return: (step_size_of_params (N, n_params), ...)
        """
        _in = torch.cat(inputs, dim=1)
        _out = self.net(_in)
        start_i = 0
        outs = []
        for i in range(len(self.out_nchs)):
            out_nch = self.out_nchs[i]
            s = scales[i]
            b = biases[i]
            out = torch.pow(10., s * _out[:, start_i : start_i + out_nch] + b)
            outs.append(out)
            start_i += out_nch
        return outs, _out


class AdaptiveBasisNet(nn.Module):
    def __init__(self, n_para, in_planes, n_planes, size, bfm, bfm_torch, n_gn, basis_init, basis_type):
        super(AdaptiveBasisNet, self).__init__()
        self.n_para = n_para
        self.size = size
        self.bfm = bfm
        self.bfm_torch = bfm_torch
        self.basis_init = basis_init
        self.basis_type = basis_type
        self.pixel_vert_idx = getattr(bfm, 'pixel_vert_idx_' + str(size))
        self.pixel_vert_weights = getattr(bfm_torch, 'pixel_vert_weights_' + str(size))

        if basis_type == 'shape':
            self.uv_coords = getattr(bfm_torch, 'uv_coords_' + str(size))
            self.basis_uv_size = size

            skip = nn.Sequential(
                nn.Conv2d(in_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(n_gn, n_planes)
            )
            self.per_view_net = nn.Sequential(
                BasicBlockGN(in_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(2, 2), residual=True, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(4, 4), residual=True, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(8, 8), residual=True, n_gn=n_gn)
            )
            self.net = nn.Sequential(
                nn.Conv2d(n_planes, n_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(n_gn, n_planes),
                nn.ReLU(True),
                nn.Conv2d(n_planes, n_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(n_gn, n_planes),
                nn.ReLU(True),
                nn.Conv2d(n_planes, 3 * n_para, kernel_size=1, stride=1, padding=0, bias=False)
            )

            if basis_init:
                self.net[-1].weight.data.fill_(0.0)
                self.EV = nn.Parameter(bfm_torch.model['shapeEV'].clone())
                self.Bias = nn.Parameter(bfm_torch.model['shapePC'].clone())
            else:
                self.EV = nn.Parameter(torch.ones((1, n_para, 1)) * 1e3)

        elif basis_type == 'exp':
            self.uv_coords = getattr(bfm_torch, 'uv_coords_' + str(size // 4))
            self.basis_uv_size = size // 4

            skip0 = nn.Sequential(
                nn.Conv2d(in_planes, n_planes, kernel_size=1, stride=2, padding=0, bias=False),
                nn.GroupNorm(n_gn, n_planes)
            )
            skip1 = nn.Sequential(
                nn.Conv2d(n_planes, n_planes, kernel_size=1, stride=2, padding=0, bias=False),
                nn.GroupNorm(n_gn, n_planes)
            )
            self.per_view_net = nn.Sequential(
                BasicBlockGN(in_planes, n_planes, stride=2, dilation=(1, 1), residual=True, downsample=skip0, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=2, dilation=(1, 1), residual=True, downsample=skip1, n_gn=n_gn)
            )
            self.net = nn.Sequential(
                nn.Conv2d(n_planes, n_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(n_gn, n_planes),
                nn.ReLU(True),
                nn.Conv2d(n_planes, n_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(n_gn, n_planes),
                nn.ReLU(True),
                nn.Conv2d(n_planes, 3 * n_para, kernel_size=1, stride=1, padding=0, bias=False)
            )

            if basis_init:
                self.net[-1].weight.data.fill_(0.0)
                self.EV = nn.Parameter(bfm_torch.model['expEV'].clone())
                self.Bias = bfm_torch.model['expPC'].clone()
            else:
                self.EV = nn.Parameter(torch.ones((1, n_para, 1)) * 1e3)

    def cuda(self, device=None):
        super(AdaptiveBasisNet, self).cuda(device)
        self.pixel_vert_weights = self.pixel_vert_weights.cuda(device)
        self.uv_coords = self.uv_coords.cuda(device)
        if self.basis_init:
            self.Bias = self.Bias.cuda(device)

    def denormalize_ap_norm(self, ap_norm):
        ap = ap_norm * self.EV
        return ap

    def sample_per_vert_feat(self, feat, vert, H_img, W_img):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return vert_feat.view(N, V, C, int(self.bfm.nver))

    def forward(self, feat, vert, vis_mask, H_img, W_img):
        """
        Compute adaptive basis based on current reconstruction (i.e. vert)
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param vis_mask: visibility mask. (N, V, 1, nver)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: adap_B: adaptive basis. (N, nver * 3, n_adap_para)
        """
        N, V, C, _, _ = feat.shape
        nver = vert.shape[2]
        vert_feat = self.sample_per_vert_feat(feat, vert, H_img, W_img)             # (N, V, C, nver)
        nonface_region_mask = torch.from_numpy(np.invert(self.bfm.face_region_mask.copy())) \
                              .to(vert.device).view(1, 1, 1, nver)
        vert_feat_masked = vert_feat * vis_mask.float() + vert_feat * nonface_region_mask.float()
        vert_pos = vert.transpose(2, 3).contiguous() / (H_img + W_img) * 2.         # (N, V, 3, nver)
        vert_feat = torch.cat([vert_feat_masked, vert_pos], dim=2).view(N * V, C + 3, nver)

        # Render to UV space
        pixel_vert_feat = vert_feat[:, :, self.pixel_vert_idx]                      # (N * V, C + 3, size, size, 3)
        pixel_vert_weighted_feat = pixel_vert_feat * self.pixel_vert_weights.view(1, 1, self.size, self.size, 3)
        uv_per_view_feat = torch.sum(pixel_vert_weighted_feat, dim=-1)              # (N * V, C + 3, size, size)

        # Conv to adaptive basis
        uv_per_view_feat = self.per_view_net(uv_per_view_feat).view(N, V, C, self.basis_uv_size, self.basis_uv_size)
        (uv_feat, _) = torch.max(uv_per_view_feat, dim=1)                           # (N, C, size, size)
        adap_B_uv = self.net(uv_feat)                                               # (N, 3 * n_adap_para, size, size)
        grid = cam_opt.x_2d_normalize(self.basis_uv_size, self.basis_uv_size, self.uv_coords[:, :, :2].clone()) \
            .view(1, nver, 1, 2).expand(N, nver, 1, 2)
        adap_B = F.grid_sample(adap_B_uv, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (N, 3 * n_adap_para, nver, 1)
        adap_B = adap_B.permute(0, 2, 1, 3).contiguous().view(N, nver * 3, self.n_para)
        if self.basis_init:
            adap_B = adap_B + self.Bias

        # Initialize adaptive displacement
        # adap_bias = self.adap_bias_net(adap_B.permute(0, 2, 1)).view(N, 1, nver, 3)

        return adap_B, adap_B_uv


class ExpressionModel(nn.Module):
    def __init__(self, n_para, model_n_para, in_planes, n_planes, size, bfm, bfm_torch, n_gn, n_level):
        super(ExpressionModel, self).__init__()
        self.n_para = n_para
        self.model_n_para = model_n_para
        self.n_planes = n_planes
        self.size = size
        self.bfm = bfm
        self.bfm_torch = bfm_torch
        self.EV = bfm_torch.model['expEV'][:, :n_planes, :]
        self.Blendshape = bfm_torch.model['expPC'][:, :, :n_planes]
        self.out_size = size
        self.uv_feat = None
        self.uv_vec = None
        self.n_level = n_level

        size = self.size // 8
        for i in range(n_level):
            size *= 2
            pixel_vert_weights = getattr(bfm_torch, 'pixel_vert_weights_' + str(size))
            setattr(self, 'pixel_vert_weights_' + str(size), pixel_vert_weights)
            uv_coords = getattr(bfm_torch, 'uv_coords_' + str(size))
            setattr(self, 'uv_coords_' + str(size), uv_coords)

        # Network to process image & vert feat
        size = self.size // 8
        for i in range(n_level):
            size *= 2
            skip = nn.Sequential(
                nn.Conv2d(in_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(n_gn, n_planes)
            )
            per_view_net = nn.Sequential(
                BasicBlockGN(in_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(2, 2), residual=True, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(4, 4), residual=True, n_gn=n_gn),
                nn.Conv2d(n_planes, 8 * n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True)
            )
            setattr(self, 'per_view_net' + str(i), per_view_net)
            net = nn.Sequential(
                nn.Conv2d(8 * n_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True, n_gn=n_gn)
            )
            setattr(self, 'net' + str(i), net)
            # exp_label_enc = nn.Sequential(
            #     nn.Linear(7, 8 * size * size, bias=False),
            #     nn.SELU(True)
            # )
            # setattr(self, 'exp_label_enc' + str(i), exp_label_enc)
            if i == 0:
                vec_net = nn.Sequential(
                    nn.Conv2d(n_planes, n_planes * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(n_gn, n_planes * 2),
                    nn.ReLU(True),
                    nn.Conv2d(n_planes * 2, n_planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(n_gn, n_planes * 4),
                    nn.ReLU(True),
                    nn.Conv2d(n_planes * 4, n_planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(n_gn, n_planes * 8),
                    nn.ReLU(True),
                    nn.Conv2d(n_planes * 8, n_planes * 16, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                setattr(self, 'vec_net' + str(i), vec_net)

        # Expression network
        size = self.size // 8
        for i in range(n_level):
            size *= 2
            if i == 0:
                mp_norm_mlp = nn.Sequential(
                    nn.Linear(model_n_para + n_planes * 16, 512, bias=False),
                    nn.SELU(True),
                    nn.Linear(512, n_para * n_planes + n_planes * n_planes, bias=False),
                    nn.Tanh()
                )
                setattr(self, 'mp_norm_mlp_level' + str(i), mp_norm_mlp)
            if i > 0:
                mp_norm_enc = nn.Sequential(
                    nn.Linear(model_n_para, n_planes // 4 * size // 4 * size // 4, bias=False),
                    nn.SELU(True)
                )
                setattr(self, 'mp_norm_enc_level' + str(i), mp_norm_enc)
                mp_dec0 = nn.Sequential(
                    BasicBlockNN(n_planes // 4, n_planes // 4, stride=1, dilation=(1, 1), residual=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    BasicBlockNN(n_planes // 4, n_planes // 4, stride=1, dilation=(1, 1), residual=True),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    BasicBlockNN(n_planes // 4, n_planes // 4, stride=1, dilation=(1, 1), residual=True)
                )
                setattr(self, 'mp_dec0_level' + str(i), mp_dec0)
                skip = nn.Conv2d(n_planes // 4 + n_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False)
                mp_dec = nn.Sequential(
                    BasicBlockNN(n_planes // 4 + n_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip),
                    nn.Conv2d(n_planes, n_para * n_planes // 8 + n_planes // 8 * n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Tanh()
                )
                setattr(self, 'mp_dec_level' + str(i), mp_dec)
                out_conv = nn.Sequential(
                    BasicBlockNN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True),
                    nn.Conv2d(n_planes, 3, kernel_size=1, stride=1, padding=0, bias=False)
                )
                out_conv[-1].weight.data.fill_(0.0)
                setattr(self, 'out_conv_level' + str(i), out_conv)
                # ep_norm_enc_mlp = nn.Sequential(
                #     nn.Linear(n_para, n_planes, bias=False),
                #     nn.Tanh(),
                #     nn.Linear(n_planes, n_planes, bias=False),
                #     nn.Tanh()
                # )
                # setattr(self, 'ep_norm_enc_mlp_level' + str(i), ep_norm_enc_mlp)
        self.ep_norm_enc_weights_level0 = None
        self.ep_norm_enc_weights_level1 = None
        self.ep_norm_enc_weights_level2 = None
        self.exp_space_feat_level0 = None
        self.exp_space_feat_level1 = None
        self.exp_space_feat_level2 = None

    def cuda(self, device=None):
        super(ExpressionModel, self).cuda(device)
        self.Blendshape = self.Blendshape.cuda(device)
        self.EV = self.EV.cuda(device)
        # self.pixel_vert_weights = self.pixel_vert_weights.cuda(device)
        # self.uv_coords = self.uv_coords.cuda(device)
        size = self.size // 8
        for i in range(self.n_level):
            size *= 2
            pixel_vert_weights = getattr(self, 'pixel_vert_weights_' + str(size)).cuda(device)
            setattr(self, 'pixel_vert_weights_' + str(size), pixel_vert_weights)
            uv_coords = getattr(self, 'uv_coords_' + str(size)).cuda(device)
            setattr(self, 'uv_coords_' + str(size), uv_coords)

    def init_model(self):
        self.uv_feat = None
        self.uv_vec = None
        self.ep_norm_enc_weights_level0 = None
        self.ep_norm_enc_weights_level1 = None
        self.ep_norm_enc_weights_level2 = None
        self.exp_space_feat_level0 = None
        self.exp_space_feat_level1 = None
        self.exp_space_feat_level2 = None

    def denormalize(self, ep_norm):
        ep = ep_norm * self.EV
        return ep

    def sample_per_vert_feat(self, feat, vert, H_img, W_img):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return vert_feat.view(N, V, C, int(self.bfm.nver))

    def update_uv_feats(self, feats, vert, vis_mask, H_img, W_img, level, exp_label):
        """
        Update feature maps of expression model based on current reconstruction (i.e. vert)
        :param feats: [(N, V, C, H, W)]
        :param vert: (N, V, nver, 3)
        :param vis_mask: visibility mask. (N, V, 1, nver)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: adap_B: adaptive basis. (N, nver * 3, n_adap_para)
                """
        N, V, nver, _ = vert.shape
        nonface_region_mask = torch.from_numpy(np.invert(self.bfm.face_region_mask.copy())) \
                              .to(vert.device).view(1, 1, 1, nver)
        vert_pos = vert.transpose(2, 3).contiguous() / (H_img + W_img) * 2.         # (N, V, 3, nver)
        size = self.size // (2 ** (2 - level))

        i = level
        feat = feats[i]
        assert size == feat.shape[3]
        uv_size = size
        pixel_vert_idx = getattr(self.bfm, 'pixel_vert_idx_' + str(size))
        pixel_vert_weights = getattr(self, 'pixel_vert_weights_' + str(size))
        per_view_net = getattr(self, 'per_view_net' + str(i))
        net = getattr(self, 'net' + str(i))
        # exp_label_enc = getattr(self, 'exp_label_enc' + str(i))

        _, C, H, W = feat.shape
        feat = feat.view(N, V, C, H, W)
        vert_feat = self.sample_per_vert_feat(feat, vert, H_img, W_img)         # (N, V, C, nver)
        vert_feat_masked = vert_feat * vis_mask.float() + vert_feat * nonface_region_mask.float()
        vert_feat = torch.cat([vert_feat_masked, vert_pos], dim=2).view(N * V, C + 3, nver)

        # Render to UV space
        pixel_vert_feat = vert_feat[:, :, pixel_vert_idx]                       # (N * V, C + 3, size, size, 3)
        pixel_vert_weighted_feat = pixel_vert_feat * pixel_vert_weights.view(1, 1, size, size, 3)
        uv_per_view_feat = torch.sum(pixel_vert_weighted_feat, dim=-1)          # (N * V, C + 3, size, size)

        # Conv to feat
        # exp_label_feat = exp_label_enc(exp_label.view(N * V, 7)).view(N * V, 8, size, size)
        # uv_per_view_feat = torch.cat([uv_per_view_feat, exp_label_feat], dim=1)
        uv_per_view_feat = per_view_net(uv_per_view_feat).view(N, V, 8 * self.n_planes, uv_size, uv_size)
        (uv_feat, _) = torch.max(uv_per_view_feat, dim=1)                       # (N, n_planes, uv_size, uv_size)
        uv_feat = net(uv_feat)                                                  # (N, n_planes, uv_size, uv_size)
        self.uv_feat = uv_feat
        if level == 0:
            vec_net = getattr(self, 'vec_net' + str(i))
            self.uv_vec = vec_net(uv_feat)

    def update_exp_space_feat(self, mp_norm, level):
        N = mp_norm.shape[0]
        if level == 0:
            mp_norm_mlp = getattr(self, 'mp_norm_mlp_level' + str(level))
            uv_vec = torch.cat([mp_norm[:, :, 0], self.uv_vec], dim=1)
            ep_norm_enc_weights = mp_norm_mlp(uv_vec) * 6                   # (N, n_para * n_planes + n_planes * n_planes)
            setattr(self, 'ep_norm_enc_weights_level' + str(level), ep_norm_enc_weights)

        if level > 0:
            mp_norm_enc = getattr(self, 'mp_norm_enc_level' + str(level))
            mp_dec0 = getattr(self, 'mp_dec0_level' + str(level))
            mp_dec = getattr(self, 'mp_dec_level' + str(level))
            out_size = self.out_size // (2 ** (2 - level))

            mp_uv_feat = mp_norm_enc(mp_norm[:, :, 0]).view(N, self.n_planes // 4, out_size // 4, out_size // 4)
            mp_uv_feat = mp_dec0(mp_uv_feat)
            mp_uv_feat = torch.cat([mp_uv_feat, self.uv_feat], dim=1)
            mp_uv_feat = mp_dec(mp_uv_feat) * 6
            setattr(self, 'exp_space_feat_level' + str(level), mp_uv_feat)
                    # torch.min(mp_uv_feat, torch.tensor(6.).to(mp_uv_feat.device)))

    def ep_norm_enc(self, ep_norm, weights, V):
        N = weights.shape[0]
        fc_mat = weights[:, :self.n_planes * self.n_para].view(N, 1, self.n_planes, self.n_para) \
                                    .expand(N, V, self.n_planes, self.n_para).contiguous() \
                                    .view(N * V, self.n_planes, self.n_para)
        ep = torch.bmm(fc_mat, ep_norm)
        fc_mat = weights[:, self.n_planes * self.n_para:].view(N, 1, self.n_planes, self.n_planes) \
                                    .expand(N, V, self.n_planes, self.n_planes).contiguous() \
                                    .view(N * V, self.n_planes, self.n_planes)
        ep = torch.bmm(fc_mat, torch.tanh(ep))
        return torch.tanh(ep)

    def forward(self, ep_norm, mp_norm, sp_vert, level, update_exp_space=True):
        N, V, _, _ = ep_norm.shape
        nver = sp_vert.shape[1]
        if update_exp_space:
            self.update_exp_space_feat(mp_norm, level)
        ep_norm = ep_norm.view(N * V, self.n_para, 1)

        # PCA displacement
        weights = self.ep_norm_enc_weights_level0
        if level > 0 or not update_exp_space:
            weights = weights.detach()
        ep = self.ep_norm_enc(ep_norm, weights, V) * 3
        ep = self.denormalize(ep)
        ep_B = self.Blendshape.expand(N * V, nver * 3, self.n_planes)
        ep_vert = torch.bmm(ep_B, ep).view(N, V, nver, 3)

        """-------------------------------------------------------------------------------------------------------------
        for analysis
        """
        self.PCA_disp = ep_vert
        """-------------------------------------------------------------------------------------------------------------
        for analysis end
        """

        # CNN displacement
        out_size = self.out_size // 4
        for i in range(1, level + 1):
            out_size *= 2
            # ep_norm_enc_mlp = getattr(self, 'ep_norm_enc_mlp_level' + str(i))
            out_conv = getattr(self, 'out_conv_level' + str(i))
            exp_space_feat = getattr(self, 'exp_space_feat_level' + str(i))
            if i < level or not update_exp_space:
                exp_space_feat = exp_space_feat.detach()

            # ep_gate = ep_norm_enc_mlp(ep_norm[:, :, 0]).unsqueeze(2).unsqueeze(3)
            ep_gate = ep_norm[:, :, 0].unsqueeze(2).unsqueeze(3).unsqueeze(4)       # (N * V, n_para, 1, 1, 1)
            _, C, H, W = exp_space_feat.shape
            exp_space_feat = exp_space_feat.view(N, 1, C, H, W).expand(N, V, C, H, W).contiguous().view(N * V, C, H, W)
            w = exp_space_feat[:, :self.n_para * self.n_planes // 8, :, :].view(N * V, self.n_para, self.n_planes // 8, H, W)
            ep_gate = torch.tanh((w * ep_gate).sum(dim=1)).unsqueeze(2)
            w = exp_space_feat[:, self.n_para * self.n_planes // 8:, :, :].view(N * V, self.n_planes // 8, self.n_planes, H, W)
            ep_gate = torch.tanh((w * ep_gate).sum(dim=1))
            ep_vert_uv = out_conv(ep_gate) * 1e4

            # Texture mapping to vertex space
            assert ep_vert_uv.shape[2] == out_size and ep_vert_uv.shape[3] == out_size
            uv_coords = getattr(self, 'uv_coords_' + str(out_size))
            grid = cam_opt.x_2d_normalize(out_size, out_size, uv_coords[:, :, :2].clone()) \
                .view(1, nver, 1, 2).expand(N * V, nver, 1, 2)
            grid_sample = MyGridSample.apply
            ep_vert_level = grid_sample(grid, ep_vert_uv)
            ep_vert_level = ep_vert_level.permute(0, 2, 1, 3).contiguous().view(N, V, nver, 3)
            ep_vert = ep_vert + ep_vert_level

            """-------------------------------------------------------------------------------------------------------------
            for analysis
            """
            if i == 1:
                self.cnn_disp = ep_vert_level
            else:
                self.cnn_disp = self.cnn_disp + ep_vert_level
            """-------------------------------------------------------------------------------------------------------------
            for analysis end
            """

        return ep_vert


class IdentityModel(nn.Module):
    def __init__(self, n_para, in_planes, n_planes, size, bfm, bfm_torch, n_gn, n_level):
        super(IdentityModel, self).__init__()
        self.n_para = n_para
        self.n_planes = n_planes
        self.size = size
        self.bfm = bfm
        self.bfm_torch = bfm_torch
        self.EV = bfm_torch.model['shapeEV'][:, :n_para, :].clone()
        self.PCA_basis = bfm_torch.model['shapePC'][:, :, :n_para].clone()
        self.out_size = size
        self.n_level = n_level

        size = self.size // 8
        for i in range(n_level):
            size *= 2
            pixel_vert_weights = getattr(bfm_torch, 'pixel_vert_weights_' + str(size))
            setattr(self, 'pixel_vert_weights_' + str(size), pixel_vert_weights)
            uv_coords = getattr(bfm_torch, 'uv_coords_' + str(size))
            setattr(self, 'uv_coords_' + str(size), uv_coords)

        # Network to process image & vert feat
        for i in range(n_level):
            skip = nn.Sequential(
                nn.Conv2d(in_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(n_gn, n_planes)
            )
            per_view_net = nn.Sequential(
                BasicBlockGN(in_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(2, 2), residual=True, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(4, 4), residual=True, n_gn=n_gn),
                nn.Conv2d(n_planes, 8 * n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True)
            )
            setattr(self, 'per_view_net' + str(i), per_view_net)
            net = nn.Sequential(
                nn.Conv2d(8 * n_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True, n_gn=n_gn)
            )
            setattr(self, 'net' + str(i), net)

        # Identity network
        size = self.size // 8
        for i in range(n_level):
            size *= 2
            sp_norm_enc = nn.Sequential(
                nn.Linear(n_para, n_planes * size // 4 * size // 4, bias=False),
                nn.SELU(True)
            )
            setattr(self, 'sp_norm_enc_level' + str(i), sp_norm_enc)
            sp_dec0 = nn.Sequential(
                BasicBlockNN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                BasicBlockNN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                BasicBlockNN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True)
            )
            setattr(self, 'sp_dec0_level' + str(i), sp_dec0)
            skip = nn.Conv2d(n_planes + n_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False)
            sp_dec1 = nn.Sequential(
                BasicBlockNN(n_planes + n_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip),
                nn.Conv2d(n_planes, 3, kernel_size=1, stride=1, padding=0, bias=False)
            )
            sp_dec1[-1].weight.data.fill_(0.0)
            setattr(self, 'sp_dec1_level' + str(i), sp_dec1)

        self.cached_sp_vert = None
        self.uv_feat = None

    def cuda(self, device=None):
        super(IdentityModel, self).cuda(device)
        self.PCA_basis = self.PCA_basis.cuda(device)
        self.EV = self.EV.cuda(device)
        # self.pixel_vert_weights = self.pixel_vert_weights.cuda(device)
        # self.uv_coords = self.uv_coords.cuda(device)
        size = self.size // 8
        for i in range(self.n_level):
            size *= 2
            pixel_vert_weights = getattr(self, 'pixel_vert_weights_' + str(size)).cuda(device)
            setattr(self, 'pixel_vert_weights_' + str(size), pixel_vert_weights)
            uv_coords = getattr(self, 'uv_coords_' + str(size)).cuda(device)
            setattr(self, 'uv_coords_' + str(size), uv_coords)

    def init_model(self):
        self.uv_feat = None
        self.cached_sp_vert = None

    def denormalize(self, sp_norm):
        sp = sp_norm * self.EV
        return sp

    def sample_per_vert_feat(self, feat, vert, H_img, W_img):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return vert_feat.view(N, V, C, int(self.bfm.nver))

    def update_uv_feats(self, feats, vert, vis_mask, H_img, W_img, level):
        """
        Update feature maps of expression model based on current reconstruction (i.e. vert)
        :param feats: [(N, V, C, H, W)]
        :param vert: (N, V, nver, 3)
        :param vis_mask: visibility mask. (N, V, 1, nver)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: adap_B: adaptive basis. (N, nver * 3, n_adap_para)
                """
        N, V, nver, _ = vert.shape
        nonface_region_mask = torch.from_numpy(np.invert(self.bfm.face_region_mask.copy())) \
                              .to(vert.device).view(1, 1, 1, nver)
        vert_pos = vert.transpose(2, 3).contiguous() / (H_img + W_img) * 2.         # (N, V, 3, nver)
        size = self.size // (2 ** (2 - level))

        i = level
        feat = feats[i]
        assert size == feat.shape[3]
        uv_size = size
        pixel_vert_idx = getattr(self.bfm, 'pixel_vert_idx_' + str(size))
        pixel_vert_weights = getattr(self, 'pixel_vert_weights_' + str(size))
        per_view_net = getattr(self, 'per_view_net' + str(i))
        net = getattr(self, 'net' + str(i))

        _, C, H, W = feat.shape
        feat = feat.view(N, V, C, H, W)
        vert_feat = self.sample_per_vert_feat(feat, vert, H_img, W_img)         # (N, V, C, nver)
        vert_feat_masked = vert_feat * vis_mask.float() + vert_feat * nonface_region_mask.float()
        vert_feat = torch.cat([vert_feat_masked, vert_pos], dim=2).view(N * V, C + 3, nver)

        # Render to UV space
        pixel_vert_feat = vert_feat[:, :, pixel_vert_idx]                       # (N * V, C + 3, size, size, 3)
        pixel_vert_weighted_feat = pixel_vert_feat * pixel_vert_weights.view(1, 1, size, size, 3)
        uv_per_view_feat = torch.sum(pixel_vert_weighted_feat, dim=-1)          # (N * V, C + 3, size, size)

        # Conv to feat
        uv_per_view_feat = per_view_net(uv_per_view_feat).view(N, V, 8 * self.n_planes, uv_size, uv_size)
        (uv_feat, _) = torch.max(uv_per_view_feat, dim=1)                       # (N, n_planes, uv_size, uv_size)
        uv_feat = net(uv_feat)                                                  # (N, n_planes, uv_size, uv_size)
        self.uv_feat = uv_feat

    def forward(self, sp_norm, level):
        N, _, _ = sp_norm.shape
        nver = int(self.bfm.nver)
        if level == 0:
            sp = self.denormalize(sp_norm)
            sp_B = self.PCA_basis.expand(N, nver * 3, self.n_para)
            sp_vert = torch.bmm(sp_B, sp).view(N, nver, 3)
        else:
            sp_vert = 0.

        size = self.out_size // (2 ** (2 - level))
        sp_norm_enc = getattr(self, 'sp_norm_enc_level' + str(level))
        sp_dec0 = getattr(self, 'sp_dec0_level' + str(level))
        sp_dec1 = getattr(self, 'sp_dec1_level' + str(level))
        sp_vert_uv = sp_norm_enc(sp_norm[:, :, 0]).view(N, self.n_planes, size // 4, size // 4)
        sp_vert_uv = sp_dec0(sp_vert_uv)
        sp_vert_uv = torch.cat([sp_vert_uv, self.uv_feat], dim=1)  # (N, n_planes + C, H, W)
        sp_vert_uv = sp_dec1(sp_vert_uv) * 1e5

        # Texture mapping to vertex space
        assert sp_vert_uv.shape[2] == size and sp_vert_uv.shape[3] == size
        uv_coords = getattr(self, 'uv_coords_' + str(size))
        grid = cam_opt.x_2d_normalize(size, size, uv_coords[:, :, :2].clone()).view(1, nver, 1, 2).expand(N, nver, 1, 2)
        # median_ep_vert = F.grid_sample(median_ep_vert_uv, grid, mode='bilinear', padding_mode='zeros',
        #                                align_corners=True)                      # (N * V, 3, nver, 1)
        grid_sample = MyGridSample.apply
        sp_vert_uv = grid_sample(grid, sp_vert_uv)
        sp_vert_uv = sp_vert_uv.permute(0, 2, 1, 3).contiguous().view(N, nver, 3)

        """-------------------------------------------------------------------------------------------------------------
        for analysis
        """
        self.cnn_disp = sp_vert_uv
        """-------------------------------------------------------------------------------------------------------------
        for analysis end
        """

        sp_vert = sp_vert + sp_vert_uv
        if self.cached_sp_vert is not None:
            sp_vert = sp_vert + self.cached_sp_vert.detach()

        return sp_vert


class AlbedoModel(nn.Module):
    def __init__(self, n_para, in_planes, n_planes, size, bfm, bfm_torch, n_gn, n_level):
        super(AlbedoModel, self).__init__()
        self.n_para = n_para
        self.n_planes = n_planes
        self.size = size
        self.bfm = bfm
        self.bfm_torch = bfm_torch
        # self.EV = bfm_torch.model['texEV'][:, :n_para, :].clone()
        # self.PCA_basis = bfm_torch.model['texPC'][:, :, :n_para].clone()
        self.out_size = size
        self.n_level = n_level
        self.exp_n_para = 8

        size = self.size // 8
        for i in range(n_level):
            size *= 2
            pixel_vert_weights = getattr(bfm_torch, 'pixel_vert_weights_' + str(size))
            setattr(self, 'pixel_vert_weights_' + str(size), pixel_vert_weights)
            uv_coords = getattr(bfm_torch, 'uv_coords_' + str(size))
            setattr(self, 'uv_coords_' + str(size), uv_coords)

        # Network to process image & vert feat
        for i in range(n_level):
            skip = nn.Sequential(
                nn.Conv2d(in_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(n_gn, n_planes)
            )
            per_view_net = nn.Sequential(
                BasicBlockGN(in_planes, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(2, 2), residual=True, n_gn=n_gn),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(4, 4), residual=True, n_gn=n_gn),
                nn.Conv2d(n_planes, 8 * n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True)
            )
            setattr(self, 'per_view_net' + str(i), per_view_net)
            net = nn.Sequential(
                nn.Conv2d(8 * n_planes, n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True),
                BasicBlockGN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True, n_gn=n_gn)
            )
            setattr(self, 'net' + str(i), net)
        # self.vec_net = nn.Sequential(
        #     nn.Conv2d(n_planes, n_planes * 2, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.GroupNorm(n_gn, n_planes * 2),
        #     nn.ReLU(True),
        #     nn.Conv2d(n_planes * 2, n_planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.GroupNorm(n_gn, n_planes * 4),
        #     nn.ReLU(True),
        #     nn.Conv2d(n_planes * 4, n_planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.GroupNorm(n_gn, n_planes * 8),
        #     nn.ReLU(True),
        #     nn.Conv2d(n_planes * 8, n_planes * 16, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten()
        # )

        # Identity network
        for i in range(n_level):
            if i == 0:
                size = self.size // 2
            else:
                size = self.size
            ap_norm_enc = nn.Sequential(
                nn.Linear(n_para, n_planes // 4 * size // 4 * size // 4, bias=False),
                nn.SELU(True)
            )
            setattr(self, 'ap_norm_enc_level' + str(i), ap_norm_enc)
            ap_dec0 = nn.Sequential(
                BasicBlockNN(n_planes // 4, n_planes // 4, stride=1, dilation=(1, 1), residual=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                BasicBlockNN(n_planes // 4, n_planes // 4, stride=1, dilation=(1, 1), residual=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                BasicBlockNN(n_planes // 4, n_planes // 4, stride=1, dilation=(1, 1), residual=True)
            )
            setattr(self, 'ap_dec0_level' + str(i), ap_dec0)
            if i < 2:
                skip = nn.Conv2d(n_planes + n_planes // 4, n_planes, kernel_size=1, stride=1, padding=0, bias=False)
                ap_dec = nn.Sequential(
                    BasicBlockNN(n_planes + n_planes // 4, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip),
                    nn.Conv2d(n_planes, 3, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Tanh()
                )
                ap_dec[-2].weight.data.fill_(0.0)
                setattr(self, 'ap_dec_level' + str(i), ap_dec)
            else:
                skip = nn.Conv2d(n_planes + n_planes // 4, n_planes, kernel_size=1, stride=1, padding=0, bias=False)
                ap_dec = nn.Sequential(
                    BasicBlockNN(n_planes + n_planes // 4, n_planes, stride=1, dilation=(1, 1), residual=True, downsample=skip),
                    nn.Conv2d(n_planes, self.exp_n_para * n_planes // 4 + n_planes // 4 * n_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Tanh()
                )
                setattr(self, 'ap_dec_level' + str(i), ap_dec)
                out_conv = nn.Sequential(
                    BasicBlockNN(n_planes, n_planes, stride=1, dilation=(1, 1), residual=True),
                    nn.Conv2d(n_planes, 3, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Tanh()
                )
                out_conv[-2].weight.data.fill_(0.0)
                setattr(self, 'out_conv_level' + str(i), out_conv)
        # self.ep_norm_mlp = nn.Sequential(
        #     nn.Linear(self.exp_n_para, n_planes, bias=False),
        #     nn.SELU(True),
        #     nn.Linear(n_planes, n_planes, bias=False),
        #     nn.Tanh()
        # )

        self.cached_albedo = None
        self.uv_feat = None
        self.dyn_albedo_feat = None

    def cuda(self, device=None):
        super(AlbedoModel, self).cuda(device)
        # self.PCA_basis = self.PCA_basis.cuda(device)
        # self.EV = self.EV.cuda(device)
        # self.pixel_vert_weights = self.pixel_vert_weights.cuda(device)
        # self.uv_coords = self.uv_coords.cuda(device)
        size = self.size // 4
        for i in range(1, self.n_level):
            size *= 2
            pixel_vert_weights = getattr(self, 'pixel_vert_weights_' + str(size)).cuda(device)
            setattr(self, 'pixel_vert_weights_' + str(size), pixel_vert_weights)
            uv_coords = getattr(self, 'uv_coords_' + str(size)).cuda(device)
            setattr(self, 'uv_coords_' + str(size), uv_coords)

    def init_model(self):
        self.uv_feat = None
        self.cached_albedo = self.bfm_torch.model['texMU'].view(1, int(self.bfm.nver), 3).permute(0, 2, 1).float() / 255.
        self.dyn_albedo_feat = None

    # def denormalize(self, ap_norm):
    #     ap = ap_norm * self.EV
    #     return ap

    def sample_per_vert_feat(self, feat, vert, H_img, W_img):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, int(self.bfm.nver), 1, 2)
        vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return vert_feat.view(N, V, C, int(self.bfm.nver))

    def update_uv_feats(self, feats, vert, vis_mask, H_img, W_img, level):
        """
        Update feature maps of expression model based on current reconstruction (i.e. vert)
        :param feats: [(N, V, C, H, W)]
        :param vert: (N, V, nver, 3)
        :param vis_mask: visibility mask. (N, V, 1, nver)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: adap_B: adaptive basis. (N, nver * 3, n_adap_para)
                """
        N, V, nver, _ = vert.shape
        nonface_region_mask = torch.from_numpy(np.invert(self.bfm.face_region_mask.copy())) \
                              .to(vert.device).view(1, 1, 1, nver)
        vert_pos = vert.transpose(2, 3).contiguous() / (H_img + W_img) * 2.         # (N, V, 3, nver)
        # size = self.size // (2 ** (2 - level))
        if level == 0:
            size = self.size // 2
        else:
            size = self.size

        i = level
        feat = feats[min(i + 1, self.n_level - 1)]
        assert size == feat.shape[3]
        uv_size = size
        pixel_vert_idx = getattr(self.bfm, 'pixel_vert_idx_' + str(size))
        pixel_vert_weights = getattr(self, 'pixel_vert_weights_' + str(size))
        per_view_net = getattr(self, 'per_view_net' + str(i))
        net = getattr(self, 'net' + str(i))
        # ap_dec = getattr(self, 'ap_dec_level' + str(i))

        _, C, H, W = feat.shape
        feat = feat.view(N, V, C, H, W)
        vert_feat = self.sample_per_vert_feat(feat, vert, H_img, W_img)         # (N, V, C, nver)
        vert_feat_masked = vert_feat * vis_mask.float() + vert_feat * nonface_region_mask.float()
        vert_feat = torch.cat([vert_feat_masked, vert_pos], dim=2).view(N * V, C + 3, nver)

        # Render to UV space
        pixel_vert_feat = vert_feat[:, :, pixel_vert_idx]                       # (N * V, C + 3, size, size, 3)
        pixel_vert_weighted_feat = pixel_vert_feat * pixel_vert_weights.view(1, 1, size, size, 3)
        uv_per_view_feat = torch.sum(pixel_vert_weighted_feat, dim=-1)          # (N * V, C + 3, size, size)

        # Conv to feat
        uv_per_view_feat = per_view_net(uv_per_view_feat).view(N, V, 8 * self.n_planes, uv_size, uv_size)
        (uv_feat, _) = torch.max(uv_per_view_feat, dim=1)                       # (N, n_planes, uv_size, uv_size)
        uv_feat = net(uv_feat)                                                  # (N, n_planes, uv_size, uv_size)
        # self.albedo_basis = ap_dec(uv_feat)
        # if level == 2:
        #     self.uv_vec = self.vec_net(uv_feat)
        self.uv_feat = uv_feat

    def update_exp_space_feat(self, ap_norm, level):
        N = ap_norm.shape[0]
        # ap_norm_mlp = getattr(self, 'ap_norm_mlp_level' + str(level))
        # uv_vec = torch.cat([ap_norm[:, :, 0], self.uv_vec], dim=1)
        # ap_norm_enc_weights = ap_norm_mlp(uv_vec) * 6                   # (N, n_para * n_planes + n_planes * n_planes)
        # return ap_norm_enc_weights
        ap_norm_enc = getattr(self, 'ap_norm_enc_level' + str(level))
        ap_dec0 = getattr(self, 'ap_dec0_level' + str(level))
        ap_dec = getattr(self, 'ap_dec_level' + str(level))
        if level == 0:
            out_size = self.out_size // 2
        else:
            out_size = self.out_size
        ap_uv_feat = ap_norm_enc(ap_norm[:, :, 0]).view(N, self.n_planes // 4, out_size // 4, out_size // 4)
        ap_uv_feat = ap_dec0(ap_uv_feat)
        ap_uv_feat = torch.cat([ap_uv_feat, self.uv_feat], dim=1)
        ap_uv_feat = ap_dec(ap_uv_feat) * 6
        self.dyn_albedo_feat = ap_uv_feat   #torch.min(ap_uv_feat, torch.tensor(6.).to(ap_uv_feat.device))

    def ep_norm_enc(self, ep_norm, weights, V):
        N = weights.shape[0]
        fc_mat = weights[:, :self.n_planes * self.exp_n_para].view(N, 1, self.n_planes, self.exp_n_para) \
                                    .expand(N, V, self.n_planes, self.exp_n_para).contiguous() \
                                    .view(N * V, self.n_planes, self.exp_n_para)
        # b = weights[:, self.n_planes * self.exp_n_para : self.n_planes * (self.exp_n_para + 1)] \
        #     .view(N, 1, self.n_planes, 1).expand(N, V, self.n_planes, 1).contiguous().view(N * V, self.n_planes, 1)
        ep = torch.bmm(fc_mat, torch.tanh(ep_norm)) #+ b
        fc_mat = weights[:, self.n_planes * self.exp_n_para:].view(N, 1, self.n_para, self.n_planes) \
                                    .expand(N, V, self.n_para, self.n_planes).contiguous() \
                                    .view(N * V, self.n_para, self.n_planes)
        ep = torch.bmm(fc_mat, torch.tanh(ep))
        return torch.tanh(ep)

    def forward(self, ap_norm, ep_norm, ep_vert, level, update_space=True):
        # N, _, _ = ap_norm.shape
        N, V, _, _ = ep_norm.shape
        nver = int(self.bfm.nver)
        # if level == 0:
        #     ap = self.denormalize(ap_norm)
        #     ap_B = self.PCA_basis.expand(N, nver * 3, self.n_para)
        #     albedo = torch.bmm(ap_B, ap).view(N, nver, 3).transpose(1, 2).contiguous()
        #     cached_albedo = self.cached_albedo.detach()
        # else:
        if level == 0:
            size = self.out_size // 2
        else:
            size = self.out_size
        if level < 2:
            ap_norm_enc = getattr(self, 'ap_norm_enc_level' + str(level))
            ap_dec0 = getattr(self, 'ap_dec0_level' + str(level))
            ap_dec = getattr(self, 'ap_dec_level' + str(level))
            ap_uv_feat = ap_norm_enc(ap_norm[:, :, 0]).view(N, self.n_planes // 4, size // 4, size // 4)
            ap_uv_feat = ap_dec0(ap_uv_feat)
            ap_uv_feat = torch.cat([ap_uv_feat, self.uv_feat], dim=1)
            albedo = ap_dec(ap_uv_feat)  # (N, 3, size, size)
        else:
            out_conv = getattr(self, 'out_conv_level' + str(level))
            ep_norm = ep_norm.view(N * V, self.exp_n_para)
            if update_space:
                self.update_exp_space_feat(ap_norm, level)
            ep_gate = ep_norm.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (N * V, exp_n_para, 1, 1, 1)
            exp_space_feat = self.dyn_albedo_feat
            _, C, H, W = exp_space_feat.shape
            exp_space_feat = exp_space_feat.view(N, 1, C, H, W).expand(N, V, C, H, W).contiguous().view(N * V, C, H, W)
            w = exp_space_feat[:, :self.exp_n_para * self.n_planes // 4, :, :].view(N * V, self.exp_n_para, self.n_planes // 4, H, W)
            ep_gate = torch.tanh((w * ep_gate).sum(dim=1)).unsqueeze(2)
            w = exp_space_feat[:, self.exp_n_para * self.n_planes // 4:, :, :].view(N * V, self.n_planes // 4, self.n_planes, H, W)
            ep_gate = torch.tanh((w * ep_gate).sum(dim=1))
            albedo = out_conv(ep_gate)

        # Texture mapping to vertex space
        assert albedo.shape[2] == size and albedo.shape[3] == size
        uv_coords = getattr(self, 'uv_coords_' + str(size))
        if level < 2:
            grid = cam_opt.x_2d_normalize(size, size, uv_coords[:, :, :2].clone()).view(1, nver, 1, 2).expand(N, nver, 1, 2)
        else:
            grid = cam_opt.x_2d_normalize(size, size, uv_coords[:, :, :2].clone()).view(1, nver, 1, 2).expand(N * V, nver, 1, 2)
        # median_ep_vert = F.grid_sample(median_ep_vert_uv, grid, mode='bilinear', padding_mode='zeros',
        #                                align_corners=True)                      # (N * V, 3, nver, 1)
        grid_sample = MyGridSample.apply
        albedo = grid_sample(grid, albedo).squeeze(3)                           # (N, 3, nver) or (N * V, 3, nver)
        if level < 2:
            cached_albedo = self.cached_albedo.detach()
        else:
            albedo = albedo.view(N, V, 3, nver)
            cached_albedo = self.cached_albedo.detach().unsqueeze(1)
            # Attention based on ep_vert
            # atten = torch.norm(ep_vert, dim=3).unsqueeze(2)  # (N, V, 1, nver)
            # # print(atten[0, -1, 0, self.bfm.face_region_mask.ravel()].min(),
            # #       atten[0, -1, 0, self.bfm.face_region_mask.ravel()].max())
            # atten = torch.min(torch.max(atten, torch.zeros_like(atten)), torch.ones_like(atten) * 5e3) / 5e3
            # # colors = atten[0, -1, 0, :].view(nver, 1).expand(nver, 3).detach().cpu().numpy()
            # # uv_texture_map = render.render_colors(self.bfm.uv_coords_128, self.bfm.model['tri'], colors, 128, 128, c=3)
            # # print(np.amin(uv_texture_map), np.amax(uv_texture_map))
            # # plt.imshow(uv_texture_map)
            # # plt.show()
            # albedo = albedo * atten
        albedo = torch.where(torch.lt(albedo, 0),
                             cached_albedo * albedo,
                             (1 - cached_albedo) * albedo)
        # if level == 2:
        #     albedo = albedo - albedo.mean(dim=3, keepdim=True)
        albedo = albedo + cached_albedo
        if level == 2:
            albedo = albedo.view(N * V, 3, nver)
        return albedo


class NRMVSOptimization(BaseNet):

    def __init__(self, opt_step_size=1e-5, MM_base_dir='./external/face3d/examples/Data/BFM'):
        super(NRMVSOptimization, self).__init__()
        self.opt_step_size = opt_step_size
        self.training = True

        # Initialize BFM
        # self.bfm = MorphabelModel(os.path.join(MM_base_dir, 'Out/BFM.mat'))
        # params_attr = sio.loadmat(os.path.join(MM_base_dir, '3ddfa/3DDFA_Release/Matlab/params_attr.mat'))
        # self.bfm.params_mean_3dffa = params_attr['params_mean']
        # self.bfm.params_std_3dffa = params_attr['params_std']
        # sigma_exp = sio.loadmat(os.path.join(MM_base_dir, 'sigma_exp.mat'))
        # self.bfm.sigma_exp = sigma_exp['sigma_exp'].reshape((29, 1))
        # self.bfm.face_region_mask = bfm_utils.get_tight_face_region(self.bfm, MM_base_dir, True)
        # self.bfm.tri_idx = bfm_utils.get_adjacent_triangle_idx(int(self.bfm.nver), self.bfm.model['tri'])
        self.bfm = bfm_utils.load_3DMM(MM_base_dir)
        model_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
        photo_weights = np.minimum(model_info['segbin'][0, :] + model_info['segbin'][1, :] + model_info['segbin'][2, :], 1)
        # photo_weights = model_info['segbin'][2, :]
        self.photo_weights = torch.from_numpy(photo_weights).view(1, 1, -1).float()

        self.bfm_torch = bfm_utils.MorphabelModel_torch(self.bfm)

        # Feature extractor
        self.rgb_net = RGBNet(input_dim=(3, 256, 256))
        self.albedo_net = RGBNetSmallSmall(input_dim=(3, 256, 256))
        # self.fpn = FPN(in_nchs=(512, 512, 256, 128, 64), out_nch=128, n_gn=8)
        self.fpn = FPN(in_nchs=(512, 512, 256, 128, 64, 32), out_nch=64, n_gn=4)
        self.basis_fpn = FPN(in_nchs=(512, 512, 256, 128, 64, 32), out_nch=64, n_gn=4)
        self.albedo_fpn = FPN(in_nchs=(256, 128, 64, 32), out_nch=64, n_gn=4)

        # Step size network
        self.step_net2 = StepSizeNet(in_nchs=(64, 2), nch=64,
                                     # out_nchs=(self.bfm.n_shape_para, self.bfm.n_exp_para, 6))
                                     out_nchs=(1, 1, 1, 1))
        self.step_net3 = StepSizeNet(in_nchs=(64, 2), nch=64,
                                     # out_nchs=(1,))
                                     out_nchs=(1, 1, 1, 1))
        self.step_net4 = StepSizeNet(in_nchs=(64, 2), nch=64,
                                     # out_nchs=(1,))
                                     out_nchs=(1, 1, 1, 1))

        self.ap_step_net0 = StepSizeNet(in_nchs=(3,), nch=8, out_nchs=(1,))
        self.ap_step_net1 = StepSizeNet(in_nchs=(3,), nch=8, out_nchs=(1,))
        self.ap_step_net2 = StepSizeNet(in_nchs=(3,), nch=8, out_nchs=(1,))

        # Adaptive basis generator
        # self.sp_B_net2 = AdaptiveBasisNet(80, 128 + 3, 128, 32, self.bfm, self.bfm_torch, 8, True, 'shape')
        # self.sp_B_net3 = AdaptiveBasisNet(32, 128 + 3, 128, 64, self.bfm, self.bfm_torch, 8, False, 'shape')
        # self.sp_B_net4 = AdaptiveBasisNet(32, 128 + 3, 128, 128, self.bfm, self.bfm_torch, 8, False, 'shape')

        # self.ep_B_net2 = AdaptiveBasisNet(64, 128 + 3, 128, 32, self.bfm, self.bfm_torch, 8, True, 'exp')
        # self.ep_B_net3 = AdaptiveBasisNet(32, 128 + 3, 128, 64, self.bfm, self.bfm_torch, 8, False, 'exp')
        # self.ep_B_net4 = AdaptiveBasisNet(32, 128 + 3, 128, 128, self.bfm, self.bfm_torch, 8, False, 'exp')

        self.exp_model = ExpressionModel(8, 128, 64 + 3, 32, 128, self.bfm, self.bfm_torch, 4, 3)
        self.iden_model = IdentityModel(80, 64 + 3, 32, 128, self.bfm, self.bfm_torch, 4, 3)
        self.albedo_model = AlbedoModel(64, 64 + 3, 16, 128, self.bfm, self.bfm_torch, 4, 3)

        self.sp_B = None
        self.n_sp_para = None
        self.sp_B_net = None
        self.ep_B = None
        self.n_ep_para = None
        self.ep_B_net = None
        self.pre_delta = None

        self.exp_abbr = {'Neutral': 'N', 'Angry': 'A', 'Disgust': 'D', 'Fear': 'F', 'Sad': 'U',
                         'Smile mouth closed': 'H1', 'Smile mouth open': 'H2', 'Surprise': 'S'}
        self.exp_idx = {'A': 0, 'D': 1, 'F': 2, 'U': 3, 'H1': 4, 'H2': 5, 'S': 6}
        self.anchor_ep_norms = nn.Parameter(torch.normal(0., 0.02, (1, 7, 8)))
        self.ep_norm_init = None

    def cuda(self, device=None):
        super(NRMVSOptimization, self).cuda(device)
        self.photo_weights = self.photo_weights.cuda(device)
        self.bfm_torch.cuda(device)
        # self.sp_B_net2.cuda(device)
        # self.sp_B_net3.cuda(device)
        # self.sp_B_net4.cuda(device)
        self.exp_model.cuda(device)
        self.iden_model.cuda(device)
        self.albedo_model.cuda(device)
        # self.ep_B_net2.cuda(device)
        # self.ep_B_net3.cuda(device)
        # self.ep_B_net4.cuda(device)

    def train(self, mode=True):
        super(NRMVSOptimization, self).train(mode)

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.rgb_net.apply(set_bn_eval)

    @torch.enable_grad()
    def compute_vert_from_params(self, sp_norm, ep_norm, pose, mp_norm, ap_norm, level):
        """
        Use BFM to compute positions of vertices of face mesh from params
        :param sp_norm: (N, n_shape_para, 1)
        :param ep_norm: (N, V, n_exp_para, 1)
        :param pose: (N, V, 6)
        :return: vert: mesh vertices. (N, V, nver, 3)
        """
        N, V, _ = pose.shape

        # Process params
        pose = pose.view(N * V, 6)
        pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm = \
            pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pose[:, 5]
        pitch, yaw, roll, s, tx, ty = \
            bfm_utils.denormalize_pose_params(pitch_norm, yaw_norm, roll_norm, s_norm, tx_norm, ty_norm)

        # if sp_norm is not None:
        #     # sp_norm = sp_norm.view(N, 1, self.n_sp_para, 1) \
        #     #                  .expand(N, V, self.n_sp_para, 1).contiguous() \
        #     #                  .view(N * V, self.n_sp_para, 1)
        #     # ep_norm = ep_norm.view(N * V, self.n_ep_para, 1)
        #     sp_norm = self.sp_B_net.denormalize_ap_norm(sp_norm)
        #     # ep_norm = self.ep_B_net.denormalize_ap_norm(ep_norm)

        # Process vertices
        vert = self.bfm_torch.model['shapeMU'].clone().view(1, int(self.bfm.nver), 3)
        vert[:, :, 2] -= 7.5e4
        nver = vert.shape[1]

        if sp_norm is not None:
            # delta_sp_vert = torch.bmm(self.sp_B, sp_norm).view(N, nver, 3)
            # sp_vert = vert + delta_sp_vert + self.pre_delta
            delta_sp_vert = self.iden_model(sp_norm, level)                  # (N, nver, 3)
            sp_vert = delta_sp_vert + vert
            ep_vert = self.exp_model(ep_norm, mp_norm, sp_vert, level)          # (N, V, nver, 3)
            sp_vert = sp_vert.view(N, 1, nver, 3)
            vert = sp_vert + ep_vert
            # delta_ep_vert = torch.bmm(self.ep_B, ep_norm).view(N * V, nver, 3)
            # delta_vert = delta_sp_vert + delta_ep_vert                          # (N * V, nver, 3)
            # pre_delta_vert = self.pre_delta.view(N * V, nver, 3)
            # vert = vert + delta_vert + pre_delta_vert
            # delta_vert_ori = delta_vert.view(N, V, nver, 3)
            # delta_sp_vert = None
        else:
            sp_vert = vert.view(1, 1, nver, 3).expand(N, 1, nver, 3)
            vert = vert.expand(N * V, -1, -1)
            delta_sp_vert = None
            # delta_vert_ori = None

        vert = vert.view(N * V, nver, 3)
        vert_obj = vert.clone()
        angles = torch.stack([pitch, yaw, roll], dim=1)
        zeros = torch.zeros_like(tx)
        t = torch.stack([tx, ty, zeros], dim=1)
        vert = self.bfm_torch.transform(vert, s, angles, t)                     # (N * V, nver, 3)

        # Compute albedo
        if ap_norm is not None:
            albedo = self.albedo_model(ap_norm, ep_norm.detach(), ep_vert.detach(), level)                 # (N, 3, nver) or (N * V, 3, nver)
        else:
            albedo = self.albedo_model.cached_albedo.expand(N, 3, nver)

        return vert.view(N, V, int(self.bfm.nver), 3), vert_obj.view(N, V, int(self.bfm.nver), 3), delta_sp_vert, \
               sp_vert, albedo

    def sample_per_vert_feat(self, feat, vert, H_img, W_img, rand_vert=False, vis_mask=None):
        """
        Project 3D vertices onto the feature map and sample feature vector for each 3D vertex
        :param feat: (N, V, C, H, W)
        :param vert: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :return: vert_feat: feature per vertex. (N, V, C, nver)
        """
        N, V, C, H, W = feat.shape
        feat = feat.view(N * V, C, H, W)
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Random sample subset vertices
        if rand_vert:
            nver = int(self.bfm.nver)
            ratio = 0.05 if self.training else 1.
            face_region_idx = np.arange(0, nver, 1).astype(np.int)[self.bfm.face_region_mask.ravel()]
            nver_rand = int(face_region_idx.shape[0] * ratio)
            vert_idx_rand = np.random.choice(face_region_idx, size=nver_rand, replace=False)
            vert = vert[:, vert_idx_rand, :]
            vis_mask = vis_mask[:, :, :, vert_idx_rand]
        else:
            nver_rand = int(self.bfm.nver)

        # Sample features
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img)
        grid = cam_opt.x_2d_normalize(H_img, W_img, vert_img[:, :, :2].clone()).view(N * V, nver_rand, 1, 2)
        # vert_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros')
        grid_sample = MyGridSample.apply
        vert_feat = grid_sample(grid, feat)

        return vert_feat.view(N, V, C, nver_rand), vis_mask

    def compute_visibility_mask(self, vert):
        """
        Compute visibility mask of the given vertices of face mesh
        :param vert: (N, V, nver, 3)
        :return: vis_mask: visibility mask. (N, V, 1, nver)
        """
        N, V, _, _ = vert.shape
        vert = vert.view(N * V, int(self.bfm.nver), 3)

        # Get visibility map
        # vis_mask = self.bfm_torch.backface_culling_cpu(vert, self.bfm.model['tri'])         # dim: (N * V, nver, 1)
        with torch.no_grad():
            # vis_mask = self.bfm_torch.backface_culling(vert, self.bfm)                    # dim: (N * V, nver, 1)
            # full_face_vis_mask = vis_mask.clone()
            # vis_mask = vis_mask.byte() * self.bfm_torch.face_region_mask.byte()

            tri = np.zeros_like(self.bfm.model['tri'])
            tri[:, 0] = self.bfm.model['tri'][:, 2]
            tri[:, 1] = self.bfm.model['tri'][:, 1]
            tri[:, 2] = self.bfm.model['tri'][:, 0]
            vis_mask_albedo, _ = bfm_utils.compute_mesh_visibility(vert, tri, self.bfm_torch, 256, 256)
            vis_mask_albedo = vis_mask_albedo.unsqueeze(2)
            full_face_vis_mask = vis_mask_albedo.clone()
            vis_mask_albedo = vis_mask_albedo.byte() * self.bfm_torch.face_region_mask.byte()
            vis_mask = vis_mask_albedo.clone()

        # full_face_vis_mask = torch.from_numpy(np.copy(vis_mask)).to(vert.device)            # dim: (N * V, nver, 1)
        # vis_mask = np.logical_and(vis_mask, self.bfm.face_region_mask[np.newaxis, ...])
        # vis_mask = torch.from_numpy(vis_mask).to(vert.device)                               # dim: (N * V, nver, 1)

        return vis_mask.byte().view(N, V, 1, int(self.bfm.nver)),\
               full_face_vis_mask.byte().view(N, V, 1, int(self.bfm.nver)), \
               vis_mask_albedo.byte().view(N, V, 1, int(self.bfm.nver))

    @torch.enable_grad()
    def compute_albedo(self, sh_coeff, vert, ori_images, H_img, W_img):
        N, V, nver, _ = vert.shape
        colors, _ = self.sample_per_vert_feat(ori_images, vert, H_img, W_img)
        tri = np.zeros_like(self.bfm.model['tri'])
        tri[:, 0] = self.bfm.model['tri'][:, 2]
        tri[:, 1] = self.bfm.model['tri'][:, 1]
        tri[:, 2] = self.bfm.model['tri'][:, 0]
        shading = bfm_utils.light_sh_rgb_torch(vert, tri, sh_coeff, self.bfm)
        albedo = colors / shading       # (N, V, 3, nver)
        return albedo

    @torch.enable_grad()
    def image_reconstruction(self, ori_images, vert, vis_mask, albedo, sh_coeff, H_img, W_img, level, update_light=False):
        N, V, nver, _ = vert.shape
        tri = np.zeros_like(self.bfm.model['tri'])
        tri[:, 0] = self.bfm.model['tri'][:, 2]
        tri[:, 1] = self.bfm.model['tri'][:, 1]
        tri[:, 2] = self.bfm.model['tri'][:, 0]

        # Compute normal & SH
        pt0 = vert[:, :, tri[:, 0], :]  # (N, V, ntri, 3)
        pt1 = vert[:, :, tri[:, 1], :]  # (N, V, ntri, 3)
        pt2 = vert[:, :, tri[:, 2], :]  # (N, V, ntri, 3)
        tri_normal = torch.cross(pt0 - pt1, pt0 - pt2, dim=-1)  # (N, V, ntri, 3). normal of each triangle
        tri_normal = torch.cat([tri_normal, torch.zeros_like(tri_normal[:, :, :1, :])], dim=2)  # (N, V, ntri + 1, 3)
        vert_tri_normal = tri_normal[:, :, self.bfm.tri_idx.ravel(), :].view(N, V, nver, self.bfm.tri_idx.shape[1], 3)
        normal = torch.sum(vert_tri_normal, dim=3)  # (N, V, nver, 3)
        n = torch.nn.functional.normalize(normal, dim=3, p=2)

        sh = torch.stack([
            torch.ones_like(n[:, :, :, 0]), n[:, :, :, 0], n[:, :, :, 1], n[:, :, :, 2],
            n[:, :, :, 0] * n[:, :, :, 1], n[:, :, :, 0] * n[:, :, :, 2], n[:, :, :, 1] * n[:, :, :, 2],
            n[:, :, :, 0] ** 2 - n[:, :, :, 1] ** 2, 3 * (n[:, :, :, 2] ** 2) - 1
        ], dim=3)  # (N, V, nver, 9)

        # Sample color_gt
        colors_gt, _ = self.sample_per_vert_feat(ori_images, vert, H_img, W_img)  # (N, V, 3, nver)

        if level < 2:
            albedo = albedo.view(N, 1, 3, nver)
        else:
            albedo = albedo.view(N, V, 3, nver)
        if update_light:
            with torch.no_grad():
                # Solve lighting
                sh_coeff_ori = sh_coeff
                sh_coeff = torch.zeros((N, V, 3, 9), device=ori_images.device)
                shading_gt = (colors_gt / albedo).permute(0, 1, 3, 2)  # (N, V, nver, 3)
                for i in range(N):
                    for j in range(V):
                        mask = vis_mask[i, j, 0, :].cpu().numpy().astype(np.bool)
                        B = shading_gt[i, j, mask, :]
                        A = sh[i, j, mask, :]
                        x = sh_coeff_ori[i, j, :].view(9, 3)
                        B_old = torch.mm(A, x)
                        W = torch.max(torch.norm(B_old - B, dim=1) ** 1.5, torch.tensor(1e-5, device=x.device)).reciprocal()
                        # cur_sh_coeff = torch.lstsq(B, A)[0][:9, :]     # (9, 3)
                        # eye = torch.eye(9, device=ori_images.device) * 0.1
                        # eye[0, 0] = 0
                        # try:
                        cur_sh_coeff = torch.mm(
                            torch.mm(
                                torch.inverse(torch.mm(A.transpose(0, 1) * W.unsqueeze(0), A)), A.transpose(0, 1)
                            ) * W.unsqueeze(0),
                            # + eye), A.transpose(0, 1)),
                            B
                        )
                        # raise Exception('test')
                        # except:
                        #     bg = np.ascontiguousarray(ori_images[i, j].cpu().numpy().transpose((1, 2, 0)))
                        #     plt.subplot(131)
                        #     plt.imshow(bg)
                        #     plt.subplot(132)
                        #     geo_vis = esrc_utils.visualize_geometry(vert[i, j].detach().cpu().numpy(), np.copy(bg), self.bfm.model['tri'])
                        #     plt.imshow(geo_vis.transpose((1, 2, 0)))
                        #     plt.subplot(133)
                        #     colors_vis = colors_gt[i, j].detach().cpu().numpy().transpose((1, 0))
                        #     colors_vis[np.invert(mask)] = 0
                        #     uv_texture_map = mesh.render.render_colors(self.bfm.uv_coords_128, self.bfm.model['tri'],
                        #                                                colors_vis, 128, 128, c=3)
                        #     uv_texture_map = np.minimum(np.maximum(uv_texture_map, 0), 1)
                        #     plt.imshow(uv_texture_map)
                        #     plt.show()

                        sh_coeff[i, j, :] = cur_sh_coeff.permute(1, 0)
        else:
            sh_coeff = torch.stack((sh_coeff[:, :, 0::3], sh_coeff[:, :, 1::3], sh_coeff[:, :, 2::3]), dim=2)

        # Compute reconstructed color
        shading = (sh.view(N, V, 1, nver, 9) * sh_coeff.view(N, V, 3, 1, 9)).sum(dim=4)  # (N, V, 3, nver)
        colors = shading * albedo
        return colors, colors_gt, sh_coeff.transpose(2, 3).contiguous().view(N, V, 27)

    def feature_metric_loss(self, feat, vis_mask):
        N, V, C, nver = feat.shape
        loss = 0
        abs_residuals_mean = [[] for i in range(V)]
        for i in range(V - 1):
            for j in range(i + 1, V):
                # Compute loss
                cur_loss = F.mse_loss(feat[:, i, :, :], feat[:, j, :, :], reduction='none')     # dim: (N, C, nver)
                cur_loss_sum = torch.sum(cur_loss, dim=1, keepdim=True)                         # dim: (N, 1, nver)
                # cur_loss_sum = F.cosine_similarity(feat[:, i, :, :], feat[:, j, :, :], dim=1).view(N, 1, nver)
                mask = vis_mask[:, i, :, :] * vis_mask[:, j, :, :]
                cur_loss_masked = torch.masked_select(cur_loss_sum, mask.bool())
                err = torch.mean(cur_loss_masked)
                loss += err

                # Compute abs residual
                with torch.no_grad():
                    abs_residual = torch.abs(feat[:, i, :, :] - feat[:, j, :, :])           # dim: (N, C, nver)
                    abs_residual_sum = torch.sum(abs_residual * mask.float(), dim=2)        # dim: (N, C)
                    abs_residual_mean = abs_residual_sum / torch.sum(mask.float(), dim=2)   # dim: (N, C)
                    abs_residuals_mean[i].append(abs_residual_mean)
                    abs_residuals_mean[j].append(abs_residual_mean)
        abs_residuals_mean = [torch.mean(torch.stack(abs_residuals_mean[i], dim=0), dim=0) for i in range(V)]
        abs_residuals = torch.stack(abs_residuals_mean, dim=1)                          # dim: (N, V, C)

        return loss / (V * (V - 1.0) / 2.0), abs_residuals

    def landmark_loss(self, vert, kpts_gt, full_face_vis_mask, vis_mask, H_img, W_img):
        N, V, nver, _ = vert.shape
        vert = vert.view(N * V, nver, 3)
        vert_img = self.bfm_torch.to_image(vert, H_img, W_img).view(N, V, nver, 3)
        kpts = vert_img[:, :, self.bfm.kpt_ind, :]                                              # dim: (N, V, 68, 3)

        # Get fixed landmark mask
        kpts_vis_mask = full_face_vis_mask[:, :, 0, self.bfm.kpt_ind]                           # dim: (N, V, 68)
        invar_idx = np.concatenate([np.arange(6, 11), np.arange(17, 68)])
        kpts_fix_mask = torch.zeros_like(kpts_vis_mask)                                         # dim: (N, V, 68)
        kpts_fix_mask[:, :, invar_idx] = 1
        kpts_fix_mask = torch.min(kpts_fix_mask + kpts_vis_mask, torch.ones_like(kpts_fix_mask))
        eyebrows_idx = np.arange(17, 27)

        # Compute loss on fixed landmarks
        loss = F.mse_loss(kpts[:, :, :, :2], kpts_gt[:, :, :, :2], reduction='none')            # dim: (N, V, 68, 2)
        loss_sum = torch.sum(loss, dim=3) * kpts_fix_mask.float()                               # dim: (N, V, 68)
        # loss_sum[:, :, eyebrows_idx] = 0.0
        err = torch.mean(loss_sum)

        # Compute abs residual
        with torch.no_grad():
            fix_abs_residual = torch.abs(kpts[:, :, :, :2] - kpts_gt[:, :, :, :2])                  # dim: (N, V, 68, 2)
            fix_abs_residual_mean = torch.mean(fix_abs_residual * kpts_fix_mask.unsqueeze(3).float(), dim=2)# dim: (N, V, 2)

        # Select closest vert to dynamic landmarks
        invis_factor = (1 - vis_mask) * 1e6                                                     # dim: (N, V, 1, nver)
        vert_to_kpts_dist = torch.norm(
            vert_img[:, :, :, :2].detach().view(N, V, nver, 1, 2) - kpts_gt.view(N, V, 1, 68, 2),
            p=2, dim=-1)                                                                        # dim: (N, V, nver, 68)
        vert_to_kpts_dist = vert_to_kpts_dist + invis_factor.view(N, V, nver, 1).float()
        cls_vert_dix = torch.argmin(vert_to_kpts_dist, dim=2)                                   # dim: (N, V, 68)
        cls_vert = torch.gather(vert_img[:, :, :, :2].view(N, V, nver, 1, 2).expand(N, V, nver, 68, 2),
                                index=cls_vert_dix.view(N, V, 1, 68, 1).expand(N, V, 1, 68, 2),
                                dim=2).view(N, V, 68, 2)                                        # dim: (N, V, 68, 2)

        # Compute loss on dynamic landmarks
        loss = F.mse_loss(cls_vert, kpts_gt, reduction='none')                                  # dim: (N, V, 68, 2)
        loss_sum = torch.sum(loss, dim=3) * (1.0 - kpts_fix_mask.float())                       # dim: (N, V, 68)
        err += torch.mean(loss_sum)

        # Compute abs residual
        with torch.no_grad():
            dyn_abs_residual = torch.abs(cls_vert[:, :, :, :] - kpts_gt[:, :, :, :])                # dim: (N, V, 68, 2)
            dyn_abs_residual_mean = torch.mean(dyn_abs_residual * (1.0 - kpts_fix_mask.float()).unsqueeze(3), dim=2)# dim: (N, V, 2)
            abs_residuals = fix_abs_residual_mean + dyn_abs_residual_mean                           # dim: (N, V, 2)
            abs_residuals[:, :, 0] = abs_residuals[:, :, 0] / (W_img - 1.)
            abs_residuals[:, :, 1] = abs_residuals[:, :, 1] / (H_img - 1.)

        return err, abs_residuals

    def reg_loss(self, sp_norm, ep_norm):
        abs_residual_sp = (sp_norm * sp_norm)[:, :80, :].squeeze(2)
        abs_residual_ep = torch.mean(ep_norm * ep_norm, dim=1).squeeze(2)
        sp_reg = torch.mean(abs_residual_sp)
        ep_reg = torch.mean(abs_residual_ep)
        return sp_reg, abs_residual_sp, abs_residual_ep

    def area_loss(self, vert):
        N, V, nver, _ = vert.shape
        vert = vert * 1e-3

        # Compute area per view
        tri_mask = self.bfm.face_region_tri_mask.ravel()
        pt0 = vert[:, :, self.bfm.model['tri'][tri_mask, 2], :]                 # (N, V, ntri_masked, 3)
        pt1 = vert[:, :, self.bfm.model['tri'][tri_mask, 1], :]                 # (N, V, ntri_masked, 3)
        pt2 = vert[:, :, self.bfm.model['tri'][tri_mask, 0], :]                 # (N, V, ntri_masked, 3)
        tri_area = torch.norm(torch.cross(pt0 - pt1, pt0 - pt2, dim=3), dim=3)  # (N, V, ntri_masked)
        # tri_area_max, _ = torch.max(tri_area, dim=1, keepdim=True)
        # tri_area_max = tri_area_max.detach()                                    # (N, 1, ntri_masked)
        #
        # # Compute loss
        # loss = torch.mean((tri_area - tri_area_max) ** 2)                       # (N, V, ntri_masked)
        #
        # # Compute abs residual
        # abs_residuals = torch.mean(torch.abs(tri_area - tri_area_max), dim=2, keepdim=True)     # (N, V, 1)
        #
        # return loss, abs_residuals

        loss = 0
        abs_residuals_mean = [[] for i in range(V)]
        for i in range(V - 1):
            for j in range(i + 1, V):
                # Compute loss
                cur_loss = F.mse_loss(tri_area[:, i, :], tri_area[:, j, :], reduction='none')   # (N, ntri_masked)
                loss += torch.mean(cur_loss)

                # Compute abs residual
                abs_residual = torch.abs(tri_area[:, i, :] - tri_area[:, j, :])                 # (N, ntri_masked)
                abs_residual_mean = torch.mean(abs_residual, dim=1, keepdim=True)               # (N, 1)
                abs_residuals_mean[i].append(abs_residual_mean)
                abs_residuals_mean[j].append(abs_residual_mean)
        abs_residuals_mean = [torch.mean(torch.stack(abs_residuals_mean[i], dim=0), dim=0) for i in range(V)]
        abs_residuals = torch.stack(abs_residuals_mean, dim=1)                                  # (N, V, 1)

        return loss / (V * (V - 1.0) / 2.0), abs_residuals

    def ep_cosine_loss(self, ep_norm, exp_label):
        ep_norm = ep_norm.squeeze(3)
        N, V, n_param = ep_norm.shape
        ep_norm = F.normalize(ep_norm, dim=2, p=2) * 2.
        anchor_ep_norms = F.normalize(self.anchor_ep_norms, dim=2, p=2) * 2.
        loss = 0.
        for i in range(N):
            for j in range(V):
                if exp_label[i, j, :].sum() == 0:
                    loss += (ep_norm[i, j] ** 2).mean()
                else:
                    pos_exp_idx = torch.argmax(exp_label[i, j, :]).item()
                    ep_pos_dot = (ep_norm[i, j].unsqueeze(0) * anchor_ep_norms[:, pos_exp_idx, :]).sum(dim=1)  # (10,)
                    max_idx = torch.argmax(ep_pos_dot).item()
                    anchors = torch.cat([anchor_ep_norms[max_idx, pos_exp_idx, :].unsqueeze(0),
                                         anchor_ep_norms[:, :pos_exp_idx, :].contiguous().view(-1, n_param),
                                         anchor_ep_norms[:, pos_exp_idx + 1:, :].contiguous().view(-1, n_param)],
                                        dim=0)
                    ep_dot = (ep_norm[i, j].unsqueeze(0) * anchors).sum(dim=1)  # (1 + 10 * 6,)
                    assert ep_dot.shape[0] == 1 + 1 * 6 and len(ep_dot.shape) == 1
                    loss += F.cross_entropy(ep_dot.unsqueeze(0),
                                            torch.tensor((0,), device=ep_dot.device).long())
                    # print(ep_dot)
        # print(model.anchor_ep_norms[:1, :, :5])

        loss /= N
        return loss

    def ep_l2_loss(self, ep_norm, target):
        return ((ep_norm - target) ** 2).mean()

    def image_recon_loss(self, vis_mask, colors, colors_gt):
        # N, V, nver, _ = vert.shape
        # colors_gt, _ = self.sample_per_vert_feat(ori_images, vert, H_img, W_img)            # (N, V, 3, nver)
        # Compute loss
        cur_loss = F.mse_loss(colors, colors_gt, reduction='none')                          # (N, V, 3, nver)
        cur_loss_sum = torch.sum(cur_loss, dim=2, keepdim=True)                             # (N, V, 1, nver)
        cur_loss_masked = cur_loss_sum * vis_mask.float()
        loss = torch.mean(cur_loss_masked)
        # Compute abs residuals
        with torch.no_grad():
            abs_residuals = F.mse_loss(colors, colors_gt, reduction='none')                     # (N, V, 3, nver)
            abs_residuals_masked = abs_residuals * vis_mask.float()
            abs_residuals = abs_residuals_masked.mean(dim=3)                                    # (N, V, 3)
        return loss, abs_residuals

    def gradient_descent(self, images, ori_images, albedo, colors, colors_gt, feat, kpts, vert, vert_obj, H_img, W_img,
                         sp_norm, pose, ep_norm, mp_norm, sh_coeff, ap_norm, vis_mask,
                         full_face_vis_mask, vis_mask_albedo, step_net, step_s, step_b, level, exp_label, itr):
        """
        Update params with 1st order optimization
        :param images: (N, V, C_img, H_img, W_img)
        :param feat: (N, V, C, H, W)
        :param kpts: (N, V, 68, 2)
        :param vert: (N, V, nver, 3)
        :param vert_obj: (N, V, nver, 3)
        :param H_img: the default image height when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param W_img: the default image width when compute vert.
                      (vert can be directly orthogonal project to (H_img x W_img) image)
        :param sp_norm:
        :param pose:
        :param ep_norm:
        :param vis_mask: visibility mask of tight face region. np.array (N, V, 1, nver)
        :param full_face_vis_mask: visibility mask of full BFM mesh. np.array (N, V, 1, nver)
        :return:
        """
        N, V, C, _, _ = feat.shape
        with torch.enable_grad():
            vert_feat, vis_mask_rand = self.sample_per_vert_feat(feat, vert, H_img, W_img, True, vis_mask)  # dim: (N, V, C, nver)
            feat_loss, feat_abs_res = self.feature_metric_loss(vert_feat, vis_mask_rand)
            # print(feat_loss)
            # print(torch.norm(feat_abs_res, dim=1))
            land_loss, land_abs_res = self.landmark_loss(vert, kpts, full_face_vis_mask, vis_mask, H_img, W_img)
            # print(torch.norm(land_abs_res, dim=1))
            # reg_loss, sp_abs_res, ep_abs_res = self.reg_loss(sp_norm, ep_norm)
            # if level > 0:
            #     area_loss, area_abs_res = self.area_loss(vert_obj)
            #     print(area_loss)
            # else:
            #     with torch.no_grad():
            #         area_loss, area_abs_res = self.area_loss(vert_obj)
            #     print(area_loss)
            # print(area_abs_res.shape)
            # ep_loss = self.ep_l2_loss(ep_norm, self.ep_norm_init) #self.ep_cosine_loss(ep_norm, exp_label)
            # albedo_loss, albedo_abs_res = self.feature_metric_loss(albedo, vis_mask)
            # if level > 1:
            colors_loss, colors_abs_res = self.image_recon_loss(vis_mask_albedo, colors, colors_gt)
            # else:
            # colors_loss, colors_abs_res = self.image_recon_loss(ori_images, vert.detach(), vis_mask, colors, H_img, W_img)
            loss = 0.25 * feat_loss + 0.025 * land_loss + colors_loss #+ 0.1 * ep_loss
            # if level > 0:
            #     loss = loss + 2.0 * area_loss
            loss = loss * N

        # Compute gradient and update params
        sp_norm_grad, pose_grad, ep_norm_grad, mp_norm_grad, ap_norm_grad = \
            torch.autograd.grad(loss, [sp_norm, pose, ep_norm, mp_norm, ap_norm], create_graph=self.training)
        # sp_norm_grad, pose_grad, mp_norm_grad = \
        #     torch.autograd.grad(loss, [sp_norm, pose, mp_norm], create_graph=self.training)

        """-------------------------------------------------------------------------------------------------------------
        for analysis
        """
        # print('mp_norm branch feat grad -------------------------------')
        # mp_norm_uv_ori0_grad = \
        #     torch.autograd.grad(loss, [self.exp_model.mp_norm_uv_ori0,], create_graph=self.training)
        # print(mp_norm_uv_ori0_grad[0].max(), mp_norm_uv_ori0_grad[0].min())
        # mp_norm_uv_ori1_grad = \
        #     torch.autograd.grad(loss, [self.exp_model.mp_norm_uv_ori1, ], create_graph=self.training)
        # print(mp_norm_uv_ori1_grad[0].max(), mp_norm_uv_ori1_grad[0].min())
        # mp_norm_uv_ori2_grad = \
        #     torch.autograd.grad(loss, [self.exp_model.mp_norm_uv_ori2, ], create_graph=self.training)
        # print(mp_norm_uv_ori2_grad[0].max(), mp_norm_uv_ori2_grad[0].min())
        """-------------------------------------------------------------------------------------------------------------
        for analysis end
        """

        N = vert.shape[0]
        feat_abs_res = torch.mean(feat_abs_res, dim=1).detach()          # (N, C)
        land_abs_res = torch.mean(land_abs_res, dim=1).detach()          # (N, 2)
        colors_abs_res = torch.mean(colors_abs_res, dim=1).detach()      # (N, 3)
        (sp_step_size, pose_step_size, ep_step_size, mp_step_size), raw_step_size = \
            step_net((feat_abs_res, land_abs_res), step_s[:-1], step_b[:-1])
        ap_step_net = getattr(self, 'ap_step_net' + str(level))
        (ap_step_size,), _ = ap_step_net((colors_abs_res,), step_s[-1:], step_b[-1:])
        # print(step_s[:-1], step_b[:-1], step_s[-1:], step_b[-1:])
        sp_step_size = sp_step_size.view(N, 1, 1)
        pose_step_size = pose_step_size.view(N, 1, 1)
        ep_step_size = ep_step_size.view(N, 1, 1, 1)
        mp_step_size = mp_step_size.view(N, 1, 1)
        # sh_step_size = sh_step_size.view(N, 1, 1)
        ap_step_size = ap_step_size.view(N, 1, 1)

        with torch.enable_grad():
            sp_norm = sp_norm - sp_step_size * sp_norm_grad
            pose = pose - pose_step_size * pose_grad * V
            ep_norm = ep_norm - ep_step_size * ep_norm_grad * V
            mp_norm = mp_norm - mp_step_size * mp_norm_grad
            # sh_coeff = sh_coeff - sh_step_size * sh_coeff_grad * V
            # sh_coeff[:, 0, :3] = 1.
            ap_norm = ap_norm - ap_step_size * ap_norm_grad

        """-------------------------------------------------------------------------------------------------------------
        for analysis
        """
        # print(self.exp_model.EV)
        # print(sp_norm_grad.max(), sp_norm_grad.min())
        # print(pose_grad.max(), pose_grad.min())
        # print(ep_loss.max(), ep_loss.min())
        # print(ep_norm.max(), ep_norm.min())
        # print(ep_norm_grad.max(), ep_norm_grad.min())
        # if level > 0:
        #     # print(mp_norm.view(-1))
        # print(mp_norm_grad.max(), mp_norm_grad.min())
        # print(sh_coeff_grad.max(), sh_coeff_grad.min())
        # print(ap_norm_grad.max(), ap_norm_grad.min())
        # print('---------------------------------------')
        """-------------------------------------------------------------------------------------------------------------
        for analysis end
        """

        new_vert, new_vert_obj, delta_vert, new_sp_vert, new_albedo = \
            self.compute_vert_from_params(sp_norm, ep_norm, pose, mp_norm, ap_norm, level)
        vis_mask, full_face_vis_mask, vis_mask_albedo = self.compute_visibility_mask(new_vert)  # dim: (N, V, 1, nver)
        N, V, nver, _ = new_vert.shape
        new_vert_img = self.bfm_torch.to_image(new_vert.view(N * V, nver, 3), H_img, W_img).view(N, V, nver, 3)
        # new_colors, _ = self.sample_per_vert_feat(images, new_vert, H_img, W_img)
        # if level > 1:
        #     new_colors, new_colors_gt = self.image_reconstruction(ori_images, new_vert, vis_mask, new_albedo, H_img, W_img)
        # else:
        new_colors, new_colors_gt, sh_coeff = self.image_reconstruction(ori_images, new_vert.detach(), vis_mask_albedo,
                                                                        new_albedo, sh_coeff, H_img, W_img, level, True if itr == 2 else False)
        return sp_norm, pose, ep_norm, mp_norm, sh_coeff, ap_norm, new_vert, new_vert_obj, new_sp_vert, new_vert_img, \
               new_colors, new_colors_gt, new_albedo, vis_mask, full_face_vis_mask, vis_mask_albedo, raw_step_size, delta_vert

    def forward(self, kpts, sp_norm, pose, images, ori_images, exp_label, only_regr):
        """
        Main loop of the NRMVS optimization
        :param kpts: 2D lamdmarks. (N, V, 68, 2)
        :param sp_norm: (N, n_shape_para, 1)
        :param pose: (N, V, 6)
        :param images: (N, V, C, H, W)
        :param ori_images: (N, V, C, H, W)
        :param only_regr: only pose regression or whole optimization. bool
        :return:
        """
        N, V, _ = pose.shape
        _, _, C_img, H_img, W_img = ori_images.shape

        # Set initial params & outputs
        verts = []
        verts_obj = []
        sp_verts = []
        verts_img = []
        colors_list = []
        vis_masks = []
        full_face_vis_masks = []
        vis_masks_albedo = []
        raw_step_sizes = []
        albedo_list = []
        sp_B_uv_list = []
        delta_vert_list = []
        self.exp_model.init_model()
        self.iden_model.init_model()
        self.albedo_model.init_model()
        self.ep_norm_init = None

        # Generate initial vertices (mesh)
        vert, vert_obj, _, sp_vert, albedo = self.compute_vert_from_params(None, None, pose, None, None, 0)
        nver = vert.shape[2]
        vert_img = self.bfm_torch.to_image(vert.view(N * V, nver, 3), H_img, W_img).view(N, V, nver, 3)
        colors, _ = self.sample_per_vert_feat(images, vert, H_img, W_img)
        vis_mask, full_face_vis_mask, vis_mask_albedo = self.compute_visibility_mask(vert)       # dim: (N, V, 1, nver)
        verts.append(vert)
        verts_obj.append(vert_obj)
        sp_verts.append(sp_vert)
        verts_img.append(vert_img)
        colors_list.append(colors)
        vis_masks.append(vis_mask)
        full_face_vis_masks.append(full_face_vis_mask)
        vis_masks_albedo.append(vis_mask_albedo)
        albedo_list.append(self.sample_per_vert_feat(ori_images, vert.detach(), H_img, W_img)[0])

        if only_regr:
            return verts, verts_obj, sp_verts, verts_img, vis_masks, full_face_vis_masks, albedo_list, colors_list, \
                   raw_step_sizes, sp_B_uv_list, delta_vert_list

        # Get feature pyramid
        rgb_feats = self.rgb_net(images.view(N * V, C_img, H_img, W_img))           # dim: [(N * V, c, h, w)]
        fpn_feats = self.fpn(rgb_feats[0:6])
        basis_fpn_feats = self.basis_fpn(rgb_feats[0:6])
        albedo_feats = self.albedo_net(images.view(N * V, C_img, H_img, W_img))     # dim: [(N * V, c, h, w)]
        albedo_fpn_feats = self.albedo_fpn(albedo_feats[0:4])

        # Multi-Level Optimization
        # sp_norm = torch.zeros((N, self.iden_model.n_para, 1), requires_grad=True, device=images.device)
        # ep_norm_init = torch.zeros((N, V, self.exp_model.n_para, 1), requires_grad=False, device=images.device)
        # for i in range(N):
        #     for j in range(V):
        #         if exp_label[i, j, :].sum() == 0:
        #             continue
        #         pos_exp_idx = torch.argmax(exp_label[i, j, :]).item()
        #         ep_norm_init[i, j, :, 0] = self.anchor_ep_norms[0, pos_exp_idx, :]
        # self.ep_norm_init = ep_norm_init
        # ep_norm = ep_norm_init.clone()
        ep_norm = torch.zeros((N, V, self.exp_model.n_para, 1), requires_grad=True, device=images.device)
        # mp_norm = torch.zeros((N, self.exp_model.model_n_para, 1), requires_grad=True, device=images.device)
        sh_coeff = torch.zeros((N, V, 27), requires_grad=True, device=images.device)
        sh_coeff[:, :, :3] = 1.
        # sh_coeff = None
        _, _, sh_coeff = self.image_reconstruction(ori_images, vert.detach(), vis_mask_albedo, albedo, sh_coeff, H_img, W_img, 0, True)
        step_s = [(2., 2., 4., 4., 4.), (4., 4., 4., 4., 4.), (4., 4., 4., 4., 4.)]
        step_b = [(0., -2.5, -3., -2., -2.5), (0., -2.5, -3.5, -2., -2.5), (0., -2.5, -4., -2., -2.5)]


        """-------------------------------------------------------------------------------------------------------------
        for analysis
        """
        self.ep_norms = [ep_norm]
        """-------------------------------------------------------------------------------------------------------------
        for analysis end
        """


        for l in range(2, len(fpn_feats)):
            feat = fpn_feats[l]
            # basis_feat = basis_fpn_feats[l]
            _, C_opt, H, W = feat.shape
            feat = feat.view(N, V, C_opt, H, W)
            # _, C_basis, H, W = basis_feat.shape
            # basis_feat = basis_feat.view(N, V, C_basis, H, W)
            cur_level_verts = []
            cur_level_verts_obj = []
            cur_level_sp_verts = []
            cur_level_verts_img = []
            cur_level_colors_list = []
            cur_level_vis_masks = []
            cur_level_full_face_vis_masks = []
            cur_level_vis_masks_albedo = []
            cur_level_raw_step_sizes = []
            cur_level_delta_vert_list = []
            cur_level_albedo_list = []

            # if sp_norm is None:
            #     self.pre_delta = torch.zeros((N, nver, 3), device=images.device)
            # else:
            #     self.pre_delta = self.pre_delta + delta_vert.detach()
                # Block the gradient to previous level AdapBasis during training
            # Generate sp basis
            # sp_B_net = getattr(self, 'sp_B_net' + str(l))
            # self.sp_B_net = sp_B_net
            # self.n_sp_para = sp_B_net.n_para
            # sp_norm = torch.zeros((N, self.n_sp_para, 1), requires_grad=True, device=images.device)
            # self.sp_B, sp_B_uv = sp_B_net(basis_feat, vert.detach(), vis_mask, H_img, W_img)  # (N, nver * 3, n_sp_para)
            # sp_B_uv_list.append(sp_B_uv)

            # Update identity model
            self.iden_model.update_uv_feats(basis_fpn_feats[2:], vert.detach(), vis_mask, H_img, W_img, l - 2)
            if l - 2 > 0:
                self.iden_model.cached_sp_vert = delta_vert

            # Update expression model
            self.exp_model.update_uv_feats(basis_fpn_feats[2:], vert.detach(), vis_mask, H_img, W_img, l - 2, exp_label)

            # Update albedo model
            self.albedo_model.update_uv_feats(albedo_fpn_feats, vert.detach(), vis_mask_albedo, H_img, W_img, l - 2)
            if l - 2 > 0:
                self.albedo_model.cached_albedo = albedo

            sp_norm = torch.zeros((N, self.iden_model.n_para, 1), requires_grad=True, device=images.device)
            mp_norm = torch.zeros((N, self.exp_model.model_n_para, 1), requires_grad=True, device=images.device)
            ap_norm = torch.zeros((N, self.albedo_model.n_para, 1), requires_grad=True, device=images.device)

            # Generate exp basis
            # ep_B_net = getattr(self, 'ep_B_net' + str(l))
            # self.ep_B_net = ep_B_net
            # self.n_ep_para = ep_B_net.n_para
            # ep_norm = torch.zeros((N, V, self.n_ep_para, 1), requires_grad=True, device=images.device)
            # self.ep_B, ep_B_uv = ep_B_net(basis_feat, vert.detach(), vis_mask, H_img, W_img)  # (N, nver * 3, n_ep_para)
            # self.ep_B = self.ep_B.view(N, 1, int(self.bfm.nver) * 3, self.n_ep_para) \
            #                      .expand(N, V, int(self.bfm.nver) * 3, self.n_ep_para) \
            #                      .contiguous().view(N * V, int(self.bfm.nver) * 3, self.n_ep_para)

            with torch.enable_grad():
                # sp_norm = sp_norm.detach()
                # sp_norm.requires_grad = True
                ep_norm = ep_norm.detach()
                ep_norm.requires_grad = True
                pose = pose.detach()
                pose.requires_grad = True
                # mp_norm = mp_norm.detach()
                # mp_norm.requires_grad = True
                # sh_coeff = sh_coeff.detach()
                # sh_coeff.requires_grad = True
            vert, vert_obj, delta_vert, sp_vert, albedo = \
                self.compute_vert_from_params(sp_norm, ep_norm, pose, mp_norm, ap_norm, l - 2)
            vis_mask, full_face_vis_mask, vis_mask_albedo = self.compute_visibility_mask(vert)           # dim: (N, V, 1, nver)
            # if l - 2 > 1:
            #     colors, colors_gt = self.image_reconstruction(ori_images, vert, vis_mask, albedo, H_img, W_img)
            # else:
            colors, colors_gt, sh_coeff = self.image_reconstruction(ori_images, vert.detach(), vis_mask_albedo, albedo, sh_coeff, H_img, W_img, l - 2)

            # if l - 2 > 0:
            #     cur_level_verts.append(vert)
            # cur_level_sp_verts.append(sp_vert)

            """-------------------------------------------------------------------------------------------------------------
            for analysis
            """
            self.ep_norms.append([])
            """-------------------------------------------------------------------------------------------------------------
            for analysis end
            """

            # Optimization iter
            for i in range(3):
                # Compute updated params
                sp_norm, pose, ep_norm, mp_norm, sh_coeff, ap_norm, vert, vert_obj, sp_vert, vert_img, colors, colors_gt, albedo, \
                vis_mask, full_face_vis_mask, vis_mask_albedo, raw_step_size, \
                    delta_vert = self.gradient_descent(images, ori_images, albedo, colors, colors_gt, feat, kpts, vert, vert_obj, H_img, W_img,
                                                       sp_norm, pose, ep_norm, mp_norm, sh_coeff, ap_norm, vis_mask, full_face_vis_mask, vis_mask_albedo,
                                                       getattr(self, 'step_net' + str(l)),
                                                       step_s[l - 2], step_b[l - 2], l - 2, exp_label, i)
                cur_level_verts.append(vert)
                cur_level_verts_obj.append(vert_obj)
                cur_level_sp_verts.append(sp_vert)
                cur_level_verts_img.append(vert_img)
                cur_level_colors_list.append(colors)
                cur_level_vis_masks.append(vis_mask)
                cur_level_full_face_vis_masks.append(full_face_vis_mask)
                cur_level_vis_masks_albedo.append(vis_mask_albedo)
                cur_level_raw_step_sizes.append(raw_step_size)
                cur_level_delta_vert_list.append(delta_vert)
                cur_level_albedo_list.append(albedo)

                """-------------------------------------------------------------------------------------------------------------
                for analysis
                """
                self.ep_norms[-1].append(ep_norm)
                """-------------------------------------------------------------------------------------------------------------
                for analysis end
                """

            albedo_list.append(cur_level_albedo_list)
            verts.append(cur_level_verts)
            verts_obj.append(cur_level_verts_obj)
            sp_verts.append(cur_level_sp_verts)
            verts_img.append(cur_level_verts_img)
            colors_list.append(cur_level_colors_list)
            vis_masks.append(cur_level_vis_masks)
            full_face_vis_masks.append(cur_level_full_face_vis_masks)
            vis_masks_albedo.append(cur_level_vis_masks_albedo)
            raw_step_sizes.append(cur_level_raw_step_sizes)
            delta_vert_list.append(cur_level_delta_vert_list)

        self.mp_norm = mp_norm
        self.pose = pose
        self.fpn_feats = fpn_feats
        self.sh_coeff = sh_coeff

        return verts, verts_obj, sp_verts, verts_img, vis_masks, full_face_vis_masks, vis_masks_albedo, albedo_list, colors_list, \
               raw_step_sizes, sp_B_uv_list, delta_vert_list


class INORig(BaseNet):

    def __init__(self, opt_step_size=1e-5, MM_base_dir='./external/face3d/examples/Data/BFM'):
        super(INORig, self).__init__()
        self.regressor = RegressionNet()
        self.opt_layer = NRMVSOptimization(opt_step_size=opt_step_size, MM_base_dir=MM_base_dir)

    def save_net_def(self, dir):
        super(INORig, self).save_net_def(dir)
        shutil.copy(os.path.realpath(__file__), dir)

    def cuda(self, device=None):
        super(INORig, self).cuda(device)
        self.regressor.cuda(device)
        self.opt_layer.cuda(device)

    def parameters_cnns(self):
        return itertools.chain(
            self.opt_layer.rgb_net.parameters(),
            self.opt_layer.fpn.parameters(),
            self.opt_layer.adap_B_net2.parameters(),
            self.opt_layer.adap_B_net3.parameters()
        )

    def parameters_stepnets(self):
        return itertools.chain(
            self.opt_layer.step_net2.parameters(),
            self.opt_layer.step_net3.parameters()
        )

    def forward(self, images, ori_images, kpts, kpts_3d, exp_label, only_regr):
        exp_label = None
        if only_regr:
            pose = self.regressor.forward(images)
        else:
            self.regressor.eval()
            with torch.no_grad():
                pose = self.regressor.forward(images)

        N, V, _, _, _ = images.shape
        sp_norm = torch.zeros((N, self.opt_layer.bfm.n_shape_para, 1), device=images.device)
        ep_norm = torch.zeros((N, self.opt_layer.bfm.n_exp_para, 1), device=images.device)
        opt_verts, opt_verts_obj, opt_sp_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, opt_vis_masks_albedo, \
            albedo_list, colors_list, raw_step_sizes, adap_B_uv_list, delta_vert_list = \
            self.opt_layer.forward(kpts, sp_norm, pose, images, ori_images, exp_label, only_regr)

        return pose, sp_norm, ep_norm, \
               opt_verts, opt_verts_obj, opt_sp_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, opt_vis_masks_albedo, \
               albedo_list, colors_list, raw_step_sizes, adap_B_uv_list, delta_vert_list


if __name__ == '__main__':
    with torch.cuda.device(0):
        model = RegressionNet()
        model.cuda()
        rand_input = torch.rand(2, 4, 3, 256, 256).cuda()
        sp_norm, ep_norm, pose, _ = model.forward(rand_input)
        print(sp_norm.shape, ep_norm.shape, pose.shape)
