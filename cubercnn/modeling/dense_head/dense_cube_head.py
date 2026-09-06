import logging
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.structures import Instances, Boxes
from detectron2.utils.registry import Registry
from detectron2.utils.events import get_event_storage
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)
from pytorch3d.transforms.so3 import so3_relative_angle

from cubercnn.modeling.dense_head.assigner import build_assigner, build_tal_assigner
from cubercnn import util

logger = logging.getLogger(__name__)

E_CONSTANT = 2.71828183
SQRT_2_CONSTANT = 1.41421356

DENSE_CUBE_HEAD_REGISTRY = Registry("DENSE_CUBE_HEAD")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@DENSE_CUBE_HEAD_REGISTRY.register()
class DenseCubeHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        in_features: List[str],
        fpn_strides: List[int],
        num_convs: int,
        conv_dim: int,
        pose_type: str,
        use_conf: bool,
        cluster_bins: int,
        prior_prob: float,
        focal_alpha: float,
        focal_gamma: float,
        assigner,           #TODO use yolo TAL in the future
        loss_w_3d: float,
        loss_w_cls: float,
        loss_w_box2d: float,
        loss_w_xy: float,
        loss_w_z: float,
        loss_w_dims: float,
        loss_w_pose: float,
        loss_w_joint: float,
        test_score_thresh: float,
        test_topk_candidates: int,
        test_nms_thresh: float,
        test_max_detections: int,
        z_type: str,
        dims_priors_enabled: bool,
        dims_priors_func: str,
        chamfer_pose: bool,
        disentangled_loss: bool,
        inverse_z_weight: bool,
        allocentric_pose: bool,
        virtual_depth: bool,
        virtual_focal: float,
        priors: Optional[dict] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_features = in_features
        self.fpn_strides = fpn_strides
        self.num_levels = len(in_features)
        self.pose_type = pose_type
        self.use_conf = use_conf
        self.cluster_bins = max(cluster_bins, 1)
        self.assigner = assigner

        self.loss_w_3d = loss_w_3d
        self.loss_w_cls = loss_w_cls
        self.loss_w_box2d = loss_w_box2d
        self.loss_w_xy = loss_w_xy
        self.loss_w_z = loss_w_z
        self.loss_w_dims = loss_w_dims
        self.loss_w_pose = loss_w_pose
        self.loss_w_joint = loss_w_joint

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.test_max_detections = test_max_detections

        self.z_type = z_type
        self.dims_priors_enabled = dims_priors_enabled
        self.dims_priors_func = dims_priors_func
        self.chamfer_pose = chamfer_pose
        self.disentangled_loss = disentangled_loss
        self.inverse_z_weight = inverse_z_weight
        self.allocentric_pose = allocentric_pose
        self.virtual_depth = virtual_depth
        self.virtual_focal = virtual_focal

        if self.dims_priors_enabled and priors is not None:
            self.priors_dims_per_cat = nn.Parameter(
                torch.FloatTensor(priors['priors_dims_per_cat']).unsqueeze(0)
            )
        else:
            self.priors_dims_per_cat = nn.Parameter(torch.ones(1, num_classes, 2, 3))

        if self.cluster_bins > 1 and priors is not None:
            priors_z_scales = torch.stack([torch.FloatTensor(p[1]) for p in priors['priors_bins']])
            self.priors_z_scales = nn.Parameter(priors_z_scales)
        else:
            self.priors_z_scales = nn.Parameter(torch.ones(num_classes, self.cluster_bins))

        if self.z_type == 'clusters':
            assert self.cluster_bins > 1, 'z_type=clusters needs cluster_bins > 1'
            if priors is None:
                self.priors_z_stats = nn.Parameter(torch.ones(num_classes, self.cluster_bins, 2).float())
            else:
                priors_z_stats = torch.cat([torch.FloatTensor(p[2]).unsqueeze(0) for p in priors['priors_bins']])
                self.priors_z_stats = nn.Parameter(priors_z_stats)

        pose_dim = {'6d': 6, 'quaternion': 4, 'euler': 3}[self.pose_type]

        # self.cls_towers = nn.ModuleList()
        # self.box2d_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.box2d_preds = nn.ModuleList()
        self.cube_2d_deltas_preds = nn.ModuleList()
        self.cube_dims_preds = nn.ModuleList()
        self.cube_z_preds = nn.ModuleList()
        self.cube_pose_preds = nn.ModuleList()
        if self.use_conf:
            self.cube_uncert_preds = nn.ModuleList()

        # self.cls_towers.append(self._make_tower(conv_dim, num_convs))
        # self.box2d_towers.append(self._make_tower(conv_dim, num_convs))
        self.cls_towers = self._make_tower(conv_dim, num_convs)
        self.box2d_towers = self._make_tower(conv_dim, num_convs)
        self.cube_tower = self._make_tower(conv_dim, num_convs)
        for _ in range(self.num_levels):
            # 2D branch
            self.cls_preds.append(nn.Conv2d(conv_dim, num_classes, 1))
            self.box2d_preds.append(nn.Conv2d(conv_dim, 4, 1))

            # 3D Cube branch
            self.cube_2d_deltas_preds.append(nn.Conv2d(conv_dim, 2, 1))
            self.cube_z_preds.append(nn.Conv2d(conv_dim, self.cluster_bins, 1))
            self.cube_dims_preds.append(nn.Conv2d(conv_dim, 3, 1))
            self.cube_pose_preds.append(nn.Conv2d(conv_dim, pose_dim, 1))
            if self.use_conf:
                self.cube_uncert_preds.append(nn.Conv2d(conv_dim, 1, 1))

        self._init_weights(prior_prob)

    def _make_tower(self, conv_dim: int, num_convs: int) -> nn.Sequential:
        layers = []
        for _ in range(num_convs):
            layers.append(ConvBlock(conv_dim, conv_dim, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def _init_weights(self, prior_prob: float):
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_pred in self.cls_preds:
            nn.init.constant_(cls_pred.bias, bias_value)
            nn.init.normal_(cls_pred.weight, std=0.01)
        for box_pred in self.box2d_preds:
            nn.init.normal_(box_pred.weight, std=0.01)
            nn.init.constant_(box_pred.bias, 0.0)

        for preds in [self.cube_2d_deltas_preds, self.cube_dims_preds, self.cube_pose_preds]:
            for p in preds:
                nn.init.normal_(p.weight, std=0.001)
                nn.init.constant_(p.bias, 0.0)

        if self.use_conf:
            for uncert_pred in self.cube_uncert_preds:
                nn.init.normal_(uncert_pred.weight, std=0.001)
                nn.init.constant_(uncert_pred.bias, 5.0)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], priors: Optional[dict] = None):
        in_features = cfg.MODEL.DENSE_HEAD.IN_FEATURES
        return {
            "num_classes": cfg.MODEL.DENSE_HEAD.NUM_CLASSES,
            "in_features": in_features,
            "fpn_strides": [input_shape[f].stride for f in in_features],
            "num_convs": cfg.MODEL.DENSE_HEAD.NUM_CONVS,
            "conv_dim": cfg.MODEL.DENSE_HEAD.CONV_DIM,
            "pose_type": cfg.MODEL.DENSE_HEAD.POSE_TYPE,
            "use_conf": cfg.MODEL.DENSE_HEAD.USE_CONFIDENCE,
            "cluster_bins": cfg.MODEL.DENSE_HEAD.CLUSTER_BINS,
            "prior_prob": cfg.MODEL.DENSE_HEAD.PRIOR_PROB,
            "focal_alpha": cfg.MODEL.DENSE_HEAD.FOCAL_ALPHA,
            "focal_gamma": cfg.MODEL.DENSE_HEAD.FOCAL_GAMMA,
            "assigner": build_assigner(cfg),
            "loss_w_3d": cfg.MODEL.DENSE_HEAD.LOSS_W_3D if hasattr(cfg.MODEL.DENSE_HEAD, "LOSS_W_3D") else 1.0,
            "loss_w_cls": cfg.MODEL.DENSE_HEAD.LOSS_W_CLS,
            "loss_w_box2d": cfg.MODEL.DENSE_HEAD.LOSS_W_BOX2D,
            "loss_w_xy": cfg.MODEL.DENSE_HEAD.LOSS_W_XY,
            "loss_w_z": cfg.MODEL.DENSE_HEAD.LOSS_W_Z,
            "loss_w_dims": cfg.MODEL.DENSE_HEAD.LOSS_W_DIMS,
            "loss_w_pose": cfg.MODEL.DENSE_HEAD.LOSS_W_POSE,
            "loss_w_joint": cfg.MODEL.DENSE_HEAD.LOSS_W_JOINT if hasattr(cfg.MODEL.DENSE_HEAD, "LOSS_W_JOINT") else 0.0,
            "test_score_thresh": cfg.MODEL.DENSE_HEAD.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.DENSE_HEAD.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.DENSE_HEAD.NMS_THRESH_TEST,
            "test_max_detections": cfg.MODEL.DENSE_HEAD.MAX_DETECTIONS_PER_IMAGE,
            "z_type": cfg.MODEL.DENSE_HEAD.Z_TYPE,
            "dims_priors_enabled": cfg.MODEL.DENSE_HEAD.DIMS_PRIORS_ENABLED,
            "dims_priors_func": cfg.MODEL.DENSE_HEAD.DIMS_PRIORS_FUNC,
            "chamfer_pose": cfg.MODEL.DENSE_HEAD.CHAMFER_POSE if hasattr(cfg.MODEL.DENSE_HEAD, "CHAMFER_POSE") else False,
            "disentangled_loss": cfg.MODEL.DENSE_HEAD.DISENTANGLED_LOSS,
            "inverse_z_weight": cfg.MODEL.DENSE_HEAD.INVERSE_Z_WEIGHT,
            "allocentric_pose": cfg.MODEL.DENSE_HEAD.ALLOCENTRIC_POSE,
            "virtual_depth": cfg.MODEL.DENSE_HEAD.VIRTUAL_DEPTH,
            "virtual_focal": cfg.MODEL.DENSE_HEAD.VIRTUAL_FOCAL,
            "priors": priors,
        }

    # ===================================================================
    # 前向傳播 (Forward Pass)
    # ===================================================================
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_logits, box2d_reg = [], []
        cube_2d_deltas, cube_z, cube_dims, cube_pose, cube_uncert = [], [], [], [], []

        for level, f in enumerate(self.in_features):
            x = features[f]
            stride = self.fpn_strides[level]
            # cls_feat = self.cls_towers[level](x)
            # box_feat = self.box2d_towers[level](x)
            cls_feat = self.cls_towers(x)
            box_feat = self.box2d_towers(x)
            cube_feat = self.cube_tower(x)

            cls_logits.append(self.cls_preds[level](cls_feat))
            box2d_reg.append(F.relu(self.box2d_preds[level](box_feat)) * stride)

            cube_2d_deltas.append(self.cube_2d_deltas_preds[level](cube_feat))
            cube_dims.append(self.cube_dims_preds[level](cube_feat))
            cube_pose.append(self.cube_pose_preds[level](cube_feat))
            cube_z.append(self.cube_z_preds[level](cube_feat))
            if self.use_conf:
                cube_uncert.append(self.cube_uncert_preds[level](cube_feat).clamp(min=0.01))

        return {
            "cls_logits": cls_logits,
            "box2d_reg": box2d_reg,
            "cube_2d_deltas": cube_2d_deltas,
            "cube_z": cube_z,
            "cube_dims": cube_dims,
            "cube_pose": cube_pose,
            "cube_uncert": cube_uncert if self.use_conf else None,
        }
    
    def compute_locations(self, features: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        locations = []
        strides_per_point = []
        for f, stride in zip(self.in_features, self.fpn_strides):
            h, w = features[f].shape[-2:]
            device = features[f].device
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            points = torch.stack((shift_x, shift_y), dim=1) + stride // 2
            locations.append(points)
            strides_per_point.append(torch.full((points.shape[0],), stride, dtype=torch.float32, device=device))
        return locations, strides_per_point

    def _decode_box2d(self, points: torch.Tensor, box2d_reg: torch.Tensor) -> torch.Tensor:
        x1 = points[:, 0] - box2d_reg[:, 0]
        y1 = points[:, 1] - box2d_reg[:, 1]
        x2 = points[:, 0] + box2d_reg[:, 2]
        y2 = points[:, 1] + box2d_reg[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)

    def l1_loss(self, vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(vals, target, reduction='none', beta=0.0)

    def chamfer_loss(self, vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = vals.shape[0]
        xx = vals.view(B, 8, 1, 3)
        yy = target.view(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)
        return l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1)

    def _safely_reduce(self, loss: torch.Tensor) -> torch.Tensor:
        valid = (~loss.isinf()) & (~loss.isnan())
        if valid.any():
            return loss[valid].mean()
        return loss.mean() * 0.0

    # ===================================================================
    # Loss Computation
    # ===================================================================
    def losses(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        features: Dict[str, torch.Tensor],
        gt_instances: List[Instances],
        Ks: List[torch.Tensor],
        im_scales_ratio: List[float],
        im_current_dims: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, torch.Tensor]:

        locations, strides_per_point = self.compute_locations(features)
        points_all = torch.cat(locations, dim=0)          # (P, 2)
        strides_all = torch.cat(strides_per_point, dim=0)  # (P,)

        def flatten_level(t: List[torch.Tensor]) -> torch.Tensor:
            N, C = t[0].shape[0], t[0].shape[1]
            return torch.cat([x.permute(0, 2, 3, 1).reshape(N, -1, C) for x in t], dim=1)

        cls_logits = flatten_level(outputs["cls_logits"])      # (N, P, C)
        box2d_reg = flatten_level(outputs["box2d_reg"])        # (N, P, 4)
        cube_2d_deltas_flat = flatten_level(outputs["cube_2d_deltas"])    # (N, P, 2)
        cube_z_flat = flatten_level(outputs["cube_z"])          # (N, P, cluster_bins)
        cube_dims_flat = flatten_level(outputs["cube_dims"])    # (N, P, 3)
        cube_pose_flat = flatten_level(outputs["cube_pose"])    # (N, P, pose_ch)
        cube_uncert_flat = flatten_level(outputs["cube_uncert"]) if self.use_conf else None

        N = cls_logits.shape[0]
        device = cls_logits.device

        all_labels, all_gt_inds = [], []

        gt_boxes3D_valid_list = []
        gt_poses_valid_list = []
        gt_box2d_valid_list = []

        for i in range(N):
            gts_i = gt_instances[i]
            valid = gts_i.gt_classes >= 0

            gt_boxes_i = gts_i.gt_boxes.tensor[valid]
            gt_labels_i = gts_i.gt_classes[valid]
            gt_boxes_ign_i = gts_i.gt_boxes.tensor[~valid] if (~valid).any() else None

            gt_boxes3D_valid_list.append(gts_i.gt_boxes3D[valid])
            gt_poses_valid_list.append(gts_i.gt_poses[valid])
            gt_box2d_valid_list.append(gts_i.gt_boxes.tensor[valid])

            with torch.no_grad():
                box_preds_i = self._decode_box2d(points_all, box2d_reg[i].detach())

            labels_i, gt_inds_i = self.assigner.assign(
                cls_logits[i].detach(), box_preds_i, points_all, strides_all,
                gt_boxes_i, gt_labels_i, gt_boxes_ign_i,
            )
            all_labels.append(labels_i)
            all_gt_inds.append(gt_inds_i)

        gt_labels = torch.stack(all_labels)      # (N, P)
        gt_gt_inds = torch.stack(all_gt_inds)    # (N, P)

        valid_mask = gt_labels >= 0
        fg_mask = (gt_labels < self.num_classes) & valid_mask

        num_fg = max(fg_mask.sum().item(), 1)

        storage = get_event_storage()
        storage.put_scalar("dense_cube/num_fg", num_fg / N)

        # -----------------------------------------------------------
        # Focal loss
        # -----------------------------------------------------------
        cls_target = torch.zeros_like(cls_logits)
        pos_labels = gt_labels[fg_mask]
        cls_target[fg_mask] = F.one_hot(pos_labels, self.num_classes).float()

        loss_cls = sigmoid_focal_loss_jit(
            cls_logits[valid_mask],
            cls_target[valid_mask],
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / num_fg

        losses = {"DenseCube/loss_cls": loss_cls * self.loss_w_cls}

        if fg_mask.sum() == 0:
            return losses

        img_idx, pt_idx = fg_mask.nonzero(as_tuple=True)
        gt_idx = gt_gt_inds[img_idx, pt_idx]
        box_classes = gt_labels[img_idx, pt_idx]

        pts_fg = points_all[pt_idx]
        box2d_reg_fg = box2d_reg[img_idx, pt_idx]
        cube_2d_deltas_fg = cube_2d_deltas_flat[img_idx, pt_idx]      # (n_fg, 2)
        cube_dims_fg = cube_dims_flat[img_idx, pt_idx]            # (n_fg, 3)
        cube_pose_fg = cube_pose_flat[img_idx, pt_idx]            # (n_fg, pose_dim)
        cube_z_fg = cube_z_flat[img_idx, pt_idx]                  # (n_fg, cluster_bins)
        if self.use_conf:
            cube_uncert_fg = cube_uncert_flat[img_idx, pt_idx].squeeze(-1)

        gt_counts = torch.tensor(
            [v.shape[0] for v in gt_boxes3D_valid_list], device=device, dtype=torch.long
        )
        gt_offsets = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long), gt_counts.cumsum(0)[:-1]
        ])
        global_gt_idx = gt_offsets[img_idx] + gt_idx

        gt_boxes3D_cat = torch.cat(gt_boxes3D_valid_list, dim=0)
        gt_poses_cat = torch.cat(gt_poses_valid_list, dim=0)
        gt_box2d_cat = torch.cat(gt_box2d_valid_list, dim=0)

        gt_boxes3D_fg = gt_boxes3D_cat[global_gt_idx]
        gt_poses_fg = gt_poses_cat[global_gt_idx]
        gt_box2d_fg = gt_box2d_cat[global_gt_idx]

        Ks_fg = torch.stack([Ks[b] / im_scales_ratio[b] for b in img_idx.tolist()]).to(device)
        Ks_fg[:, -1, -1] = 1

        # -----------------------------------------------------------
        # GIoU loss
        # -----------------------------------------------------------
        pred_box2d = self._decode_box2d(pts_fg, box2d_reg_fg)
        loss_box2d = giou_loss(pred_box2d, gt_box2d_fg, reduction="sum") / num_fg
        losses["DenseCube/loss_box2d"] = loss_box2d * self.loss_w_box2d

        # -----------------------------------------------------------
        # Decode 3D box
        # -----------------------------------------------------------
        with torch.no_grad():
            src_box_fg = self._decode_box2d(pts_fg, box2d_reg_fg.detach())
            src_widths = (src_box_fg[:, 2] - src_box_fg[:, 0]).clamp(min=1.0)
            src_heights = (src_box_fg[:, 3] - src_box_fg[:, 1]).clamp(min=1.0)
            src_ctr_x = (src_box_fg[:, 0] + src_box_fg[:, 2]) * 0.5
            src_ctr_y = (src_box_fg[:, 1] + src_box_fg[:, 3]) * 0.5
            src_scales = (src_widths ** 2 + src_heights ** 2).sqrt()

        focal_lengths_fg = Ks_fg[:, 1, 1]
        if im_current_dims is not None:
            im_scales_fg = torch.tensor(
                [im_current_dims[b][0] for b in img_idx.tolist()], device=device, dtype=torch.float32
            )
        else:
            im_scales_fg = torch.ones(num_fg, device=device)
        im_ratios_fg = torch.tensor(
            [im_scales_ratio[b] for b in img_idx.tolist()], device=device, dtype=torch.float32
        )
        im_scales_original_fg = im_scales_fg * im_ratios_fg

        if self.virtual_depth:
            virtual_to_real = util.compute_virtual_scale_from_focal_spaces(
                focal_lengths_fg, im_scales_original_fg, self.virtual_focal, im_scales_fg
            )
            real_to_virtual = 1.0 / virtual_to_real
        else:
            real_to_virtual = virtual_to_real = 1.0

        # Decode 2D center
        cube_x = src_ctr_x + src_widths * cube_2d_deltas_fg[:, 0]
        cube_y = src_ctr_y + src_heights * cube_2d_deltas_fg[:, 1]
        cube_xy = torch.stack([cube_x, cube_y], dim=1)

        # Decode dimensions
        cube_dims_norm = cube_dims_fg
        if self.dims_priors_enabled:
            prior_dims = self.priors_dims_per_cat[0, box_classes]
            prior_dims_mean = prior_dims[:, 0, :]
            prior_dims_std = prior_dims[:, 1, :]
            if self.dims_priors_func == 'sigmoid':
                prior_dims_min = (prior_dims_mean - 3 * prior_dims_std).clamp(min=0.0)
                prior_dims_max = prior_dims_mean + 3 * prior_dims_std
                cube_dims = util.scaled_sigmoid(cube_dims_norm, min=prior_dims_min, max=prior_dims_max)
            else:  # 'exp'
                cube_dims = torch.exp(cube_dims_norm.clamp(min=-5, max=5)) * prior_dims_mean
        else:
            cube_dims = torch.exp(cube_dims_norm.clamp(min=-5, max=5))

        # Decode Rotation Pose
        if self.pose_type == '6d':
            cube_pose = rotation_6d_to_matrix(cube_pose_fg)
        elif self.pose_type == 'quaternion':
            quats = cube_pose_fg
            quats = quats / _copysign(torch.sqrt((quats * quats).sum(1)), quats[:, 0])[:, None]
            cube_pose = quaternion_to_matrix(quats)
        else:  # 'euler'
            cube_pose = euler_angles_to_matrix(cube_pose_fg, 'XYZ')

        if self.allocentric_pose:
            cube_pose_allocentric = cube_pose
            cube_pose = util.R_from_allocentric(Ks_fg, cube_pose, u=cube_x.detach(), v=cube_y.detach())

        # Decode depth-z
        cube_z_raw_fg = cube_z_fg
        if self.cluster_bins > 1:
            scales_diff = (
                self.priors_z_scales.detach().T.unsqueeze(0) - src_scales.unsqueeze(1).unsqueeze(2)
            ).abs()
            assignments = scales_diff.argmin(1)
            fg_arange = torch.arange(num_fg, device=device)
            cube_z_sel = cube_z_raw_fg[fg_arange, assignments[fg_arange, box_classes]]
        else:
            cube_z_sel = cube_z_raw_fg[:, 0]

        if self.z_type == 'direct':
            cube_z_norm = cube_z_sel
            cube_z = cube_z_sel
        elif self.z_type == 'sigmoid':
            cube_z_norm = torch.sigmoid(cube_z_sel)
            cube_z = cube_z_norm * 100
        elif self.z_type == 'log':
            cube_z_norm = cube_z_sel
            cube_z = torch.exp(cube_z_sel.clamp(min=-5, max=8))
        elif self.z_type == 'clusters':
            fg_arange = torch.arange(num_fg, device=device)
            z_means = self.priors_z_stats[box_classes, assignments[fg_arange, box_classes], 0].detach()
            z_stds = self.priors_z_stats[box_classes, assignments[fg_arange, box_classes], 1].detach()
            z_mins = (z_means - 3 * z_stds).clamp(min=0)
            z_maxs = z_means + 3 * z_stds
            cube_z_norm = cube_z_sel
            cube_z = util.scaled_sigmoid(cube_z_sel, min=z_mins, max=z_maxs)
        else:
            raise ValueError(f'Unsupport z_type: {self.z_type}')

        if self.virtual_depth:
            cube_z = cube_z * virtual_to_real

        # -----------------------------------------------------------
        # Ground Truth 3D
        # -----------------------------------------------------------
        gt_2d = gt_boxes3D_fg[:, :2]
        gt_z = gt_boxes3D_fg[:, 2]
        gt_dims = gt_boxes3D_fg[:, 3:6]

        gt_x3d = gt_z * (gt_2d[:, 0] - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
        gt_y3d = gt_z * (gt_2d[:, 1] - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
        gt_3d = torch.stack((gt_x3d, gt_y3d, gt_z), dim=1)
        gt_box3d = torch.cat((gt_3d, gt_dims), dim=1)
        gt_corners = util.get_cuboid_verts_faces(gt_box3d, gt_poses_fg)[0]

        # -----------------------------------------------------------
        # 3D loss
        # -----------------------------------------------------------
        if self.disentangled_loss:
            cube_dis_x3d_from_z = cube_z * (gt_2d[:, 0] - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            cube_dis_y3d_from_z = cube_z * (gt_2d[:, 1] - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            dis_z_box = torch.cat(
                (torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z), dim=1), gt_dims), dim=1
            )
            dis_z_corners = util.get_cuboid_verts_faces(dis_z_box, gt_poses_fg)[0]
            loss_z = self.l1_loss(dis_z_corners, gt_corners).reshape(num_fg, -1).mean(1)

            cube_dis_x3d = gt_z * (cube_x - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            cube_dis_y3d = gt_z * (cube_y - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            dis_xy_box = torch.cat(
                (torch.stack((cube_dis_x3d, cube_dis_y3d, gt_z), dim=1), gt_dims), dim=1
            )
            dis_xy_corners = util.get_cuboid_verts_faces(dis_xy_box, gt_poses_fg)[0]
            loss_xy = self.l1_loss(dis_xy_corners, gt_corners).reshape(num_fg, -1).mean(1)

            dis_dims_corners = util.get_cuboid_verts_faces(torch.cat((gt_3d, cube_dims), dim=1), gt_poses_fg)[0]
            loss_dims = self.l1_loss(dis_dims_corners, gt_corners).reshape(num_fg, -1).mean(1)

            dis_pose_corners = util.get_cuboid_verts_faces(gt_box3d, cube_pose)[0]
            if self.chamfer_pose:
                loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)
            else:
                loss_pose = self.l1_loss(dis_pose_corners, gt_corners).reshape(num_fg, -1).mean(1)
        else:
            gt_deltas_x = (gt_2d[:, 0] - src_ctr_x) / src_widths
            gt_deltas_y = (gt_2d[:, 1] - src_ctr_y) / src_heights
            gt_deltas = torch.stack([gt_deltas_x, gt_deltas_y], dim=1)
            loss_xy = self.l1_loss(cube_2d_deltas_fg, gt_deltas).mean(1)

            if self.dims_priors_enabled:
                cube_dims_gt_normspace = torch.log(gt_dims / prior_dims_mean)
                loss_dims = self.l1_loss(cube_dims_norm, cube_dims_gt_normspace).mean(1)
            else:
                loss_dims = self.l1_loss(cube_dims_norm, torch.log(gt_dims)).mean(1)

            if self.allocentric_pose:
                gt_poses_allocentric = util.R_to_allocentric(Ks_fg, gt_poses_fg, u=cube_x.detach(), v=cube_y.detach())
                raw_pose = 1 - so3_relative_angle(cube_pose_allocentric, gt_poses_allocentric, eps=0.1, cos_angle=True)
            else:
                raw_pose = 1 - so3_relative_angle(cube_pose, gt_poses_fg, eps=0.1, cos_angle=True)

            valid_pose_mask = (~raw_pose.isnan()) & (~raw_pose.isinf())
            loss_pose = torch.where(valid_pose_mask, raw_pose, torch.zeros_like(raw_pose))

            if self.z_type == 'direct':
                loss_z = self.l1_loss(cube_z, gt_z)
            elif self.z_type == 'sigmoid':
                loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual / 100).clamp(0, 1))
            elif self.z_type == 'log':
                loss_z = self.l1_loss(cube_z_norm, torch.log((gt_z * real_to_virtual).clamp(min=0.01)))
            elif self.z_type == 'clusters':
                loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual - z_means) / z_stds)

        # -----------------------------------------------------------
        # Joint (Entangled) loss
        # -----------------------------------------------------------
        loss_joint = None
        if self.loss_w_joint > 0:
            pred_x3d_from_z = cube_z * (cube_x - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            pred_y3d_from_z = cube_z * (cube_y - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            pred_3d_centers = torch.stack((pred_x3d_from_z, pred_y3d_from_z, cube_z), dim=1)
            pred_box3d = torch.cat((pred_3d_centers, cube_dims), dim=1)
            dis_joint_corners = util.get_cuboid_verts_faces(pred_box3d, cube_pose)[0]

            if self.chamfer_pose and self.disentangled_loss:
                loss_joint = self.chamfer_loss(dis_joint_corners, gt_corners)
            else:
                loss_joint = self.l1_loss(dis_joint_corners, gt_corners).reshape(num_fg, -1).mean(1)

        total_3d_loss_raw = (
                        loss_dims * self.loss_w_dims + loss_xy * self.loss_w_xy + loss_z * self.loss_w_z
                    )
        if loss_pose is not None:
            total_3d_loss_raw = total_3d_loss_raw + loss_pose * self.loss_w_pose
        if loss_joint is not None:
            total_3d_loss_raw = total_3d_loss_raw + loss_joint * self.loss_w_joint
        storage.put_scalar(
            "DenseCube/total_3D_loss",
            self.loss_w_3d * self._safely_reduce(total_3d_loss_raw.detach()).item(),
            smoothing_hint=False,
        )

        if self.inverse_z_weight:
            inverse_z_w = 1.0 / torch.log(gt_z.clamp(min=E_CONSTANT))
            loss_dims = loss_dims * inverse_z_w
            loss_xy = loss_xy * inverse_z_w
            loss_z = loss_z * inverse_z_w
            if loss_pose is not None:
                loss_pose = loss_pose * inverse_z_w
            if loss_joint is not None:
                loss_joint = loss_joint * inverse_z_w

        if self.use_conf:
            uncert_sf = SQRT_2_CONSTANT * torch.exp(-cube_uncert_fg)
            loss_dims = loss_dims * uncert_sf
            loss_xy = loss_xy * uncert_sf
            loss_z = loss_z * uncert_sf
            if loss_pose is not None:
                loss_pose = loss_pose * uncert_sf
            if loss_joint is not None:
                loss_joint = loss_joint * uncert_sf

            losses["DenseCube/loss_uncert"] = self._safely_reduce(cube_uncert_fg.clone())
            storage.put_scalar("DenseCube/conf", torch.exp(-cube_uncert_fg).mean().item())

        losses["DenseCube/loss_xy"] = self._safely_reduce(loss_xy) * self.loss_w_xy * self.loss_w_3d
        losses["DenseCube/loss_dims"] = self._safely_reduce(loss_dims) * self.loss_w_dims * self.loss_w_3d
        losses["DenseCube/loss_z"] = self._safely_reduce(loss_z) * self.loss_w_z * self.loss_w_3d
        if loss_pose is not None:
            losses["DenseCube/loss_pose"] = self._safely_reduce(loss_pose) * self.loss_w_pose * self.loss_w_3d
        if loss_joint is not None:
            valid_joint = (~loss_joint.isinf()) & (~loss_joint.isnan())
            if valid_joint.any():
                losses["DenseCube/loss_joint"] = self._safely_reduce(loss_joint[valid_joint]) * self.loss_w_joint * self.loss_w_3d

        z_error = (cube_z - gt_z).detach().abs()
        dims_error = (cube_dims - gt_dims).detach().abs()
        xy_error = (cube_xy - gt_2d).detach().abs()
        storage.put_scalar("DenseCube/z_error", z_error.mean().item(), smoothing_hint=False)
        storage.put_scalar("DenseCube/dims_error", dims_error.mean().item(), smoothing_hint=False)
        storage.put_scalar("DenseCube/xy_error", xy_error.mean().item(), smoothing_hint=False)
        storage.put_scalar("DenseCube/z_close", (z_error < 0.20).float().mean().item(), smoothing_hint=False)

        return losses

    # ===================================================================
    # Inference
    # ===================================================================
    @torch.no_grad()
    def inference(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        features: Dict[str, torch.Tensor],
        Ks: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        im_scales_ratio: List[float],
        im_current_dims: Optional[List[Tuple[float, float]]] = None
    ) -> List[Instances]:

        locations, strides_per_point = self.compute_locations(features)
        num_levels = len(locations)
        results = []
        N = outputs["cls_logits"][0].shape[0]
        device = features[self.in_features[0]].device

        for img_idx in range(N):
            boxes_all, scores_all, classes_all = [], [], []
            deltas_all, dims_all, pose_all, z_all, uncert_all = [], [], [], [], []

            for lvl in range(num_levels):
                # 1. 僅對分類特徵進行 permute 排序，找出 Top-K 索引
                cls_l = outputs["cls_logits"][lvl][img_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                scores_l = cls_l.sigmoid()
                
                topk = min(self.test_topk_candidates, scores_l.numel())
                flat_scores, flat_idx = scores_l.reshape(-1).topk(topk)
                
                keep_mask = flat_scores > self.test_score_thresh
                flat_scores = flat_scores[keep_mask]
                flat_idx = flat_idx[keep_mask]

                pt_idx = torch.div(flat_idx, self.num_classes, rounding_mode='floor')
                cls_idx = flat_idx % self.num_classes

                if len(pt_idx) == 0:
                    continue

                # 2. 延遲提取 (Late Extraction)：僅對有保留的空間位置 (pt_idx) 抽取 3D 特徵
                pts_l = locations[lvl][pt_idx]
                
                box_l_sel = outputs["box2d_reg"][lvl][img_idx].view(4, -1)[:, pt_idx].T
                boxes_dec = self._decode_box2d(pts_l, box_l_sel)

                boxes_all.append(boxes_dec)
                scores_all.append(flat_scores)
                classes_all.append(cls_idx)
                
                # 直接以 view 展開空間維度並做索引，完全避免昂貴的 permute 與全圖記憶體搬移
                deltas_all.append(outputs["cube_2d_deltas"][lvl][img_idx].view(2, -1)[:, pt_idx].T)
                dims_all.append(outputs["cube_dims"][lvl][img_idx].view(3, -1)[:, pt_idx].T)
                pose_dim = outputs["cube_pose"][lvl][img_idx].shape[0]
                pose_all.append(outputs["cube_pose"][lvl][img_idx].view(pose_dim, -1)[:, pt_idx].T)
                z_all.append(outputs["cube_z"][lvl][img_idx].view(self.cluster_bins, -1)[:, pt_idx].T)
                
                if self.use_conf:
                    uncert_all.append(outputs["cube_uncert"][lvl][img_idx].view(1, -1)[:, pt_idx].T.squeeze(-1))

            h, w = image_sizes[img_idx]

            if len(boxes_all) == 0 or torch.cat(boxes_all).shape[0] == 0:
                empty = Instances((h, w))
                empty.pred_boxes = Boxes(torch.zeros((0, 4), device=device))
                empty.scores = torch.zeros(0, device=device)
                empty.pred_classes = torch.zeros(0, dtype=torch.int64, device=device)
                empty.pred_center_cam = torch.zeros((0, 3), device=device)
                empty.pred_center_2D = torch.zeros((0, 2), device=device)
                empty.pred_dimensions = torch.zeros((0, 3), device=device)
                empty.pred_pose = torch.zeros((0, 3, 3), device=device)
                empty.pred_bbox3D = torch.zeros((0, 8, 3), device=device)
                if self.use_conf:
                    empty.pred_uncertainty = torch.zeros(0, device=device)
                results.append(empty)
                continue

            boxes_all = torch.cat(boxes_all)
            scores_all = torch.cat(scores_all)
            classes_all = torch.cat(classes_all)

            keep = batched_nms(boxes_all, scores_all, classes_all, self.test_nms_thresh)
            keep = keep[: self.test_max_detections]

            boxes_k = boxes_all[keep]
            deltas_k = torch.cat(deltas_all)[keep]
            dims_k = torch.cat(dims_all)[keep]
            pose_k = torch.cat(pose_all)[keep]
            z_k = torch.cat(z_all)[keep]
            scores_k = scores_all[keep]
            classes_k = classes_all[keep]
            if self.use_conf:
                uncert_k = torch.cat(uncert_all)[keep]

            n_k = boxes_k.shape[0]

            K = (Ks[img_idx] / im_scales_ratio[img_idx]).to(device)
            K[-1, -1] = 1
            Ks_k = K.unsqueeze(0).repeat(n_k, 1, 1)

            src_widths = (boxes_k[:, 2] - boxes_k[:, 0]).clamp(min=1.0)
            src_heights = (boxes_k[:, 3] - boxes_k[:, 1]).clamp(min=1.0)
            src_ctr_x = (boxes_k[:, 0] + boxes_k[:, 2]) * 0.5
            src_ctr_y = (boxes_k[:, 1] + boxes_k[:, 3]) * 0.5
            src_scales = (src_widths ** 2 + src_heights ** 2).sqrt()

            focal_lengths_k = Ks_k[:, 1, 1]
            
            # 3. 避免迴圈內引發 CPU 與 GPU 同步等待，將純量變數一次性轉為 Device Tensor
            im_scale_val = float(im_current_dims[img_idx][0]) if im_current_dims is not None else 1.0
            im_scale_t = torch.full((n_k,), im_scale_val, device=device, dtype=torch.float32)
            im_scales_orig_t = torch.full((n_k,), im_scale_val * im_scales_ratio[img_idx], device=device, dtype=torch.float32)

            if self.virtual_depth:
                virtual_to_real = util.compute_virtual_scale_from_focal_spaces(
                    focal_lengths_k,
                    im_scales_orig_t,
                    self.virtual_focal,
                    im_scale_t,
                )
            else:
                virtual_to_real = 1.0

            cube_x = src_ctr_x + src_widths * deltas_k[:, 0]
            cube_y = src_ctr_y + src_heights * deltas_k[:, 1]

            if self.dims_priors_enabled:
                prior_dims = self.priors_dims_per_cat[0, classes_k]
                prior_dims_mean = prior_dims[:, 0, :]
                if self.dims_priors_func == 'sigmoid':
                    prior_dims_std = prior_dims[:, 1, :]
                    prior_dims_min = (prior_dims_mean - 3 * prior_dims_std).clamp(min=0.0)
                    prior_dims_max = prior_dims_mean + 3 * prior_dims_std
                    cube_dims = util.scaled_sigmoid(dims_k, min=prior_dims_min, max=prior_dims_max)
                else:
                    cube_dims = torch.exp(dims_k.clamp(min=-5, max=5)) * prior_dims_mean
            else:
                cube_dims = torch.exp(dims_k.clamp(min=-5, max=5))

            if self.pose_type == '6d':
                cube_pose = rotation_6d_to_matrix(pose_k)
            elif self.pose_type == 'quaternion':
                quats = pose_k
                quats = quats / _copysign(torch.sqrt((quats * quats).sum(1)), quats[:, 0])[:, None]
                cube_pose = quaternion_to_matrix(quats)
            else:
                cube_pose = euler_angles_to_matrix(pose_k, 'XYZ')

            if self.allocentric_pose:
                cube_pose = util.R_from_allocentric(Ks_k, cube_pose, u=cube_x, v=cube_y)

            if self.cluster_bins > 1:
                scales_diff = (self.priors_z_scales.T.unsqueeze(0) - src_scales.unsqueeze(1).unsqueeze(2)).abs()
                assignments = scales_diff.argmin(1)
                k_arange = torch.arange(n_k, device=device)
                cube_z_sel = z_k[k_arange, assignments[k_arange, classes_k]]
            else:
                cube_z_sel = z_k[:, 0]

            if self.z_type == 'direct':
                cube_z = cube_z_sel
            elif self.z_type == 'sigmoid':
                cube_z = torch.sigmoid(cube_z_sel) * 100
            elif self.z_type == 'log':
                cube_z = torch.exp(cube_z_sel.clamp(min=-5, max=8))
            elif self.z_type == 'clusters':
                k_arange = torch.arange(n_k, device=device)
                z_means = self.priors_z_stats[classes_k, assignments[k_arange, classes_k], 0]
                z_stds = self.priors_z_stats[classes_k, assignments[k_arange, classes_k], 1]
                z_mins = (z_means - 3 * z_stds).clamp(min=0)
                z_maxs = z_means + 3 * z_stds
                cube_z = util.scaled_sigmoid(cube_z_sel, min=z_mins, max=z_maxs)

            if self.virtual_depth:
                cube_z = cube_z * virtual_to_real

            cube_x3d = cube_z * (cube_x - K[0, 2]) / K[0, 0]
            cube_y3d = cube_z * (cube_y - K[1, 2]) / K[1, 1]

            if self.use_conf:
                cube_conf = torch.exp(-uncert_k)
                final_scores = (scores_k * cube_conf).clamp(min=0).sqrt()
            else:
                final_scores = scores_k

            pred_boxes_out = boxes_k.clone()
            pred_boxes_out[:, 0::2] = pred_boxes_out[:, 0::2].clamp(0, w)
            pred_boxes_out[:, 1::2] = pred_boxes_out[:, 1::2].clamp(0, h)

            result = Instances((h, w))
            result.pred_boxes = Boxes(pred_boxes_out)
            result.scores = final_scores
            result.pred_classes = classes_k
            result.pred_center_cam = torch.stack([cube_x3d, cube_y3d, cube_z], dim=1)
            result.pred_center_2D = torch.stack([cube_x, cube_y], dim=1) * im_scales_ratio[img_idx]
            result.pred_dimensions = cube_dims
            result.pred_pose = cube_pose
            result.pred_bbox3D = util.get_cuboid_verts_faces(
                torch.cat([result.pred_center_cam, cube_dims], dim=1), cube_pose
            )[0]
            if self.use_conf:
                result.pred_uncertainty = cube_conf  

            results.append(result)

        return results


def build_dense_cube_head(cfg, input_shape: Dict[str, ShapeSpec], priors: Optional[dict] = None):
    name = cfg.MODEL.DENSE_HEAD.NAME
    return DENSE_CUBE_HEAD_REGISTRY.get(name)(cfg, input_shape, priors=priors)
