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
from detectron2.modeling.poolers import ROIPooler
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)
from pytorch3d.transforms.so3 import so3_relative_angle

from cubercnn.modeling.dense_head.assigner import build_assigner, SimOTAAssigner, build_tal_assigner, TaskAlignedAssigner
from cubercnn.modeling.roi_heads.cube_head import build_cube_head
from cubercnn import util

logger = logging.getLogger(__name__)

E_CONSTANT = 2.71828183
SQRT_2_CONSTANT = 1.41421356

CASCADE_DENSE_CUBE_HEAD_REGISTRY = Registry("CASCADE_DENSE_CUBE_HEAD")


def quality_focal_loss(pred_logits: torch.Tensor, target_scores: torch.Tensor, beta: float = 2.0, reduction: str = "sum"):
    pred_sigmoid = pred_logits.sigmoid()
    
    # |y - sigma|^beta
    scale_factor = (pred_sigmoid - target_scores).abs().pow(beta)
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
    loss = scale_factor * bce_loss
    
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@CASCADE_DENSE_CUBE_HEAD_REGISTRY.register()
class CascadeDenseCubeHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        in_features: List[str],
        fpn_strides: List[int],
        num_convs: int,
        conv_dim: int,
        prior_prob: float,
        focal_alpha: float,
        focal_gamma: float,
        assigner: TaskAlignedAssigner,  
        cube_head: nn.Module,
        cube_pooler: nn.Module,
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
        use_conf: bool,
        cluster_bins: int,
        max_proposals: int,
        gt_in_proposals: bool,
        priors: Optional[dict] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_features = in_features
        self.fpn_strides = fpn_strides
        self.num_levels = len(in_features)
        self.assigner = assigner

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_w_cls = loss_w_cls
        self.loss_w_box2d = loss_w_box2d

        self.loss_w_3d = loss_w_3d
        self.loss_w_xy = loss_w_xy
        self.loss_w_z = loss_w_z
        self.loss_w_dims = loss_w_dims
        self.loss_w_pose = loss_w_pose
        self.loss_w_joint = loss_w_joint
        self.z_type = z_type
        self.dims_priors_enabled = dims_priors_enabled
        self.dims_priors_func = dims_priors_func
        self.chamfer_pose = chamfer_pose
        self.disentangled_loss = disentangled_loss
        self.inverse_z_weight = inverse_z_weight
        self.allocentric_pose = allocentric_pose
        self.virtual_depth = virtual_depth
        self.virtual_focal = virtual_focal
        self.use_conf = use_conf
        self.cluster_bins = max(cluster_bins, 1)
        self.gt_in_proposals = gt_in_proposals
        self.max_proposals = max_proposals

        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.test_max_detections = test_max_detections

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

        # -------------------------------------------------------------
        # 2D branch
        # -------------------------------------------------------------
        # self.cls_towers = nn.ModuleList()
        # self.box2d_towers = nn.ModuleList()
        # self.cls_preds = nn.ModuleList()
        # self.box2d_preds = nn.ModuleList()

        # for _ in range(self.num_levels):
        #     self.cls_towers.append(self._make_tower(conv_dim, num_convs))
        #     self.cls_preds.append(nn.Conv2d(conv_dim, num_classes, 1))
        #     self.box2d_towers.append(self._make_tower(conv_dim, num_convs))
        #     self.box2d_preds.append(nn.Conv2d(conv_dim, 4, 1))

        # Shared Conv tower
        self.cls_tower = self._make_tower(conv_dim, num_convs)
        self.box2d_tower = self._make_tower(conv_dim, num_convs)
        
        self.cls_preds = nn.ModuleList()
        self.box2d_preds = nn.ModuleList()
        for _ in range(self.num_levels):
            self.cls_preds.append(nn.Conv2d(conv_dim, num_classes, 1))
            self.box2d_preds.append(nn.Conv2d(conv_dim, 4, 1))

        self._init_weights(prior_prob)

        # -------------------------------------------------------------
        # ROI-based 3D branch 
        # -------------------------------------------------------------
        self.cube_head = cube_head
        self.cube_pooler = cube_pooler

    def _make_tower(self, conv_dim: int, num_convs: int) -> nn.Sequential:
        layers = [ConvBlock(conv_dim, conv_dim, kernel_size=3, padding=1) for _ in range(num_convs)]
        return nn.Sequential(*layers)

    def _init_weights(self, prior_prob: float):
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_pred in self.cls_preds:
            nn.init.constant_(cls_pred.bias, bias_value)
            nn.init.normal_(cls_pred.weight, std=0.01)
        for box_pred in self.box2d_preds:
            nn.init.normal_(box_pred.weight, std=0.01)
            nn.init.constant_(box_pred.bias, 0.0)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], priors: Optional[dict] = None):
        in_features = cfg.MODEL.DENSE_HEAD.IN_FEATURES

        pooler_resolution = cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[f].stride for f in in_features)
        cube_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE,
        )
        cube_head_shape = ShapeSpec(
            channels=cfg.MODEL.DENSE_HEAD.CONV_DIM,
            width=pooler_resolution,
            height=pooler_resolution,
        )
        cube_head = build_cube_head(cfg, cube_head_shape)

        return {
            "num_classes": cfg.MODEL.DENSE_HEAD.NUM_CLASSES,
            "in_features": in_features,
            "fpn_strides": [input_shape[f].stride for f in in_features],
            "num_convs": cfg.MODEL.DENSE_HEAD.NUM_CONVS,
            "conv_dim": cfg.MODEL.DENSE_HEAD.CONV_DIM,
            "prior_prob": cfg.MODEL.DENSE_HEAD.PRIOR_PROB,
            "focal_alpha": cfg.MODEL.DENSE_HEAD.FOCAL_ALPHA,
            "focal_gamma": cfg.MODEL.DENSE_HEAD.FOCAL_GAMMA,
            "assigner": build_assigner(cfg),
            # "assigner": build_tal_assigner(cfg),
            "cube_head": cube_head,
            "cube_pooler": cube_pooler,
            "loss_w_cls": cfg.MODEL.DENSE_HEAD.LOSS_W_CLS,
            "loss_w_box2d": cfg.MODEL.DENSE_HEAD.LOSS_W_BOX2D,
            "test_score_thresh": cfg.MODEL.DENSE_HEAD.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.DENSE_HEAD.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.DENSE_HEAD.NMS_THRESH_TEST,
            "test_max_detections": cfg.MODEL.DENSE_HEAD.MAX_DETECTIONS_PER_IMAGE,
            "gt_in_proposals": cfg.MODEL.DENSE_HEAD.GT_IN_PROPOSALS,
            "loss_w_3d": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D,
            "loss_w_xy": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY,
            "loss_w_z": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z,
            "loss_w_dims": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS,
            "loss_w_pose": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE,
            "loss_w_joint": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT,
            "z_type": cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE,
            "dims_priors_enabled": cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED,
            "dims_priors_func": cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC,
            "chamfer_pose": cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE,
            "disentangled_loss": cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS,
            "inverse_z_weight": cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT,
            "allocentric_pose": cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE,
            "virtual_depth": cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH,
            "virtual_focal": cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL,
            "use_conf": cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE,
            "cluster_bins": cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS,
            "max_proposals": cfg.MODEL.ROI_CUBE_HEAD.MAX_PROPOSALS,
            "priors": priors,
        }

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        # cls_logits, box2d_reg = [], []
        # for level, f in enumerate(self.in_features):
        #     x = features[f]
        #     stride = self.fpn_strides[level]
        #     cls_feat = self.cls_towers[level](x)
        #     box_feat = self.box2d_towers[level](x)
        #     cls_logits.append(self.cls_preds[level](cls_feat))
        #     box2d_reg.append(F.relu(self.box2d_preds[level](box_feat)) * stride)
        # return {"cls_logits": cls_logits, "box2d_reg": box2d_reg}

        # Shared conv tower
        cls_logits, box2d_reg = [], []
        for level, f in enumerate(self.in_features):
            x = features[f]
            stride = self.fpn_strides[level]
            cls_feat = self.cls_tower(x)
            box_feat = self.box2d_tower(x)
            cls_logits.append(self.cls_preds[level](cls_feat))
            box2d_reg.append(F.relu(self.box2d_preds[level](box_feat)) * stride)
        return {"cls_logits": cls_logits, "box2d_reg": box2d_reg}

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

    def _augment_proposals_with_gt(
        self,
        proposal_boxes_per_image: List[torch.Tensor],
        box_classes_per_image: List[torch.Tensor],
        gt_boxes3D_per_image: List[torch.Tensor],
        gt_poses_per_image: List[torch.Tensor],
        gt_instances: List[Instances],
    ):
        merged_boxes, merged_classes, merged_boxes3D, merged_poses = [], [], [], []

        for b in range(len(gt_instances)):
            gi = gt_instances[b]
            valid = gi.gt_classes >= 0

            merged_boxes.append(torch.cat([proposal_boxes_per_image[b], gi.gt_boxes.tensor[valid]], dim=0))
            merged_classes.append(torch.cat([box_classes_per_image[b], gi.gt_classes[valid]], dim=0))
            merged_boxes3D.append(torch.cat([gt_boxes3D_per_image[b], gi.gt_boxes3D[valid]], dim=0))
            merged_poses.append(torch.cat([gt_poses_per_image[b], gi.gt_poses[valid]], dim=0))

        return merged_boxes, merged_classes, merged_boxes3D, merged_poses

    # ===================================================================
    # 統一的 3D Forward 操作 (供 losses 與 inference 共用)
    # ===================================================================
    def _forward_cube(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
        Ks: List[torch.Tensor],
        im_scales_ratio: List[float],
        im_current_dims: Optional[List[Tuple[float, float]]] = None,
        is_training=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        if is_training==None:
            is_training = self.training
        
        num_boxes_per_image = [len(inst) for inst in instances]
        n = sum(num_boxes_per_image)
        device = features[self.in_features[0]].device

        if n == 0:
            return (instances, {}) if is_training else instances

        # 根據階段抽取屬性
        # if self.training:
        #     proposal_boxes = [x.proposal_boxes for x in instances]
        #     box_classes_cat = torch.cat([x.gt_classes for x in instances], dim=0)
        #     gt_boxes3D_fg = torch.cat([x.gt_boxes3D for x in instances], dim=0)
        #     gt_poses_fg = torch.cat([x.gt_poses for x in instances], dim=0)
        # else:
        #     proposal_boxes = [x.pred_boxes for x in instances]
        #     box_classes_cat = torch.cat([x.pred_classes for x in instances], dim=0)
        if is_training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            box_classes_cat = torch.cat([x.gt_classes for x in instances], dim=0)
            gt_boxes3D_fg = torch.cat([x.gt_boxes3D for x in instances], dim=0)
            gt_poses_fg = torch.cat([x.gt_poses for x in instances], dim=0)
        else:
            proposal_boxes = [x.pred_boxes for x in instances]
            box_classes_cat = torch.cat([x.pred_classes for x in instances], dim=0)

        # ---- ROI 特徵提取與 CubeHead 前向傳播 ----
        roi_features = self.cube_pooler(
            [features[f] for f in self.in_features], proposal_boxes
        ).flatten(1)

        cube_2d_deltas, cube_z, cube_dims, cube_pose, cube_uncert = self.cube_head(roi_features)

        fg_inds = torch.arange(n, device=device)
        src_boxes = torch.cat([b.tensor for b in proposal_boxes], dim=0)
        src_widths = (src_boxes[:, 2] - src_boxes[:, 0]).clamp(min=1.0)
        src_heights = (src_boxes[:, 3] - src_boxes[:, 1]).clamp(min=1.0)
        src_ctr_x = (src_boxes[:, 0] + src_boxes[:, 2]) * 0.5
        src_ctr_y = (src_boxes[:, 1] + src_boxes[:, 3]) * 0.5
        src_scales = (src_widths ** 2 + src_heights ** 2).sqrt()

        # 組裝動態 Ks
        Ks_fg = torch.cat([
            (Ks[b] / im_scales_ratio[b]).unsqueeze(0).repeat(num, 1, 1)
            for b, num in enumerate(num_boxes_per_image)
        ]).to(device)
        Ks_fg[:, -1, -1] = 1

        # 深度 Cluster 處理
        if self.cluster_bins > 1:
            scales_diff = (self.priors_z_scales.detach().T.unsqueeze(0) - src_scales.unsqueeze(1).unsqueeze(2)).abs()
            assignments = scales_diff.argmin(1)
            cube_z = cube_z[fg_inds, :, box_classes_cat, :][fg_inds, assignments[fg_inds, box_classes_cat]]
        else:
            cube_z = cube_z[fg_inds, box_classes_cat, :]

        # 針對類別取得預測結果
        cube_dims = cube_dims[fg_inds, box_classes_cat, :]
        cube_pose = cube_pose[fg_inds, box_classes_cat, :, :]
        cube_2d_deltas = cube_2d_deltas[fg_inds, box_classes_cat, :]
        if self.use_conf:
            cube_uncert = cube_uncert[fg_inds, box_classes_cat]

        cube_x = src_ctr_x + src_widths * cube_2d_deltas[:, 0]
        cube_y = src_ctr_y + src_heights * cube_2d_deltas[:, 1]
        cube_xy = torch.stack([cube_x, cube_y], dim=1)

        # Dims 處理
        cube_dims_norm = cube_dims
        if self.dims_priors_enabled:
            prior_dims = self.priors_dims_per_cat.detach().repeat([n, 1, 1, 1])[fg_inds, box_classes_cat]
            prior_dims_mean = prior_dims[:, 0, :]
            prior_dims_std = prior_dims[:, 1, :]
            if self.dims_priors_func == 'sigmoid':
                prior_dims_min = (prior_dims_mean - 3 * prior_dims_std).clamp(min=0.0)
                prior_dims_max = prior_dims_mean + 3 * prior_dims_std
                cube_dims = util.scaled_sigmoid(cube_dims_norm, min=prior_dims_min, max=prior_dims_max)
            else:
                cube_dims = torch.exp(cube_dims_norm.clamp(max=5)) * prior_dims_mean
        else:
            cube_dims = torch.exp(cube_dims_norm.clamp(max=5))

        # Allocentric 視角修正
        if self.allocentric_pose:
            cube_pose_allocentric = cube_pose
            cube_pose = util.R_from_allocentric(Ks_fg, cube_pose, u=cube_x.detach(), v=cube_y.detach())

        cube_z = cube_z.squeeze(-1) if cube_z.dim() > 1 else cube_z

        # Virtual focal 解析
        focal_lengths_fg = Ks_fg[:, 1, 1]
        if im_current_dims is not None:
            im_scales_fg = torch.cat([
                torch.full((num,), float(im_current_dims[b][0]), device=device)
                for b, num in enumerate(num_boxes_per_image)
            ])
        else:
            im_scales_fg = torch.ones(n, device=device)
            
        im_ratios_fg = torch.cat([
            torch.full((num,), float(im_scales_ratio[b]), device=device)
            for b, num in enumerate(num_boxes_per_image)
        ])
        im_scales_original_fg = im_scales_fg * im_ratios_fg

        if self.virtual_depth:
            virtual_to_real = util.compute_virtual_scale_from_focal_spaces(
                focal_lengths_fg, im_scales_original_fg, self.virtual_focal, im_scales_fg
            )
            real_to_virtual = 1.0 / virtual_to_real
        else:
            real_to_virtual = virtual_to_real = 1.0

        # Z 空間轉換
        if self.z_type == 'sigmoid':
            cube_z_norm = torch.sigmoid(cube_z)
            cube_z = cube_z_norm * 100
        elif self.z_type == 'log':
            cube_z_norm = cube_z
            cube_z = torch.exp(cube_z.clamp(min=-5, max=8))
        elif self.z_type == 'clusters':
            z_means = self.priors_z_stats[box_classes_cat, assignments[fg_inds, box_classes_cat], 0].detach()
            z_stds = self.priors_z_stats[box_classes_cat, assignments[fg_inds, box_classes_cat], 1].detach()
            z_mins = (z_means - 3 * z_stds).clamp(min=0)
            z_maxs = z_means + 3 * z_stds
            cube_z_norm = cube_z
            cube_z = util.scaled_sigmoid(cube_z, min=z_mins, max=z_maxs)
        else:
            cube_z_norm = cube_z

        if self.virtual_depth:
            cube_z = cube_z * virtual_to_real

        # -------------------------------------------------------------------
        # 訓練階段：計算 3D 損失 (完全等價搬移自原始 losses 內部邏輯)
        # -------------------------------------------------------------------
        if is_training:
            losses = {}
            storage = get_event_storage()
            
            gt_2d = gt_boxes3D_fg[:, :2]
            gt_z = gt_boxes3D_fg[:, 2]
            gt_dims = gt_boxes3D_fg[:, 3:6]

            gt_x3d = gt_z * (gt_2d[:, 0] - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            gt_y3d = gt_z * (gt_2d[:, 1] - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            gt_3d = torch.stack((gt_x3d, gt_y3d, gt_z), dim=1)
            gt_box3d = torch.cat((gt_3d, gt_dims), dim=1)
            gt_corners = util.get_cuboid_verts_faces(gt_box3d, gt_poses_fg)[0]

            if self.disentangled_loss:
                cube_dis_x3d_from_z = cube_z * (gt_2d[:, 0] - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
                cube_dis_y3d_from_z = cube_z * (gt_2d[:, 1] - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
                dis_z_box = torch.cat((torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z), dim=1), gt_dims), dim=1)
                dis_z_corners = util.get_cuboid_verts_faces(dis_z_box, gt_poses_fg)[0]
                loss_z = self.l1_loss(dis_z_corners, gt_corners).reshape(n, -1).mean(1)

                cube_dis_x3d = gt_z * (cube_x - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
                cube_dis_y3d = gt_z * (cube_y - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
                dis_xy_box = torch.cat((torch.stack((cube_dis_x3d, cube_dis_y3d, gt_z), dim=1), gt_dims), dim=1)
                dis_xy_corners = util.get_cuboid_verts_faces(dis_xy_box, gt_poses_fg)[0]
                loss_xy = self.l1_loss(dis_xy_corners, gt_corners).reshape(n, -1).mean(1)

                dis_dims_corners = util.get_cuboid_verts_faces(torch.cat((gt_3d, cube_dims), dim=1), gt_poses_fg)[0]
                loss_dims = self.l1_loss(dis_dims_corners, gt_corners).reshape(n, -1).mean(1)

                dis_pose_corners = util.get_cuboid_verts_faces(gt_box3d, cube_pose)[0]
                if self.chamfer_pose:
                    loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)
                else:
                    loss_pose = self.l1_loss(dis_pose_corners, gt_corners).reshape(n, -1).mean(1)
            else:
                gt_deltas_x = (gt_2d[:, 0] - src_ctr_x) / src_widths
                gt_deltas_y = (gt_2d[:, 1] - src_ctr_y) / src_heights
                gt_deltas = torch.stack([gt_deltas_x, gt_deltas_y], dim=1)
                loss_xy = self.l1_loss(cube_2d_deltas, gt_deltas).mean(1)

                if self.dims_priors_enabled:
                    cube_dims_gt_normspace = torch.log(gt_dims / prior_dims_mean)
                    loss_dims = self.l1_loss(cube_dims_norm, cube_dims_gt_normspace).mean(1)
                else:
                    loss_dims = self.l1_loss(cube_dims_norm, torch.log(gt_dims)).mean(1)

                try:
                    if self.allocentric_pose:
                        gt_poses_allocentric = util.R_to_allocentric(Ks_fg, gt_poses_fg, u=cube_x.detach(), v=cube_y.detach())
                        loss_pose = 1 - so3_relative_angle(cube_pose_allocentric, gt_poses_allocentric, eps=0.1, cos_angle=True)
                    else:
                        loss_pose = 1 - so3_relative_angle(cube_pose, gt_poses_fg, eps=0.1, cos_angle=True)
                except Exception:
                    loss_pose = torch.zeros(n, device=device)

                if self.z_type == 'direct':
                    loss_z = self.l1_loss(cube_z, gt_z)
                elif self.z_type == 'sigmoid':
                    loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual / 100).clamp(0, 1))
                elif self.z_type == 'log':
                    loss_z = self.l1_loss(cube_z_norm, torch.log((gt_z * real_to_virtual).clamp(min=0.01)))
                elif self.z_type == 'clusters':
                    loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual - z_means) / z_stds)

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
                    loss_joint = self.l1_loss(dis_joint_corners, gt_corners).reshape(n, -1).mean(1)

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
                uncert_sf = SQRT_2_CONSTANT * torch.exp(-cube_uncert)
                loss_dims = loss_dims * uncert_sf
                loss_xy = loss_xy * uncert_sf
                loss_z = loss_z * uncert_sf
                if loss_pose is not None:
                    loss_pose = loss_pose * uncert_sf
                if loss_joint is not None:
                    loss_joint = loss_joint * uncert_sf

                losses["DenseCube/loss_uncert"] = self._safely_reduce(cube_uncert.clone())
                storage.put_scalar("DenseCube/conf", torch.exp(-cube_uncert).mean().item())

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

            return instances, losses

        # -------------------------------------------------------------------
        # 推論階段：整理並分配預測屬性
        # -------------------------------------------------------------------
        else:
            cube_x3d = cube_z * (cube_x - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            cube_y3d = cube_z * (cube_y - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            
            cube_3D = torch.stack([cube_x3d, cube_y3d, cube_z], dim=1)
            
            # 使用 split 分配回各圖片
            cube_3D_split = cube_3D.split(num_boxes_per_image)
            cube_xy_split = cube_xy.split(num_boxes_per_image)
            cube_dims_split = cube_dims.split(num_boxes_per_image)
            cube_pose_split = cube_pose.split(num_boxes_per_image)
            if self.use_conf:
                cube_conf = torch.exp(-cube_uncert)
                cube_conf_split = cube_conf.split(num_boxes_per_image)

            for b, result in enumerate(instances):
                if num_boxes_per_image[b] == 0:
                    continue
                result.pred_center_cam = cube_3D_split[b]
                result.pred_center_2D = cube_xy_split[b] * im_scales_ratio[b]
                result.pred_dimensions = cube_dims_split[b]
                result.pred_pose = cube_pose_split[b]
                
                result.pred_bbox3D = util.get_cuboid_verts_faces(
                    torch.cat([result.pred_center_cam, result.pred_dimensions], dim=1), 
                    result.pred_pose
                )[0]

                if self.use_conf:
                    result.pred_uncertainty = cube_conf_split[b]
                    result.scores = (result.scores * result.pred_uncertainty).clamp(min=0).sqrt()

            return instances


    # -------------------------------------------------------------
    # Loss Entry
    # -------------------------------------------------------------
    def losses(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        features: Dict[str, torch.Tensor],
        gt_instances: List[Instances],
        Ks: List[torch.Tensor],
        im_scales_ratio: List[float],
        im_current_dims: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, torch.Tensor]:

        locations, strides_per_point = self.compute_locations(features)
        points_all = torch.cat(locations, dim=0)
        strides_all = torch.cat(strides_per_point, dim=0)

        def flatten_level(t: List[torch.Tensor]) -> torch.Tensor:
            N, C = t[0].shape[0], t[0].shape[1]
            return torch.cat([x.permute(0, 2, 3, 1).reshape(N, -1, C) for x in t], dim=1)

        cls_logits = flatten_level(outputs["cls_logits"])
        box2d_reg = flatten_level(outputs["box2d_reg"])

        N = cls_logits.shape[0]
        device = cls_logits.device

        all_labels, all_gt_inds, all_target_scores = [], [], []
        gt_boxes3D_valid_list, gt_poses_valid_list, gt_box2d_valid_list = [], [], []

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
            # labels_i, gt_inds_i, target_score_i = self.assigner.assign(
            #     cls_logits[i].detach(), box_preds_i, points_all, strides_all,
            #     gt_boxes_i, gt_labels_i, gt_boxes_ign_i,
            # )

            all_labels.append(labels_i)
            all_gt_inds.append(gt_inds_i)
            # all_target_scores.append(target_score_i)

        gt_labels = torch.stack(all_labels)
        gt_gt_inds = torch.stack(all_gt_inds)

        valid_mask = gt_labels >= 0
        fg_mask = (gt_labels < self.num_classes) & valid_mask
        num_fg = max(fg_mask.sum().item(), 1)

        storage = get_event_storage()
        storage.put_scalar("dense_cube/num_fg", num_fg / N)

        # ---- 1. 分類 focal loss ----
        cls_target = torch.zeros_like(cls_logits)
        pos_labels = gt_labels[fg_mask]
        cls_target[fg_mask] = F.one_hot(pos_labels, self.num_classes).float()
        # cls_target = torch.stack(all_target_scores)

        loss_cls = sigmoid_focal_loss_jit(
            cls_logits[valid_mask], cls_target[valid_mask],
            alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum",
        ) / num_fg

        # loss_cls = quality_focal_loss(
        #     cls_logits[valid_mask], cls_target[valid_mask],
        #     beta=self.focal_gamma, reduction="sum",
        # ) / num_fg

        losses = {"DenseCube/loss_cls": loss_cls * self.loss_w_cls}

        if fg_mask.sum() == 0:
            return losses

        img_idx, pt_idx = fg_mask.nonzero(as_tuple=True)
        gt_idx = gt_gt_inds[img_idx, pt_idx]
        box_classes = gt_labels[img_idx, pt_idx]

        pts_fg = points_all[pt_idx]
        box2d_reg_fg = box2d_reg[img_idx, pt_idx]
        pred_box2d = self._decode_box2d(pts_fg, box2d_reg_fg)

        gt_counts = torch.tensor([v.shape[0] for v in gt_boxes3D_valid_list], device=device, dtype=torch.long)
        gt_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), gt_counts.cumsum(0)[:-1]])
        global_gt_idx = gt_offsets[img_idx] + gt_idx

        gt_boxes3D_cat = torch.cat(gt_boxes3D_valid_list, dim=0)
        gt_poses_cat = torch.cat(gt_poses_valid_list, dim=0)
        gt_box2d_cat = torch.cat(gt_box2d_valid_list, dim=0)

        gt_boxes3D_fg = gt_boxes3D_cat[global_gt_idx]
        gt_poses_fg = gt_poses_cat[global_gt_idx]
        gt_box2d_fg = gt_box2d_cat[global_gt_idx]

        # ---- 2. 2D GIoU loss ----
        loss_box2d = giou_loss(pred_box2d, gt_box2d_fg, reduction="sum") / num_fg
        losses["DenseCube/loss_box2d"] = loss_box2d * self.loss_w_box2d

        if self.loss_w_3d <= 0:
            return losses

        # ---- 3. 交由 _forward_cube 計算 3D Losses ----
        # with torch.no_grad():
        #     proposal_boxes_fg_all = self._decode_box2d(pts_fg, box2d_reg_fg.detach())

        # proposal_boxes_per_image, box_classes_per_image = [], []
        # gt_boxes3D_per_image, gt_poses_per_image = [], []

        # for b in range(N):
        #     m = (img_idx == b)
        #     proposal_boxes_per_image.append(proposal_boxes_fg_all[m])
        #     box_classes_per_image.append(box_classes[m])
        #     gt_boxes3D_per_image.append(gt_boxes3D_fg[m])
        #     gt_poses_per_image.append(gt_poses_fg[m])

        # # 訓練穩定化：混入GT框
        # if self.gt_in_proposals:
        #     proposal_boxes_per_image, box_classes_per_image, gt_boxes3D_per_image, gt_poses_per_image = \
        #         self._augment_proposals_with_gt(
        #             proposal_boxes_per_image, box_classes_per_image,
        #             gt_boxes3D_per_image, gt_poses_per_image, gt_instances,
        #         )

        # # 包裝成 Instances 列表以配合 _forward_cube 的標準介面
        # proposals = []
        # for b in range(N):
        #     inst = Instances(gt_instances[b].image_size)
        #     inst.proposal_boxes = Boxes(proposal_boxes_per_image[b])
        #     inst.gt_classes = box_classes_per_image[b]
        #     inst.gt_boxes3D = gt_boxes3D_per_image[b]
        #     inst.gt_poses = gt_poses_per_image[b]
        #     proposals.append(inst)

        # _, cube_losses = self._forward_cube(
        #     features=features, instances=proposals, Ks=Ks, 
        #     im_scales_ratio=im_scales_ratio, im_current_dims=im_current_dims,
        #     is_training=True
        # )
        # losses.update(cube_losses)

        proposals = self._sample_and_augment_proposals(
            num_fg=num_fg,
            img_idx=img_idx,
            box_classes=box_classes,
            pts_fg=pts_fg,
            box2d_reg_fg=box2d_reg_fg,
            gt_boxes3D_fg=gt_boxes3D_fg,
            gt_poses_fg=gt_poses_fg,
            gt_instances=gt_instances,
            N=N,
            device=device,
            max_proposals=self.max_proposals
        )

        _, cube_losses = self._forward_cube(
            features=features, instances=proposals, Ks=Ks, 
            im_scales_ratio=im_scales_ratio, im_current_dims=im_current_dims,
            is_training=True
        )
        losses.update(cube_losses)

        return losses

    # ===================================================================
    # Inference Entry
    # ===================================================================
    @torch.no_grad()
    def inference(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        features: Dict[str, torch.Tensor],
        Ks: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        im_scales_ratio: List[float],
        im_current_dims: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Instances]:

        locations, strides_per_point = self.compute_locations(features)
        num_levels = len(locations)
        results = []
        N = outputs["cls_logits"][0].shape[0]
        device = features[self.in_features[0]].device

        for img_idx in range(N):
            boxes_all, scores_all, classes_all = [], [], []

            for lvl in range(num_levels):
                cls_l = outputs["cls_logits"][lvl][img_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                box_l = outputs["box2d_reg"][lvl][img_idx].permute(1, 2, 0).reshape(-1, 4)

                scores_l = cls_l.sigmoid()
                topk = min(self.test_topk_candidates, scores_l.numel())
                flat_scores, flat_idx = scores_l.reshape(-1).topk(topk)
                keep_mask = flat_scores > self.test_score_thresh
                flat_scores = flat_scores[keep_mask]
                flat_idx = flat_idx[keep_mask]

                pt_idx = torch.div(flat_idx, self.num_classes, rounding_mode='floor')
                cls_idx = flat_idx % self.num_classes

                pts_l = locations[lvl][pt_idx]
                box_l_sel = box_l[pt_idx]
                # 保持未裁剪，與訓練時 assigner 使用的參考框一致，
                # 避免邊界物體在訓練/推理間的 3D XY 解碼系統性偏移。
                boxes_dec = self._decode_box2d(pts_l, box_l_sel)

                boxes_all.append(boxes_dec)
                scores_all.append(flat_scores)
                classes_all.append(cls_idx)

            h, w = image_sizes[img_idx]

            if len(boxes_all) == 0 or torch.cat(boxes_all).shape[0] == 0:
                results.append(self._empty_instances((h, w), device))
                continue

            boxes_all = torch.cat(boxes_all)
            scores_all = torch.cat(scores_all)
            classes_all = torch.cat(classes_all)

            # NMS 同樣在未裁剪框上進行，與訓練時 assigner 看到的座標系一致
            keep = batched_nms(boxes_all, scores_all, classes_all, self.test_nms_thresh)
            keep = keep[: self.test_max_detections]

            boxes_k = boxes_all[keep]
            scores_k = scores_all[keep]
            classes_k = classes_all[keep]
            n_k = boxes_k.shape[0]

            if n_k == 0:
                results.append(self._empty_instances((h, w), device))
                continue

            result = Instances((h, w))
            # 暫存未裁剪框：供後續 cube ROI pooling / XY 解碼使用，最終輸出前才裁剪
            result.pred_boxes = Boxes(boxes_k)
            result.scores = scores_k
            result.pred_classes = classes_k
            results.append(result)

        # 統一走一次 3D 分支（未裁剪框全程參與 ROI pooling 與幾何解碼，與訓練路徑一致）
        if self.loss_w_3d > 0:
            results = self._forward_cube(
                features=features, instances=results, Ks=Ks,
                im_scales_ratio=im_scales_ratio, im_current_dims=im_current_dims,
                is_training=False
            )

        # 3D 計算完成後，才把 2D 輸出框裁剪到影像邊界內
        for res, (h, w) in zip(results, image_sizes):
            if len(res) == 0:
                continue
            clipped = res.pred_boxes.tensor.clone()
            clipped[:, 0::2] = clipped[:, 0::2].clamp(0, w)
            clipped[:, 1::2] = clipped[:, 1::2].clamp(0, h)
            res.pred_boxes = Boxes(clipped)

        return results

    def _sample_and_augment_proposals(
        self,
        num_fg: int,
        img_idx: torch.Tensor,
        box_classes: torch.Tensor,
        pts_fg: torch.Tensor,
        box2d_reg_fg: torch.Tensor,
        gt_boxes3D_fg: torch.Tensor,
        gt_poses_fg: torch.Tensor,
        gt_instances: List[Instances],
        N: int,
        device: torch.device,
        max_proposals: int = 256
    ) -> List[Instances]:
        
        # 1. 隨機抽樣限制數量 (Subsample)
        if num_fg > max_proposals:
            perm = torch.randperm(num_fg, device=device)[:max_proposals]
            img_idx = img_idx[perm]
            box_classes = box_classes[perm]
            pts_fg = pts_fg[perm]
            box2d_reg_fg = box2d_reg_fg[perm]
            gt_boxes3D_fg = gt_boxes3D_fg[perm]
            gt_poses_fg = gt_poses_fg[perm]

        # 2. 幾何解碼
        with torch.no_grad():
            proposal_boxes_fg_all = self._decode_box2d(pts_fg, box2d_reg_fg.detach())

        # 3. 依據 Batch Index (img_idx) 分配回各張圖片
        proposal_boxes_per_image, box_classes_per_image = [], []
        gt_boxes3D_per_image, gt_poses_per_image = [], []

        for b in range(N):
            m = (img_idx == b)
            proposal_boxes_per_image.append(proposal_boxes_fg_all[m])
            box_classes_per_image.append(box_classes[m])
            gt_boxes3D_per_image.append(gt_boxes3D_fg[m])
            gt_poses_per_image.append(gt_poses_fg[m])

        # 4. 混入 Ground Truth
        if self.gt_in_proposals:
            proposal_boxes_per_image, box_classes_per_image, gt_boxes3D_per_image, gt_poses_per_image = \
                self._augment_proposals_with_gt(
                    proposal_boxes_per_image, box_classes_per_image,
                    gt_boxes3D_per_image, gt_poses_per_image, gt_instances,
                )

        # 5. 封裝為 Instances
        proposals = []
        for b in range(N):
            inst = Instances(gt_instances[b].image_size)
            inst.proposal_boxes = Boxes(proposal_boxes_per_image[b])
            inst.gt_classes = box_classes_per_image[b]
            inst.gt_boxes3D = gt_boxes3D_per_image[b]
            inst.gt_poses = gt_poses_per_image[b]
            proposals.append(inst)

        return proposals

    def _empty_instances(self, image_size, device):
        empty = Instances(image_size)
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
        return empty


def build_cascade_dense_cube_head(cfg, input_shape: Dict[str, ShapeSpec], priors: Optional[dict] = None):
    name = cfg.MODEL.DENSE_HEAD.NAME
    return CASCADE_DENSE_CUBE_HEAD_REGISTRY.get(name)(cfg, input_shape, priors=priors)
