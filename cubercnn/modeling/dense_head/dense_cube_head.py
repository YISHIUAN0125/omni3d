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

from cubercnn.modeling.dense_head.assigner import build_assigner, SimOTAAssigner
from cubercnn import util

logger = logging.getLogger(__name__)

E_CONSTANT = 2.71828183
SQRT_2_CONSTANT = 1.41421356

DENSE_CUBE_HEAD_REGISTRY = Registry("DENSE_CUBE_HEAD")


class Scale(nn.Module):
    """
    用於 2D 邊界框迴歸的分層可學習縮放因子（FCOS 風格）。
    允許每個 FPN 特徵層級自適應調整迴歸 offsets 的尺度範圍。
    """
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


@DENSE_CUBE_HEAD_REGISTRY.register()
class DenseCubeHead(nn.Module):
    """
    單階段 Dense 3D 邊界框預測頭（Cube R-CNN Single-Stage 版）。
    省去 RPN 與 RoIAlign，在多尺度特徵圖上密集預測 2D 框、Centerness、分類機率，
    以及 3D 方體參數（XY 投影偏移、深度 Z、3D 尺寸、3D 旋轉姿態與不確定性信心度）。
    """

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
        assigner: SimOTAAssigner,
        loss_w_3d: float,
        loss_w_cls: float,
        loss_w_box2d: float,
        loss_w_center: float,
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
        self.pose_type = pose_type
        self.use_conf = use_conf
        self.cluster_bins = max(cluster_bins, 1)
        self.assigner = assigner

        # 損失權重設定
        self.loss_w_3d = loss_w_3d
        self.loss_w_cls = loss_w_cls
        self.loss_w_box2d = loss_w_box2d
        self.loss_w_center = loss_w_center
        self.loss_w_xy = loss_w_xy
        self.loss_w_z = loss_w_z
        self.loss_w_dims = loss_w_dims
        self.loss_w_pose = loss_w_pose
        self.loss_w_joint = loss_w_joint

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # 推論與後處理參數
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.test_max_detections = test_max_detections

        # 3D 迴歸與幾何損失配置
        self.z_type = z_type
        self.dims_priors_enabled = dims_priors_enabled
        self.dims_priors_func = dims_priors_func
        self.chamfer_pose = chamfer_pose
        self.disentangled_loss = disentangled_loss
        self.inverse_z_weight = inverse_z_weight
        self.allocentric_pose = allocentric_pose
        self.virtual_depth = virtual_depth
        self.virtual_focal = virtual_focal

        # 初始化各類別 3D 尺寸先驗 (Dimensions Priors)
        if self.dims_priors_enabled and priors is not None:
            self.priors_dims_per_cat = nn.Parameter(
                torch.FloatTensor(priors['priors_dims_per_cat']).unsqueeze(0)
            )
        else:
            self.priors_dims_per_cat = nn.Parameter(torch.ones(1, num_classes, 2, 3))

        # 初始化深度群聚尺度區間 (Z-Cluster Scale Bins)
        if self.cluster_bins > 1 and priors is not None:
            priors_z_scales = torch.stack([torch.FloatTensor(p[1]) for p in priors['priors_bins']])
            self.priors_z_scales = nn.Parameter(priors_z_scales)
        else:
            self.priors_z_scales = nn.Parameter(torch.ones(num_classes, self.cluster_bins))

        # 初始化基於群聚的深度統計值 (均值與標準差)
        if self.z_type == 'clusters':
            assert self.cluster_bins > 1, 'z_type=clusters 需要 cluster_bins > 1'
            if priors is None:
                self.priors_z_stats = nn.Parameter(torch.ones(num_classes, self.cluster_bins, 2).float())
            else:
                priors_z_stats = torch.cat([torch.FloatTensor(p[2]).unsqueeze(0) for p in priors['priors_bins']])
                self.priors_z_stats = nn.Parameter(priors_z_stats)

        # -------------------------------------------------------------
        # 卷積特徵提取塔 (Convolutional Towers)
        # -------------------------------------------------------------
        self.cls_tower = self._make_tower(conv_dim, num_convs)
        self.box2d_tower = self._make_tower(conv_dim, num_convs)
        self.cube_tower = self._make_tower(conv_dim, num_convs)

        # -------------------------------------------------------------
        # 密集預測輸出分支 (Dense Prediction Heads)
        # -------------------------------------------------------------
        self.cls_score = nn.Conv2d(conv_dim, num_classes, 3, padding=1)
        self.centerness = nn.Conv2d(conv_dim, 1, 3, padding=1)
        self.box2d_reg = nn.Conv2d(conv_dim, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in fpn_strides])

        # 3D 方體預測頭 (類別共享特徵結構)
        self.cube_2d_deltas = nn.Conv2d(conv_dim, 2, 3, padding=1)
        self.cube_dims = nn.Conv2d(conv_dim, 3, 3, padding=1)
        self.cube_z = nn.Conv2d(conv_dim, self.cluster_bins, 3, padding=1)

        if pose_type == '6d':
            pose_ch = 6
        elif pose_type == 'quaternion':
            pose_ch = 4
        elif pose_type == 'euler':
            pose_ch = 3
        else:
            raise ValueError(f'不支援的 3D 姿態類型: {pose_type}')
        self.cube_pose = nn.Conv2d(conv_dim, pose_ch, 3, padding=1)

        if self.use_conf:
            self.cube_uncert = nn.Conv2d(conv_dim, 1, 3, padding=1)

        self._init_weights(prior_prob)

    def _make_tower(self, conv_dim: int, num_convs: int) -> nn.Sequential:
        layers = []
        for _ in range(num_convs):
            conv = nn.Conv2d(conv_dim, conv_dim, 3, padding=1)
            weight_init.c2_xavier_fill(conv)
            layers.append(conv)
            layers.append(nn.GroupNorm(32, conv_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _init_weights(self, prior_prob: float):
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)
        for m in [self.cube_2d_deltas, self.cube_dims, self.cube_pose, self.cube_z]:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        if self.use_conf:
            nn.init.normal_(self.cube_uncert.weight, std=0.001)
            nn.init.constant_(self.cube_uncert.bias, 5)

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
            "loss_w_center": cfg.MODEL.DENSE_HEAD.LOSS_W_CENTERNESS,
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
        cls_logits, centerness, box2d_reg = [], [], []
        cube_deltas, cube_z, cube_dims, cube_pose, cube_uncert = [], [], [], [], []

        for level, f in enumerate(self.in_features):
            x = features[f]

            cls_feat = self.cls_tower(x)
            box_feat = self.box2d_tower(x)
            cube_feat = self.cube_tower(x)

            cls_logits.append(self.cls_score(cls_feat))
            centerness.append(self.centerness(box_feat))
            # ReLU 確保 2D 距離 offsets 非負，Scale 模組自適應調整各 FPN 層級尺度
            # box2d_reg.append(F.relu(self.scales[level](self.box2d_reg(box_feat))))
            stride = self.fpn_strides[level]
            box2d_reg.append(F.relu(self.scales[level](self.box2d_reg(box_feat))) * stride)

            cube_deltas.append(self.cube_2d_deltas(cube_feat))
            cube_z.append(self.cube_z(cube_feat))
            cube_dims.append(self.cube_dims(cube_feat))
            cube_pose.append(self.cube_pose(cube_feat))
            if self.use_conf:
                cube_uncert.append(self.cube_uncert(cube_feat).clamp(min=0.01))

        return {
            "cls_logits": cls_logits,
            "centerness": centerness,
            "box2d_reg": box2d_reg,
            "cube_deltas": cube_deltas,
            "cube_z": cube_z,
            "cube_dims": cube_dims,
            "cube_pose": cube_pose,
            "cube_uncert": cube_uncert if self.use_conf else None,
        }

    # ===================================================================
    # 幾何座標網格與輔助工具
    # ===================================================================
    def compute_locations(self, features: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        計算所有 FPN 特徵層級的影像空間中心點座標 (x, y) 與對應 stride。
        """
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
            # 特徵感受野單元的中心點座標
            points = torch.stack((shift_x, shift_y), dim=1) + stride // 2
            locations.append(points)
            strides_per_point.append(torch.full((points.shape[0],), stride, dtype=torch.float32, device=device))
        return locations, strides_per_point

    def _decode_box2d(self, points: torch.Tensor, box2d_reg: torch.Tensor) -> torch.Tensor:
        """
        將中心點與 (l, t, r, b) 偏移量解碼為 (x1, y1, x2, y2) 2D 框座標。
        """
        x1 = points[:, 0] - box2d_reg[:, 0]
        y1 = points[:, 1] - box2d_reg[:, 1]
        x2 = points[:, 0] + box2d_reg[:, 2]
        y2 = points[:, 1] + box2d_reg[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)

    def l1_loss(self, vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """與原版 Cube R-CNN 完全一致的 Smooth L1 損失 (beta=0.0 即為標準 L1)。"""
        return F.smooth_l1_loss(vals, target, reduction='none', beta=0.0)

    def chamfer_loss(self, vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 3D 邊界框 8 個角點點集之間的雙向 Chamfer L1 距離。
        vals, target 形狀: (B, 8, 3)
        """
        B = vals.shape[0]
        xx = vals.view(B, 8, 1, 3)
        yy = target.view(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)
        return l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1)

    def _safely_reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """過濾 NaN/Inf 數值以確保 Loss Reduce 過程數值穩定。"""
        valid = (~loss.isinf()) & (~loss.isnan())
        if valid.any():
            return loss[valid].mean()
        return loss.mean() * 0.0

    # ===================================================================
    # 損失計算 (Loss Computation)
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
        centerness = flatten_level(outputs["centerness"])      # (N, P, 1)
        box2d_reg = flatten_level(outputs["box2d_reg"])        # (N, P, 4)
        cube_deltas = flatten_level(outputs["cube_deltas"])    # (N, P, 2)
        cube_z_raw = flatten_level(outputs["cube_z"])          # (N, P, cluster_bins)
        cube_dims_raw = flatten_level(outputs["cube_dims"])    # (N, P, 3)
        cube_pose_raw = flatten_level(outputs["cube_pose"])    # (N, P, pose_ch)
        cube_uncert_raw = flatten_level(outputs["cube_uncert"]) if self.use_conf else None

        N = cls_logits.shape[0]
        device = cls_logits.device

        all_labels, all_gt_inds = [], []

        # 對每張影像執行 SimOTA 正負樣本動態分配
        for i in range(N):
            gts_i = gt_instances[i]
            valid = gts_i.gt_classes >= 0
            gt_boxes_i = gts_i.gt_boxes.tensor[valid]
            gt_labels_i = gts_i.gt_classes[valid]
            gt_boxes_ign_i = gts_i.gt_boxes.tensor[~valid] if (~valid).any() else None

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

        valid_mask = gt_labels >= 0              # 排除 ignore 區域 (-1)
        fg_mask = (gt_labels < self.num_classes) & valid_mask  # 僅保留正樣本點

        num_fg = max(fg_mask.sum().item(), 1)

        storage = get_event_storage()
        storage.put_scalar("dense_cube/num_fg", num_fg / N)

        # -----------------------------------------------------------
        # 1. 類別 Focal Loss
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

        # -----------------------------------------------------------
        # 提取正樣本對應的預測與 GT 張量
        # -----------------------------------------------------------
        img_idx, pt_idx = fg_mask.nonzero(as_tuple=True)
        gt_idx = gt_gt_inds[img_idx, pt_idx]
        box_classes = gt_labels[img_idx, pt_idx]

        pts_fg = points_all[pt_idx]
        box2d_reg_fg = box2d_reg[img_idx, pt_idx]
        centerness_fg = centerness[img_idx, pt_idx, 0]
        cube_deltas_fg = cube_deltas[img_idx, pt_idx]
        cube_dims_fg = cube_dims_raw[img_idx, pt_idx]
        cube_pose_fg = cube_pose_raw[img_idx, pt_idx]
        cube_z_fg = cube_z_raw[img_idx, pt_idx]
        cube_uncert_fg = cube_uncert_raw[img_idx, pt_idx, 0] if self.use_conf else None

        gt_boxes3D_fg = torch.cat([gt_instances[b].gt_boxes3D[g].unsqueeze(0)
                                    for b, g in zip(img_idx.tolist(), gt_idx.tolist())], dim=0)
        gt_poses_fg = torch.cat([gt_instances[b].gt_poses[g].unsqueeze(0)
                                  for b, g in zip(img_idx.tolist(), gt_idx.tolist())], dim=0)
        gt_box2d_fg = torch.cat([gt_instances[b].gt_boxes.tensor[g].unsqueeze(0)
                                  for b, g in zip(img_idx.tolist(), gt_idx.tolist())], dim=0)

        Ks_fg = torch.stack([Ks[b] / im_scales_ratio[b] for b in img_idx.tolist()]).to(device)
        Ks_fg[:, -1, -1] = 1

        # -----------------------------------------------------------
        # 2. 2D 邊界框 (GIoU) 與 Centerness 損失
        # -----------------------------------------------------------
        pred_box2d = self._decode_box2d(pts_fg, box2d_reg_fg)
        loss_box2d = giou_loss(pred_box2d, gt_box2d_fg, reduction="sum") / num_fg

        l_t = pts_fg[:, 0] - gt_box2d_fg[:, 0]
        t_t = pts_fg[:, 1] - gt_box2d_fg[:, 1]
        r_t = gt_box2d_fg[:, 2] - pts_fg[:, 0]
        b_t = gt_box2d_fg[:, 3] - pts_fg[:, 1]
        lr = torch.stack([l_t, r_t], dim=1).clamp(min=0)
        tb = torch.stack([t_t, b_t], dim=1).clamp(min=0)
        centerness_target = torch.sqrt(
            (lr.min(1)[0] / lr.max(1)[0].clamp(min=1e-6))
            * (tb.min(1)[0] / tb.max(1)[0].clamp(min=1e-6))
        ).clamp(0, 1)
        loss_centerness = F.binary_cross_entropy_with_logits(
            centerness_fg, centerness_target.detach(), reduction="sum"
        ) / num_fg

        losses["DenseCube/loss_box2d"] = loss_box2d * self.loss_w_box2d
        losses["DenseCube/loss_centerness"] = loss_centerness * self.loss_w_center

        # -----------------------------------------------------------
        # 3. 3D 預測解碼 (以 Detached 2D 框作為參考基準)
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

        # 解碼投影 2D 中心 (XY)
        cube_x = src_ctr_x + src_widths * cube_deltas_fg[:, 0]
        cube_y = src_ctr_y + src_heights * cube_deltas_fg[:, 1]
        cube_xy = torch.stack([cube_x, cube_y], dim=1)

        # 解碼 3D 尺寸 (Dimensions)
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
                cube_dims = torch.exp(cube_dims_norm.clamp(max=5)) * prior_dims_mean
        else:
            cube_dims = torch.exp(cube_dims_norm.clamp(max=5))

        # 解碼 3D 旋轉姿態 (Rotation Pose)
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

        # 解碼 3D 深度 (Z)
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
            cube_z = torch.exp(cube_z_sel)
        elif self.z_type == 'clusters':
            fg_arange = torch.arange(num_fg, device=device)
            z_means = self.priors_z_stats[box_classes, assignments[fg_arange, box_classes], 0].detach()
            z_stds = self.priors_z_stats[box_classes, assignments[fg_arange, box_classes], 1].detach()
            z_mins = (z_means - 3 * z_stds).clamp(min=0)
            z_maxs = z_means + 3 * z_stds
            cube_z_norm = cube_z_sel
            cube_z = util.scaled_sigmoid(cube_z_sel, min=z_mins, max=z_maxs)
        else:
            raise ValueError(f'不支援的 z_type: {self.z_type}')

        if self.virtual_depth:
            cube_z = cube_z * virtual_to_real

        # -----------------------------------------------------------
        # 4. Ground Truth 3D 方體與角點投影
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
        # 5. 3D 幾何損失計算 (Disentangled vs Direct)
        # -----------------------------------------------------------
        if self.disentangled_loss:
            # 深度 Z 解耦損失
            cube_dis_x3d_from_z = cube_z * (gt_2d[:, 0] - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            cube_dis_y3d_from_z = cube_z * (gt_2d[:, 1] - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            dis_z_box = torch.cat(
                (torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z), dim=1), gt_dims), dim=1
            )
            dis_z_corners = util.get_cuboid_verts_faces(dis_z_box, gt_poses_fg)[0]
            loss_z = self.l1_loss(dis_z_corners, gt_corners).reshape(num_fg, -1).mean(1)

            # 中心 XY 解耦損失
            cube_dis_x3d = gt_z * (cube_x - Ks_fg[:, 0, 2]) / Ks_fg[:, 0, 0]
            cube_dis_y3d = gt_z * (cube_y - Ks_fg[:, 1, 2]) / Ks_fg[:, 1, 1]
            dis_xy_box = torch.cat(
                (torch.stack((cube_dis_x3d, cube_dis_y3d, gt_z), dim=1), gt_dims), dim=1
            )
            dis_xy_corners = util.get_cuboid_verts_faces(dis_xy_box, gt_poses_fg)[0]
            loss_xy = self.l1_loss(dis_xy_corners, gt_corners).reshape(num_fg, -1).mean(1)

            # 尺寸 Dims 解耦損失
            dis_dims_corners = util.get_cuboid_verts_faces(torch.cat((gt_3d, cube_dims), dim=1), gt_poses_fg)[0]
            loss_dims = self.l1_loss(dis_dims_corners, gt_corners).reshape(num_fg, -1).mean(1)

            # 旋轉姿態 Pose 解耦損失 (支援 Chamfer Pose)
            dis_pose_corners = util.get_cuboid_verts_faces(gt_box3d, cube_pose)[0]
            if self.chamfer_pose:
                loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)
            else:
                loss_pose = self.l1_loss(dis_pose_corners, gt_corners).reshape(num_fg, -1).mean(1)
        else:
            gt_deltas_x = (gt_2d[:, 0] - src_ctr_x) / src_widths
            gt_deltas_y = (gt_2d[:, 1] - src_ctr_y) / src_heights
            gt_deltas = torch.stack([gt_deltas_x, gt_deltas_y], dim=1)
            loss_xy = self.l1_loss(cube_deltas_fg, gt_deltas).mean(1)

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
                loss_pose = torch.zeros(num_fg, device=device)

            if self.z_type == 'direct':
                loss_z = self.l1_loss(cube_z, gt_z)
            elif self.z_type == 'sigmoid':
                loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual / 100).clamp(0, 1))
            elif self.z_type == 'log':
                loss_z = self.l1_loss(cube_z_norm, torch.log((gt_z * real_to_virtual).clamp(min=0.01)))
            elif self.z_type == 'clusters':
                loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual - z_means) / z_stds)

        # -----------------------------------------------------------
        # 6. Joint (Entangled) 3D 邊界框組合損失
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

        # -----------------------------------------------------------
        # 7. 損失加權機制 (Inverse-Z & Uncertainty Weighting)
        # -----------------------------------------------------------
        if self.inverse_z_weight:
            inverse_z_w = 1.0 / torch.log(gt_z.clamp(min=E_CONSTANT))
            loss_dims = loss_dims * inverse_z_w
            if loss_xy is not None:
                loss_xy = loss_xy * inverse_z_w
            if loss_z is not None:
                loss_z = loss_z * inverse_z_w
            if loss_pose is not None:
                loss_pose = loss_pose * inverse_z_w
            if loss_joint is not None:
                loss_joint = loss_joint * inverse_z_w

        if self.use_conf:
            uncert_sf = SQRT_2_CONSTANT * torch.exp(-cube_uncert_fg)
            loss_dims = loss_dims * uncert_sf
            if loss_xy is not None:
                loss_xy = loss_xy * uncert_sf
            if loss_z is not None:
                loss_z = loss_z * uncert_sf
            if loss_pose is not None:
                loss_pose = loss_pose * uncert_sf
            if loss_joint is not None:
                loss_joint = loss_joint * uncert_sf

            losses["DenseCube/loss_uncert"] = self._safely_reduce(cube_uncert_fg.clone())
            storage.put_scalar("DenseCube/conf", torch.exp(-cube_uncert_fg).mean().item())

        # 彙總並輸出各項 3D 損失 (乘上 loss_w_3d 全域縮放因子)
        losses["DenseCube/loss_xy"] = self._safely_reduce(loss_xy) * self.loss_w_xy * self.loss_w_3d
        losses["DenseCube/loss_dims"] = self._safely_reduce(loss_dims) * self.loss_w_dims * self.loss_w_3d
        losses["DenseCube/loss_z"] = self._safely_reduce(loss_z) * self.loss_w_z * self.loss_w_3d
        losses["DenseCube/loss_pose"] = self._safely_reduce(loss_pose) * self.loss_w_pose * self.loss_w_3d
        if loss_joint is not None:
            valid_joint = (~loss_joint.isinf()) & (~loss_joint.isnan())
            if valid_joint.any():
                losses["DenseCube/loss_joint"] = self._safely_reduce(loss_joint[valid_joint]) * self.loss_w_joint * self.loss_w_3d

        # 記錄評估指標至 Event Storage (TensorBoard)
        z_error = (cube_z - gt_z).detach().abs()
        dims_error = (cube_dims - gt_dims).detach().abs()
        xy_error = (cube_xy - gt_2d).detach().abs()
        storage.put_scalar("DenseCube/z_error", z_error.mean().item(), smoothing_hint=False)
        storage.put_scalar("DenseCube/dims_error", dims_error.mean().item(), smoothing_hint=False)
        storage.put_scalar("DenseCube/xy_error", xy_error.mean().item(), smoothing_hint=False)
        storage.put_scalar("DenseCube/z_close", (z_error < 0.20).float().mean().item(), smoothing_hint=False)

        total_3d_loss = loss_dims * self.loss_w_dims + loss_xy * self.loss_w_xy + loss_z * self.loss_w_z + loss_pose * self.loss_w_pose
        if loss_joint is not None:
            total_3d_loss = total_3d_loss + (loss_joint * self.loss_w_joint)
        storage.put_scalar("DenseCube/total_3D_loss", self.loss_w_3d * self._safely_reduce(total_3d_loss.detach()).item(), smoothing_hint=False)

        return losses

    # ===================================================================
    # 推論 (Inference)
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
                cls_l = outputs["cls_logits"][lvl][img_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                box_l = outputs["box2d_reg"][lvl][img_idx].permute(1, 2, 0).reshape(-1, 4)
                ctr_l = outputs["centerness"][lvl][img_idx].permute(1, 2, 0).reshape(-1)
                delta_l = outputs["cube_deltas"][lvl][img_idx].permute(1, 2, 0).reshape(-1, 2)
                dims_l = outputs["cube_dims"][lvl][img_idx].permute(1, 2, 0).reshape(-1, 3)
                pose_l = outputs["cube_pose"][lvl][img_idx].permute(1, 2, 0)
                pose_l = pose_l.reshape(-1, pose_l.shape[-1])
                z_l = outputs["cube_z"][lvl][img_idx].permute(1, 2, 0).reshape(-1, self.cluster_bins)
                if self.use_conf:
                    uncert_l = outputs["cube_uncert"][lvl][img_idx].permute(1, 2, 0).reshape(-1)

                # 綜合分數: sqrt(分類機率 * centerness)
                scores_l = (cls_l.sigmoid() * ctr_l.sigmoid().unsqueeze(1)).sqrt()
                topk = min(self.test_topk_candidates, scores_l.numel())
                flat_scores, flat_idx = scores_l.reshape(-1).topk(topk)
                keep_mask = flat_scores > self.test_score_thresh
                flat_scores = flat_scores[keep_mask]
                flat_idx = flat_idx[keep_mask]

                pt_idx = flat_idx // self.num_classes
                cls_idx = flat_idx % self.num_classes

                pts_l = locations[lvl][pt_idx]
                box_l_sel = box_l[pt_idx]
                boxes_dec = self._decode_box2d(pts_l, box_l_sel)

                boxes_all.append(boxes_dec)
                scores_all.append(flat_scores)
                classes_all.append(cls_idx)
                deltas_all.append(delta_l[pt_idx])
                dims_all.append(dims_l[pt_idx])
                pose_all.append(pose_l[pt_idx])
                z_all.append(z_l[pt_idx])
                if self.use_conf:
                    uncert_all.append(uncert_l[pt_idx])

            if len(boxes_all) == 0 or torch.cat(boxes_all).shape[0] == 0:
                h, w = image_sizes[img_idx]
                empty = Instances((h, w))
                empty.pred_boxes = Boxes(torch.zeros((0, 4), device=device))
                empty.scores = torch.zeros(0, device=device)
                empty.pred_classes = torch.zeros(0, dtype=torch.int64, device=device)
                empty.pred_center_cam = torch.zeros((0, 3), device=device)
                empty.pred_center_2D = torch.zeros((0, 2), device=device)
                empty.pred_dimensions = torch.zeros((0, 3), device=device)
                empty.pred_pose = torch.zeros((0, 3, 3), device=device)
                empty.pred_bbox3D = torch.zeros((0, 8, 3), device=device)
                results.append(empty)
                continue

            boxes_all = torch.cat(boxes_all)
            scores_all = torch.cat(scores_all)
            classes_all = torch.cat(classes_all)

            # 限制 2D 預測框在影像範圍內
            h, w = image_sizes[img_idx]
            boxes_all[:, 0::2] = boxes_all[:, 0::2].clamp(0, w)
            boxes_all[:, 1::2] = boxes_all[:, 1::2].clamp(0, h)

            # 依類別執行 Batched NMS
            keep = batched_nms(boxes_all, scores_all, classes_all, self.test_nms_thresh)
            keep = keep[: self.test_max_detections]

            deltas_k = torch.cat(deltas_all)[keep]
            dims_k = torch.cat(dims_all)[keep]
            pose_k = torch.cat(pose_all)[keep]
            z_k = torch.cat(z_all)[keep]
            boxes_k = boxes_all[keep]

            n_k = boxes_k.shape[0]
            classes_k = classes_all[keep]

            K = (Ks[img_idx] / im_scales_ratio[img_idx]).to(device)
            K[-1, -1] = 1
            Ks_k = K.unsqueeze(0).repeat(n_k, 1, 1)

            src_widths = (boxes_k[:, 2] - boxes_k[:, 0]).clamp(min=1.0)
            src_heights = (boxes_k[:, 3] - boxes_k[:, 1]).clamp(min=1.0)
            src_ctr_x = (boxes_k[:, 0] + boxes_k[:, 2]) * 0.5
            src_ctr_y = (boxes_k[:, 1] + boxes_k[:, 3]) * 0.5
            src_scales = (src_widths ** 2 + src_heights ** 2).sqrt()

            focal_lengths_k = Ks_k[:, 1, 1]
            im_scale = float(im_current_dims[img_idx][0]) if im_current_dims is not None else 1.0
            im_scales_original_k = im_scale * im_scales_ratio[img_idx]

            if self.virtual_depth:
                virtual_to_real = util.compute_virtual_scale_from_focal_spaces(
                    focal_lengths_k,
                    torch.full((n_k,), im_scales_original_k, device=device),
                    self.virtual_focal,
                    torch.full((n_k,), im_scale, device=device),
                )
            else:
                virtual_to_real = 1.0

            cube_x = src_ctr_x + src_widths * deltas_k[:, 0]
            cube_y = src_ctr_y + src_heights * deltas_k[:, 1]

            # 3D 尺寸推論解碼
            if self.dims_priors_enabled:
                prior_dims = self.priors_dims_per_cat[0, classes_k]
                prior_dims_mean = prior_dims[:, 0, :]
                if self.dims_priors_func == 'sigmoid':
                    prior_dims_std = prior_dims[:, 1, :]
                    prior_dims_min = (prior_dims_mean - 3 * prior_dims_std).clamp(min=0.0)
                    prior_dims_max = prior_dims_mean + 3 * prior_dims_std
                    cube_dims = util.scaled_sigmoid(dims_k, min=prior_dims_min, max=prior_dims_max)
                else:
                    cube_dims = torch.exp(dims_k.clamp(max=5)) * prior_dims_mean
            else:
                cube_dims = torch.exp(dims_k.clamp(max=5))

            # 3D 姿態推論解碼
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

            # 3D 深度推論解碼
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
                cube_z = torch.exp(cube_z_sel)
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

            result = Instances((h, w))
            result.pred_boxes = Boxes(boxes_all[keep])
            result.scores = scores_all[keep]
            result.pred_classes = classes_all[keep]
            result.pred_center_cam = torch.stack([cube_x3d, cube_y3d, cube_z], dim=1)
            result.pred_center_2D = torch.stack([cube_x, cube_y], dim=1) * im_scales_ratio[img_idx]
            result.pred_dimensions = cube_dims
            result.pred_pose = cube_pose
            result.pred_bbox3D = util.get_cuboid_verts_faces(
                torch.cat([result.pred_center_cam, cube_dims], dim=1), cube_pose
            )[0]

            if self.use_conf:
                cube_conf = torch.exp(-torch.cat(uncert_all)[keep])
                result.scores = (result.scores * cube_conf).sqrt()

            results.append(result)

        return results


def build_dense_cube_head(cfg, input_shape: Dict[str, ShapeSpec], priors: Optional[dict] = None):
    name = cfg.MODEL.DENSE_HEAD.NAME
    return DENSE_CUBE_HEAD_REGISTRY.get(name)(cfg, input_shape, priors=priors)
