from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cubercnn import util as cubeutil

from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, TaskAlignedAssigner3D, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import rotation_6d_to_matrix, euler_angles_to_matrix, quaternion_to_matrix
from pytorch3d.transforms.so3 import so3_relative_angle

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist
from .mgiou import CM3DConfig, CubeMGIoUQualityBuilder


class CubeLoss(nn.Module):

    E_CONSTANT = 2.71828183
    SQRT_2_CONSTANT = 1.41421356

    def __init__(self, model_head: nn.Module):
        super().__init__()
        self.disentangled_loss = model_head.disentangled_loss
        self.chamfer_pose = model_head.chamfer_pose
        self.allocentric_pose = model_head.allocentric_pose
        self.pose_type = model_head.pose_type
        self.z_type = model_head.z_type
        self.cluster_bins = model_head.cluster_bins
        self.use_conf = model_head.use_conf
        self.virtual_depth = model_head.virtual_depth
        self.virtual_focal = model_head.virtual_focal
        self.dims_priors_enabled = model_head.dims_priors_enabled
        self.dims_priors_func = getattr(model_head, "dims_priors_func", "exp")

        self.loss_w_3d = getattr(model_head, "loss_w_3d", 1.0)
        self.loss_w_xy = getattr(model_head, "loss_w_xy", 1.0)
        self.loss_w_z = getattr(model_head, "loss_w_z", 1.0)
        self.loss_w_dims = getattr(model_head, "loss_w_dims", 1.0)
        self.loss_w_pose = getattr(model_head, "loss_w_pose", 1.0)
        self.loss_w_joint = getattr(model_head, "loss_w_joint", 0.0)
        self.inverse_z_weight = getattr(model_head, "inverse_z_weight", False)

        self.priors_dims_per_cat = getattr(model_head, "priors_dims_per_cat", None)
        self.priors_z_scales = getattr(model_head, "priors_z_scales", None)
        self.priors_z_stats = getattr(model_head, "priors_z_stats", None)


    @staticmethod
    def chamfer_loss(vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = vals.shape[0]
        xx = vals.reshape(B, 8, 1, 3)
        yy = target.reshape(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)  # (B, 8, 8)
        return l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1)  # (B,)

    @staticmethod
    def l1_loss(vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(vals, target, reduction="none", beta=0.0)

    def forward(
        self,
        cube_preds: dict[str, torch.Tensor],
        cube_decoded: dict[str, torch.Tensor],
        gt_box3d: torch.Tensor,       # (n_fg, 6) [x, y, z, w, h, l] 相機坐標系
        gt_pose: torch.Tensor,        # (n_fg, 3, 3) 相機坐標系
        gt_2d: torch.Tensor,          # (n_fg, 2) 3D 中心點投影至圖像的像素坐標
        src_boxes: torch.Tensor,      # (n_fg, 4) 2D 錨點/預測框 [x1, y1, x2, y2]
        box_classes: torch.Tensor,    # (n_fg,)
        Ks: torch.Tensor,             # (n_fg, 3, 3) 依影像尺度縮放後的相機內參
        weight: torch.Tensor,         # (n_fg, 1) YOLO TAL 賦予的正樣本匹配權重
        target_scores_sum: torch.Tensor,
        real_to_virtual: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        n_fg = gt_box3d.shape[0]
        device = gt_box3d.device

        pred_x = cube_decoded["center_2d"][..., 0]
        pred_y = cube_decoded["center_2d"][..., 1]
        pred_z = cube_decoded["z"]
        pred_dims = cube_decoded["dims"]
        pred_pose = cube_decoded["pose"]

        raw_deltas = cube_preds["deltas"]
        raw_dims = cube_preds["dims"]
        raw_z = cube_preds["z"]
        raw_pose = cube_preds["pose"]

        gt_z = gt_box3d[..., 2]
        gt_dims = gt_box3d[:, 3:6]

        gt_x3d = gt_z * (gt_2d[:, 0] - Ks[:, 0, 2]) / Ks[:, 0, 0]
        gt_y3d = gt_z * (gt_2d[:, 1] - Ks[:, 1, 2]) / Ks[:, 1, 1]
        gt_3d = torch.stack([gt_x3d, gt_y3d, gt_z], dim=-1)
        gt_box3d = torch.cat([gt_3d, gt_dims], dim=1)

        if self.disentangled_loss:
            gt_corners = cubeutil.get_cuboid_verts_faces(gt_box3d, gt_pose)[0]

            # disentangled z
            dis_x_from_z = pred_z * (gt_2d[..., 0] - Ks[..., 0, 2]) / Ks[..., 0, 0]
            dis_y_from_z = pred_z * (gt_2d[..., 1] - Ks[..., 1, 2]) / Ks[..., 1, 1]
            dis_z_box = torch.cat([torch.stack([dis_x_from_z, dis_y_from_z, pred_z], dim=1), gt_dims], dim=1)
            dis_z_corners = cubeutil.get_cuboid_verts_faces(dis_z_box, gt_pose)[0]
            loss_z = self.l1_loss(dis_z_corners, gt_corners).reshape(n_fg, -1).mean(1)

            # disentangled xy
            dis_x_from_xy = gt_z * (pred_x - Ks[..., 0, 2]) / Ks[..., 0, 0]
            dis_y_from_xy = gt_z * (pred_y - Ks[..., 1, 2]) / Ks[..., 1, 1]
            dis_xy_box = torch.cat([torch.stack([dis_x_from_xy, dis_y_from_xy, gt_z], dim=1), gt_dims], dim=1)
            dis_xy_corners = cubeutil.get_cuboid_verts_faces(dis_xy_box, gt_pose)[0]
            loss_xy = self.l1_loss(dis_xy_corners, gt_corners).reshape(n_fg, -1).mean(1)

            # disentangled dims
            dis_dims_box = torch.cat([gt_3d, pred_dims], dim=1)
            dis_dims_corners = cubeutil.get_cuboid_verts_faces(dis_dims_box, gt_pose)[0]
            loss_dims = self.l1_loss(dis_dims_corners, gt_corners).reshape(n_fg, -1).mean(1)

            # disentangled pose
            dis_pose_corners = cubeutil.get_cuboid_verts_faces(gt_box3d, pred_pose)[0]
            if self.chamfer_pose:
                loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)
            else:
                loss_pose = self.l1_loss(dis_pose_corners, gt_corners).reshape(n_fg, -1).mean(1)

        else:
            src_w = (src_boxes[:, 2] - src_boxes[:, 0]).clamp(min=1.0)
            src_h = (src_boxes[:, 3] - src_boxes[:, 1]).clamp(min=1.0)
            src_cx = (src_boxes[:, 0] + src_boxes[:, 2]) * 0.5
            src_cy = (src_boxes[:, 1] + src_boxes[:, 3]) * 0.5

            gt_deltas = torch.stack([(gt_2d[:, 0] - src_cx) / src_w, (gt_2d[:, 1] - src_cy) / src_h], dim=1)
            loss_xy = self.l1_loss(raw_deltas, gt_deltas).mean(1)

            if self.dims_priors_enabled and self.priors_dims_per_cat is not None:
                prior_dims = self.priors_dims_per_cat.detach()[0, box_classes, 0, :]
                cube_dims_gt_norm = torch.log((gt_dims / prior_dims).clamp(min=1e-5))
                loss_dims = self.l1_loss(raw_dims, cube_dims_gt_norm).mean(1)
            else:
                loss_dims = self.l1_loss(raw_dims, torch.log(gt_dims.clamp(min=1e-5))).mean(1)
            
            if self.pose_type == "6d":
                pred_rot_mat = rotation_6d_to_matrix(raw_pose)
            elif self.pose_type == "quaternion":
                pred_rot_mat = quaternion_to_matrix(F.normalize(raw_pose, dim=-1))
            else:
                pred_rot_mat = euler_angles_to_matrix(raw_pose, "XYZ")

            if self.allocentric_pose:
                gt_pose_allo = cubeutil.R_to_allocentric(Ks, gt_pose, u=pred_x.detach(), v=pred_y.detach())
                loss_pose = 1.0 - so3_relative_angle(pred_rot_mat, gt_pose_allo, eps=0.1, cos_angle=True)
            else:
                loss_pose = 1.0 - so3_relative_angle(pred_pose, gt_pose, eps=0.1, cos_angle=True)

            # loss depth-z
            z_target = gt_z * real_to_virtual
            if self.z_type == "direct":
                loss_z = self.l1_loss(pred_z, gt_z)
            elif self.z_type == "sigmoid":
                loss_z = self.l1_loss(torch.sigmoid(raw_z[:, 0]), (z_target / 100.0).clamp(0, 1))
            elif self.z_type == "log":
                loss_z = self.l1_loss(raw_z[:, 0], torch.log(z_target.clamp(min=0.01)))
            elif self.z_type == "clusters" and self.cluster_bins > 1:
                src_scales = (src_h**2 + src_w**2).sqrt()
                scales_diff = (self.priors_z_scales.detach().T.unsqueeze(0) - src_scales.unsqueeze(1).unsqueeze(2)).abs()
                assignments = scales_diff.argmin(1)
                assigned_bins = assignments[torch.arange(n_fg, device=device), box_classes]

                z_stats = self.priors_z_stats.detach()
                z_means = z_stats[:, :, 0].T.unsqueeze(0).repeat([n_fg, 1, 1])
                z_means = torch.gather(z_means, 1, assignments.unsqueeze(1)).squeeze(1)[torch.arange(n_fg, device=device), box_classes]
                z_stds = z_stats[:, :, 1].T.unsqueeze(0).repeat([n_fg, 1, 1])
                z_stds = torch.gather(z_stds, 1, assignments.unsqueeze(1)).squeeze(1)[torch.arange(n_fg, device=device), box_classes]

                z_norm_pred = raw_z.gather(1, assigned_bins.unsqueeze(1)).squeeze(1)
                loss_z = self.l1_loss(z_norm_pred, (z_target - z_means) / z_stds)
            else:
                loss_z = self.l1_loss(raw_z[:, 0], gt_z)

        # loss joint
        loss_joint = torch.zeros(n_fg, device=device)
        if self.loss_w_joint > 0:
            pred_box3d = torch.cat([cube_decoded["center_cam"], pred_dims], dim=1)
            pred_corners = cubeutil.get_cuboid_verts_faces(pred_box3d, pred_pose)[0]
            gt_corners = cubeutil.get_cuboid_verts_faces(gt_box3d, gt_pose)[0]
            if self.chamfer_pose and self.disentangled_loss:
                loss_joint = self.chamfer_loss(pred_corners, gt_corners)
            else:
                loss_joint = self.l1_loss(pred_corners, gt_corners).reshape(n_fg, -1).mean(1)

        # Inv z weight
        if self.inverse_z_weight:
            inv_z_w = 1.0 / torch.log(gt_z.clamp(min=self.E_CONSTANT))
            loss_xy *= inv_z_w
            loss_dims *= inv_z_w
            loss_z *= inv_z_w
            loss_pose *= inv_z_w
            if self.loss_w_joint > 0:
                loss_joint *= inv_z_w

        # loss uncertainty
        loss_uncert = torch.zeros(n_fg, device=device)
        if self.use_conf and "uncert" in cube_preds:
            uncert = cube_preds["uncert"]
            uncert_sf = self.SQRT_2_CONSTANT * torch.exp(-uncert)
            loss_xy *= uncert_sf
            loss_dims *= uncert_sf
            loss_z *= uncert_sf
            loss_pose *= uncert_sf
            if self.loss_w_joint > 0:
                loss_joint *= uncert_sf
            loss_uncert = uncert

        # combine with yolo TAL assigner foreground weight to reduce
        weight = weight.squeeze(-1)  # (n_fg,)
        l_xy = (loss_xy * weight).sum() / target_scores_sum * self.loss_w_xy
        l_dims = (loss_dims * weight).sum() / target_scores_sum * self.loss_w_dims
        l_z = (loss_z * weight).sum() / target_scores_sum * self.loss_w_z
        l_pose = (loss_pose * weight).sum() / target_scores_sum * self.loss_w_pose
        l_joint = (loss_joint * weight).sum() / target_scores_sum * self.loss_w_joint
        l_uncert = (loss_uncert * weight).sum() / target_scores_sum if self.use_conf else gt_box3d.sum() * 0.0

        total_3d_loss = (l_xy + l_dims + l_z + l_pose + l_joint + l_uncert) * self.loss_w_3d

        loss_items = {
            "loss_3d_xy": l_xy.detach(),
            "loss_3d_dims": l_dims.detach(),
            "loss_3d_z": l_z.detach(),
            "loss_3d_pose": l_pose.detach(),
        }
        if self.loss_w_joint > 0:
            loss_items["loss_3d_joint"] = l_joint.detach()
        if self.use_conf:
            loss_items["loss_3d_uncert"] = l_uncert.detach()

        return total_3d_loss, loss_items

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss"

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        # WARNING: line below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[0] += pred_distri[..., :0].sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class Detect3DLoss(v8DetectionLoss):
    def __init__(self, model: nn.Module, tal_topk: int = 10, 
                 tal_topk2: int | None = None, is_one2one: bool = False):
        super().__init__(model=model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        m = model.model[-1]
        self.cm3d_config = CM3DConfig(
            gamma=float(getattr(m, "cm3d_gamma", 1.0)),
            quality_floor=float(getattr(m, "cm3d_quality_floor", 0.05)),
            fast_mode=bool(getattr(m, "cm3d_fast_mode", True)),
            corner_order=getattr(m, "cm3d_corner_order", None),
        )
        self.assigner = TaskAlignedAssigner3D(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=float(getattr(m, "cm3d_alpha", 0.5)),
            beta=float(getattr(m, "cm3d_beta", 1.0)),
            gamma=self.cm3d_config.gamma,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.cm3d_quality = CubeMGIoUQualityBuilder(m, self.cm3d_config)
        self.head = m
        self.cube_loss = CubeLoss(m)
        self.loss_names = (*self.loss_names, "loss_3d")
        self.is_one2one = is_one2one

    def get_assigned_targets_and_loss(self, preds: dict, batch: dict) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_px = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        anchor_points_px = anchor_points * stride_tensor
        quality3d = self.cm3d_quality(
            branch=preds,
            batch=batch,
            pred_boxes_px=pred_bboxes_px,
            anchor_points_px=anchor_points_px,
            gt_labels=gt_labels,
            gt_boxes2d=gt_bboxes,
            mask_gt=mask_gt,
            assigner=self.assigner,
        )
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_px,
            anchor_points_px,
            gt_labels, gt_bboxes, mask_gt,
            quality3d=quality3d,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )
        else:
            loss[0] += pred_distri[..., :0].sum()

        loss[0] *= self.hyp.box if hasattr(self.hyp, "box") else self.hyp["box"]
        loss[1] *= self.hyp.cls if hasattr(self.hyp, "cls") else self.hyp["cls"]
        loss[2] *= self.hyp.dfl if hasattr(self.hyp, "dfl") else self.hyp["dfl"]

        assigned_tuple = (
            fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor,
            target_scores, target_scores_sum, pred_bboxes_px
        )
        return assigned_tuple, loss, loss.detach()

    def set_cm3d_gamma(self, gamma: float) -> None:
        """Public hook for epoch/iteration-based CM3D warm-up."""
        self.assigner.set_gamma(gamma)

    def parse_output(self, preds):
        return preds[-1] if isinstance(preds, tuple) else preds

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch = preds["one2one"] if self.is_one2one else preds["one2many"]

        (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor,
             target_scores, target_scores_sum, pred_bboxes_px),
            det_loss, _,
        ) = self.get_assigned_targets_and_loss(branch, batch)

        batch_size = branch["boxes"].shape[0]
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, 3d
        loss[0], loss[1], loss[2] = det_loss[0], det_loss[1], det_loss[2]
        loss_items = dict(zip(self.loss_names[:3], det_loss.detach()))

        # compute 3d loss
        if fg_mask.sum():
            required_cube_keys = (
                "batch_idx", "cls", "gt_boxes3D", "gt_poses", "gt_2D",
                "K", "im_scales", "im_scales_orig",
            )
            missing_cube_keys = [key for key in required_cube_keys if key not in batch]
            if missing_cube_keys:
                raise KeyError(f"Cube R-CNN batch contract is missing fields: {missing_cube_keys}")
            weight = target_scores[fg_mask].sum(-1, keepdim=True)  # (n_fg, 1)

            dense_cube_preds = branch["cube_preds"]
            cube_preds = {
                name: value[fg_mask]
                for name, value in dense_cube_preds.items()
            }

            fg_batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).repeat(1, fg_mask.shape[1])[fg_mask]
            fg_gt_idx = target_gt_idx[fg_mask]

            batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
            gt_counts = torch.bincount(batch_idx, minlength=batch_size)
            gt_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            gt_offsets[1:] = gt_counts.cumsum(0)
            flat_gt_idx = gt_offsets[fg_batch_idx] + fg_gt_idx
            num_flat_gt = batch["gt_boxes3D"].shape[0]
            if flat_gt_idx.numel() and (flat_gt_idx.min() < 0 or flat_gt_idx.max() >= num_flat_gt):
                raise RuntimeError(
                    "TAL target_gt_idx cannot be mapped to flattened Cube GT: "
                    f"range=[{int(flat_gt_idx.min())}, {int(flat_gt_idx.max())}], num_gt={num_flat_gt}"
                )

            gt_box3d = batch["gt_boxes3D"].to(self.device)[flat_gt_idx]
            gt_pose = batch["gt_poses"].to(self.device)[flat_gt_idx]
            gt_2d = batch["gt_2D"].to(self.device)[flat_gt_idx]
            box_classes = batch["cls"].to(self.device).long().view(-1)[flat_gt_idx]

            # K is already transformed into the current LetterBox input coordinates.
            Ks = batch["K"].to(self.device)[fg_batch_idx].clone()
            Ks[:, -1, -1] = 1.0

            # Original focal space is used only by virtual-depth conversion.
            orig_Ks = batch.get("K_orig", batch["K"]).to(self.device)[fg_batch_idx]
            focal_lengths = orig_Ks[:, 1, 1]

            im_scales = batch["im_scales"].to(self.device)[fg_batch_idx]
            im_scales_orig = batch["im_scales_orig"].to(self.device)[fg_batch_idx]

            if self.head.virtual_depth:
                virtual_to_real = cubeutil.compute_virtual_scale_from_focal_spaces(
                    focal_lengths,
                    im_scales_orig,
                    self.head.virtual_focal,
                    im_scales
                )
                real_to_virtual = 1.0 / virtual_to_real
            else:
                virtual_to_real = real_to_virtual = torch.ones_like(focal_lengths)

            # 修正：使用 pred_bboxes_px[fg_mask] 作為 src_boxes，與 QualityBuilder 保持完全一致
            src_boxes = pred_bboxes_px[fg_mask]

            cube_decoded = self.head.decode_cube(
                cube_preds=cube_preds, 
                box_classes=box_classes, 
                src_boxes=src_boxes,
                Ks_scaled_per_box=Ks, 
                focal_lengths=focal_lengths, 
                im_scales_orig=im_scales_orig, 
                im_scales=im_scales,
            )

            l_3d, det_3d_items = self.cube_loss(
                cube_preds=cube_preds, cube_decoded=cube_decoded, gt_box3d=gt_box3d, gt_pose=gt_pose,
                gt_2d=gt_2d, src_boxes=src_boxes, box_classes=box_classes, Ks=Ks,
                weight=weight, target_scores_sum=target_scores_sum, real_to_virtual=real_to_virtual,
            )
            loss[3] = l_3d
            loss_items.update(det_3d_items)
            loss_items["loss_3d"] = l_3d.detach()
        else:
            loss[3] += sum(
                value.sum() * 0.0
                for value in branch["cube_preds"].values()
            )
            loss_items["loss_3d"] = torch.tensor(0.0, device=self.device)

        return loss * batch_size, loss_items

    def to(self, device):
        self.device = torch.device(device)
        return self

class E2EDetect3DLoss:
    """Criterion class for end-to-end (one2many + one2one dual head) 3D detection."""

    def __init__(self, model: nn.Module):
        self.one2many = Detect3DLoss(model, tal_topk=10, is_one2one=False)
        self.one2one = Detect3DLoss(model, tal_topk=1, is_one2one=True)
        
        self.updates = 0
        self.total = 1.0

        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m

        self.final_o2m = 0.1

    def __call__(
        self, 
        preds: dict[str, Any] | tuple[torch.Tensor, dict[str, Any]], 
        batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        preds = self.one2many.parse_output(preds)

        loss_one2many, items_one2many = self.one2many.loss(preds, batch)
        loss_one2one, items_one2one = self.one2one.loss(preds, batch)

        total_loss = loss_one2many * self.o2m + loss_one2one * self.o2o

        combined_items = {f"o2m_{k}": v for k, v in items_one2many.items()}
        combined_items.update({f"o2o_{k}": v for k, v in items_one2one.items()})

        return total_loss, combined_items