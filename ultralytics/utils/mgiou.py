"""CM3D utilities for Cube-based monocular 3D detection.
This module deliberately owns only the detached assignment-quality path.
The differentiable Cube regression path remains in CubeLoss.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from cubercnn import util as cubeutil

try:
    from mgiou import MGIoU3D
except ImportError as exc:
    raise ImportError(
        "CM3D requires the official MGIoU package. Install it using "
        "`pip install mgiou` or install https://github.com/ldtho/MGIoU "
        "in editable mode."
    ) from exc


@dataclass(frozen=True)
class CM3DConfig:
    """Configuration for MGIoU-aware task-aligned assignment."""
    gamma: float = 1.0
    quality_floor: float = 0.05
    fast_mode: bool = True
    corner_order: tuple[int, ...] | None = None


class CubeMGIoUQualityBuilder(nn.Module):
    """Build sparse pairwise MGIoU quality for TAL without autograd.

    Output shape is ``(B, M, A)`` where B is batch size, M is padded GT
    count, and A is the total number of P3-P5 locations. Only anchor-GT pairs
    whose anchor centers lie inside the GT 2D box are decoded and evaluated.
    """

    def __init__(self, head: nn.Module, config: CM3DConfig | None = None):
        super().__init__()
        self.head = head
        self.config = config or CM3DConfig()
        self.mgiou = MGIoU3D(reduction="none", fast_mode=self.config.fast_mode)

        if self.config.corner_order is None:
            self.register_buffer("corner_order", torch.empty(0, dtype=torch.long), persistent=False)
        else:
            order = torch.as_tensor(self.config.corner_order, dtype=torch.long)
            if order.numel() != 8 or order.unique().numel() != 8:
                raise ValueError("corner_order must be a permutation of eight corner indices")
            self.register_buffer("corner_order", order, persistent=False)

    @staticmethod
    def _pack_flat_gt(batch: dict, batch_size: int, max_gt: int, device: torch.device):
        """Pack flat YOLO targets into TAL-compatible batch-major tensors."""
        batch_idx = batch["batch_idx"].view(-1).long().to(device)
        counts = torch.bincount(batch_idx, minlength=batch_size)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        offsets[1:] = counts.cumsum(0)
        within_idx = torch.arange(batch_idx.numel(), device=device) - offsets[batch_idx]

        # 動態相容 Omni3D 的 9 維 ([x, y, z, w, h, l, r, p, y]) 或 6 維 3D box 標籤
        gt_dim = batch["gt_boxes3d"].shape[-1] if batch["gt_boxes3d"].numel() else 6
        boxes3d = torch.zeros(batch_size, max_gt, gt_dim, device=device)
        poses = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(batch_size, max_gt, 1, 1)
        centers2d = torch.zeros(batch_size, max_gt, 2, device=device)
        
        if batch_idx.numel():
            boxes3d[batch_idx, within_idx] = batch["gt_boxes3d"].to(device)
            poses[batch_idx, within_idx] = batch["gt_poses"].to(device)
            centers2d[batch_idx, within_idx] = batch["gt_2d"].to(device)
        return boxes3d, poses, centers2d

    def _reorder(self, corners: torch.Tensor) -> torch.Tensor:
        if self.corner_order.numel() == 0:
            return corners
        return corners.index_select(1, self.corner_order.to(corners.device))

    @torch.no_grad()
    def forward(
        self,
        *,
        branch: dict,
        batch: dict,
        pred_boxes_px: torch.Tensor,
        anchor_points_px: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes2d: torch.Tensor,
        mask_gt: torch.Tensor,
        assigner: nn.Module,
    ) -> torch.Tensor:
        """Return detached 3D quality in ``[quality_floor, 1]``."""
        device = pred_boxes_px.device
        batch_size, num_anchors = pred_boxes_px.shape[:2]
        max_gt = gt_boxes2d.shape[1]
        floor = max(float(self.config.quality_floor), float(assigner.eps))
        quality = torch.full(
            (batch_size, max_gt, num_anchors),
            floor,
            dtype=pred_boxes_px.dtype,
            device=device,
        )
        if max_gt == 0 or not bool(mask_gt.any()):
            return quality

        candidate_mask = assigner.select_candidates_in_gts(
            anchor_points_px, gt_boxes2d, mask_gt
        ).bool()
        batch_idx, gt_idx, anchor_idx = candidate_mask.nonzero(as_tuple=True)
        if batch_idx.numel() == 0:
            return quality

        pair_preds = {
            name: value.detach()[batch_idx, anchor_idx]
            for name, value in branch["cube_preds"].items()
        }
        pair_classes = gt_labels[batch_idx, gt_idx, 0].long()
        pair_boxes2d = pred_boxes_px[batch_idx, anchor_idx]

        original_k = batch["Ks"].to(device)[batch_idx]
        scale_ratio = batch["im_scales_ratio"].to(device)[batch_idx]
        scaled_k = original_k / scale_ratio.view(-1, 1, 1)
        scaled_k[:, -1, -1] = 1.0

        decoded = self.head.decode_cube(
            cube_preds=pair_preds,
            box_classes=pair_classes,
            src_boxes=pair_boxes2d,
            Ks_scaled_per_box=scaled_k,
            focal_lengths=original_k[:, 1, 1],
            im_scales_orig=batch["im_scales_orig"].to(device)[batch_idx],
            im_scales=batch["im_scales"].to(device)[batch_idx],
        )
        pred_box3d = torch.cat([decoded["center_cam"], decoded["dims"]], dim=-1)
        pred_corners = cubeutil.get_cuboid_verts_faces(pred_box3d, decoded["pose"])[0]

        packed_boxes, packed_poses, packed_centers = self._pack_flat_gt(
            batch, batch_size, max_gt, device
        )
        pair_gt = packed_boxes[batch_idx, gt_idx]
        pair_gt_pose = packed_poses[batch_idx, gt_idx]
        pair_gt_center2d = packed_centers[batch_idx, gt_idx]
        gt_z = pair_gt[:, 2]
        gt_x = gt_z * (pair_gt_center2d[:, 0] - scaled_k[:, 0, 2]) / scaled_k[:, 0, 0]
        gt_y = gt_z * (pair_gt_center2d[:, 1] - scaled_k[:, 1, 2]) / scaled_k[:, 1, 1]
        
        # 精準取 pair_gt[:, 3:6] 作為 [w, h, l]，無論 GT 是 6 維或 9 維皆相容
        gt_box_cam = torch.cat(
            [torch.stack([gt_x, gt_y, gt_z], dim=-1), pair_gt[:, 3:6]], dim=-1
        )
        gt_corners = cubeutil.get_cuboid_verts_faces(gt_box_cam, pair_gt_pose)[0]

        pred_corners = self._reorder(pred_corners).float()
        gt_corners = self._reorder(gt_corners).float()
        mgiou_loss = self.mgiou(pred_corners, gt_corners)
        if mgiou_loss.ndim == 0:
            raise RuntimeError(
                "MGIoU3D returned a scalar. Use an official MGIoU version where "
                "reduction='none' returns one value per prediction-GT pair."
            )
        mgiou_loss = mgiou_loss.reshape(-1)
        if mgiou_loss.numel() != batch_idx.numel():
            raise RuntimeError(
                f"MGIoU returned {mgiou_loss.numel()} values for "
                f"{batch_idx.numel()} candidate pairs"
            )
        pair_quality = (1.0 - mgiou_loss).clamp(min=floor, max=1.0)
        pair_quality = torch.nan_to_num(pair_quality, nan=floor, posinf=1.0, neginf=floor)
        quality[batch_idx, gt_idx, anchor_idx] = pair_quality.to(quality.dtype)
        return quality