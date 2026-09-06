from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from cubercnn.modeling.losses import pairwise_bbox_iou

SIMOTA_ASSIGNER_REGISTRY = Registry("SIMOTA_ASSIGNER")
TASKALIGNED_ASSIGNER_REGISTRY = Registry("TASKALIGNED_ASSIGNER")

def build_assigner(cfg):
    name = cfg.MODEL.DENSE_HEAD.ASSIGNER
    return SIMOTA_ASSIGNER_REGISTRY.get(name)(cfg)

@SIMOTA_ASSIGNER_REGISTRY.register()
class SimOTAAssigner:

    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.DENSE_HEAD.NUM_CLASSES
        self.center_radius = cfg.MODEL.DENSE_HEAD.CENTER_RADIUS
        self.candidate_topk = cfg.MODEL.DENSE_HEAD.CANDIDATE_TOPK
        self.cls_cost_weight = cfg.MODEL.DENSE_HEAD.CLS_COST_WEIGHT
        self.iou_cost_weight = cfg.MODEL.DENSE_HEAD.IOU_COST_WEIGHT

    @torch.no_grad()
    def assign(self,
               cls_preds_i: torch.Tensor,
               box_preds_i: torch.Tensor,
               points: torch.Tensor,
               strides: torch.Tensor,
               gt_boxes_i: torch.Tensor,
               gt_labels_i: torch.Tensor,
               gt_boxes_ign_i: Optional[torch.Tensor]=None,
               obj_preds_i: Optional[torch.Tensor] = None,
               ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_points = points.shape[0]
        device = points.device

        assigned_labels = torch.full((num_points,), self.num_classes, dtype=torch.long, device=device)
        assigned_gt_inds = torch.full((num_points,), -1, dtype=torch.long, device=device)

        num_gt = gt_boxes_i.shape[0]
        if num_gt == 0:
            self._apply_ignore(assigned_labels, points, gt_boxes_ign_i)
            return assigned_labels, assigned_gt_inds

        # Points in box 
        is_in_boxes, is_in_centers = self._get_geometry_constraint(points, strides, gt_boxes_i)
        is_in_boxes_and_centers = is_in_boxes | is_in_centers # TODO Use & or |
        candidate_mask = is_in_boxes_and_centers.any(dim=1)

        if candidate_mask.sum() == 0:
            self._apply_ignore(assigned_labels, points, gt_boxes_ign_i)
            return assigned_labels, assigned_gt_inds

        cand_idx = candidate_mask.nonzero(as_tuple=True)[0]
        cls_preds_c = cls_preds_i[cand_idx]
        box_preds_c = box_preds_i[cand_idx]

        # Cost matrix
        pair_iou = retry_if_cuda_oom(pairwise_iou)(Boxes(box_preds_c), Boxes(gt_boxes_i))
        iou_cost = -torch.log(pair_iou.clamp(min=1e-8))

        num_cand = cls_preds_c.shape[0]
        cls_preds_sig = cls_preds_c.sigmoid().unsqueeze(1).repeat(1, num_gt, 1)
        gt_onehot = F.one_hot(gt_labels_i, self.num_classes).float().unsqueeze(0).repeat(num_cand, 1, 1)

        if obj_preds_i is not None:
            obj_preds_sig = obj_preds_i[cand_idx].sigmoid().unsqueeze(1).repeat(1, num_gt, 1)
            score = torch.sqrt(cls_preds_sig * obj_preds_sig)
            cls_cost = F.binary_cross_entropy(score, gt_onehot, reduction="none").sum(-1)
        else:
            cls_cost = F.binary_cross_entropy(cls_preds_sig, gt_onehot, reduction="none").sum(-1)

        cost = self.cls_cost_weight * cls_cost + self.iou_cost_weight * iou_cost

        # Dynamic K
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        topk = min(self.candidate_topk, pair_iou.shape[0])
        topk_ious, _ = torch.topk(pair_iou, topk, dim=0)
        dynamic_ks = topk_ious.sum(0).clamp(min=1).long()

        for gt_idx in range(num_gt):
            k = min(int(dynamic_ks[gt_idx].item()), cost.shape[0])
            _, pos_idx = torch.topk(cost[:, gt_idx], k, largest=False)
            matching_matrix[pos_idx, gt_idx] = 1

        # Solving confict
        # TODO Test SimOTA Assignment
        # multi_match = matching_matrix.sum(1) > 1
        # if multi_match.sum() > 0:
        #     min_cost_gt = cost[multi_match].argmin(1)
        #     matching_matrix[multi_match] = 0
        #     matching_matrix[multi_match, min_cost_gt] = 1
        multi_match = matching_matrix.sum(1) > 1
        if multi_match.any():
            multi_match_idx = multi_match.nonzero(as_tuple=True)[0]
            min_cost_gt = cost[multi_match_idx].argmin(dim=1)
            matching_matrix[multi_match_idx] = 0
            matching_matrix[multi_match_idx, min_cost_gt] = 1

        fg_mask_c = matching_matrix.sum(1) > 0
        matched_gt_inds_c = matching_matrix[fg_mask_c].argmax(1)

        fg_global_idx = cand_idx[fg_mask_c]
        assigned_labels[fg_global_idx] = gt_labels_i[matched_gt_inds_c]
        assigned_gt_inds[fg_global_idx] = matched_gt_inds_c

        # Ignore the points background and ignore region
        self._apply_ignore(assigned_labels, points, gt_boxes_ign_i)

        return assigned_labels, assigned_gt_inds


    def _get_geometry_constraint(self, points, strides, gt_boxes_i):
        x, y = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1) # (P, 1)

        l = x - gt_boxes_i[:, 0].unsqueeze(0)
        t = y - gt_boxes_i[:, 1].unsqueeze(0)
        r = gt_boxes_i[:, 2].unsqueeze(0) - x
        b = gt_boxes_i[:, 3].unsqueeze(0) - y
        is_in_boxes = torch.stack([l, t, r, b], dim=-1).min(-1)[0] > 0 # (P, G)

        gt_cx = (gt_boxes_i[:, 0] + gt_boxes_i[:, 2]) * 0.5
        gt_cy = (gt_boxes_i[:, 1] + gt_boxes_i[:, 3]) * 0.5
        radius = strides.unsqueeze(1) * self.center_radius
        cl = x - (gt_cx.unsqueeze(0) - radius)
        ct = y - (gt_cy.unsqueeze(0) - radius)
        cr = (gt_cx.unsqueeze(0) + radius) - x
        cb = (gt_cy.unsqueeze(0) + radius) - y
        is_in_centers = torch.stack([cl, ct, cr, cb], dim=-1).min(-1)[0] > 0 # (P, G)

        return is_in_boxes, is_in_centers

    def _apply_ignore(self, assigned_labels, points, gt_boxes_ign_i):
        # Marks background points that fall inside a 'dontcare' GT box as ignore (-1)
        if gt_boxes_ign_i is None or gt_boxes_ign_i.shape[0] == 0:
            return
        bg_mask = assigned_labels == self.num_classes
        if bg_mask.sum() == 0:
            return
        
        bg_idx = bg_mask.nonzero(as_tuple=True)[0]

        px = points[bg_idx, 0].unsqueeze(1)
        py = points[bg_idx, 1].unsqueeze(1)

        inside = (
            (px >= gt_boxes_ign_i[:, 0].unsqueeze(0))
            & (px <= gt_boxes_ign_i[:, 2].unsqueeze(0))
            & (py >= gt_boxes_ign_i[:, 1].unsqueeze(0))
            & (py <= gt_boxes_ign_i[:, 3].unsqueeze(0))
        )

        assigned_labels[bg_idx[inside.any(dim=1)]] = -1


def build_tal_assigner(cfg):
    name = cfg.MODEL.DENSE_HEAD.ASSIGNER
    return TASKALIGNED_ASSIGNER_REGISTRY.get(name)(cfg)

@TASKALIGNED_ASSIGNER_REGISTRY.register()
class TaskAlignedAssigner:
    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.DENSE_HEAD.NUM_CLASSES
        self.topk = getattr(cfg.MODEL.DENSE_HEAD, "TAL_TOPK", 13)
        self.alpha = getattr(cfg.MODEL.DENSE_HEAD, "TAL_ALPHA", 1.0)
        self.beta = getattr(cfg.MODEL.DENSE_HEAD, "TAL_BETA", 6.0)
        self.eps = 1e-9

    @torch.no_grad()
    def assign(self,
               cls_preds_i: torch.Tensor,
               box_preds_i: torch.Tensor,
               points: torch.Tensor,
               strides: torch.Tensor,
               gt_boxes_i: torch.Tensor,
               gt_labels_i: torch.Tensor,
               gt_boxes_ign_i: Optional[torch.Tensor] = None,
               obj_preds_i: Optional[torch.Tensor] = None,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_points = points.shape[0]
        device = points.device

        assigned_labels = torch.full((num_points,), self.num_classes, dtype=torch.long, device=device)
        assigned_gt_inds = torch.full((num_points,), -1, dtype=torch.long, device=device)
        target_scores = torch.zeros((num_points, self.num_classes), dtype=torch.float32, device=device)

        num_gt = gt_boxes_i.shape[0]
        if num_gt == 0:
            self._apply_ignore(assigned_labels, points, gt_boxes_ign_i)
            return assigned_labels, assigned_gt_inds, target_scores

        is_in_gts = self._get_geometry_constraint(points, gt_boxes_i)
        candidate_mask = is_in_gts.any(dim=1)

        if candidate_mask.sum() == 0:
            self._apply_ignore(assigned_labels, points, gt_boxes_ign_i)
            return assigned_labels, assigned_gt_inds, target_scores

        cand_idx = candidate_mask.nonzero(as_tuple=True)[0]
        cls_preds_c = cls_preds_i[cand_idx].sigmoid()
        box_preds_c = box_preds_i[cand_idx]
        is_in_gts_c = is_in_gts[cand_idx]

        # pair_iou = retry_if_cuda_oom(pairwise_iou)(Boxes(box_preds_c), Boxes(gt_boxes_i)).clamp_(0)
        pair_iou = retry_if_cuda_oom(pairwise_bbox_iou)(box_preds_c, gt_boxes_i, mode="ciou").clamp_(0)

        bbox_scores = cls_preds_c[:, gt_labels_i] 
        if obj_preds_i is not None:
            obj_preds_c = obj_preds_i[cand_idx].sigmoid()
            bbox_scores *= obj_preds_c

        align_metric = (bbox_scores.pow(self.alpha)) * (pair_iou.pow(self.beta))
        align_metric *= is_in_gts_c

        topk = min(self.topk, align_metric.shape[0])
        topk_metrics, topk_idxs = torch.topk(align_metric, topk, dim=0, largest=True) 

        matching_matrix = torch.zeros_like(align_metric, dtype=torch.uint8)
        valid_mask = topk_metrics > self.eps
        for gt_idx in range(num_gt):
            valid_cand_idxs = topk_idxs[:, gt_idx][valid_mask[:, gt_idx]]
            matching_matrix[valid_cand_idxs, gt_idx] = 1

        fg_mask_c = matching_matrix.sum(1) > 0
        if matching_matrix.sum(1).max() > 1:
            _, max_gt_idx = align_metric.max(dim=1) 
            matching_matrix.zero_()
            matching_matrix[fg_mask_c, max_gt_idx[fg_mask_c]] = 1

        # -------------------------------------------------------------
        # 產生正規化的 Target Score (Soft Label)
        # -------------------------------------------------------------
        align_metric *= matching_matrix
        pos_align_metrics = align_metric.max(dim=0, keepdim=True)[0]
        overlaps = pair_iou * matching_matrix
        pos_overlaps = overlaps.max(dim=0, keepdim=True)[0]

        # norm_align_metric = align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        norm_align_metric = align_metric * pos_overlaps.clamp(min=0.5) / (pos_align_metrics + self.eps)
        cand_scores = norm_align_metric.max(dim=1)[0] # (P_cand,)

        matched_gt_inds_c = matching_matrix.argmax(1) 
        fg_mask_c = matching_matrix.sum(1) > 0        

        fg_global_idx = cand_idx[fg_mask_c]
        matched_gts = matched_gt_inds_c[fg_mask_c]

        assigned_labels[fg_global_idx] = gt_labels_i[matched_gts]
        assigned_gt_inds[fg_global_idx] = matched_gts

        target_scores[fg_global_idx, gt_labels_i[matched_gts]] = cand_scores[fg_mask_c]

        self._apply_ignore(assigned_labels, points, gt_boxes_ign_i)

        return assigned_labels, assigned_gt_inds, target_scores

    def _get_geometry_constraint(self, points, gt_boxes_i):
        x, y = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1) 

        l = x - gt_boxes_i[:, 0].unsqueeze(0)
        t = y - gt_boxes_i[:, 1].unsqueeze(0)
        r = gt_boxes_i[:, 2].unsqueeze(0) - x
        b = gt_boxes_i[:, 3].unsqueeze(0) - y
        
        is_in_boxes = torch.stack([l, t, r, b], dim=-1).min(-1)[0] > 0 
        return is_in_boxes

    def _apply_ignore(self, assigned_labels, points, gt_boxes_ign_i):
        if gt_boxes_ign_i is None or gt_boxes_ign_i.shape[0] == 0:
            return
            
        bg_mask = assigned_labels == self.num_classes
        if bg_mask.sum() == 0:
            return
        
        bg_idx = bg_mask.nonzero(as_tuple=True)[0]

        px = points[bg_idx, 0].unsqueeze(1)
        py = points[bg_idx, 1].unsqueeze(1)

        inside = (
            (px >= gt_boxes_ign_i[:, 0].unsqueeze(0))
            & (px <= gt_boxes_ign_i[:, 2].unsqueeze(0))
            & (py >= gt_boxes_ign_i[:, 1].unsqueeze(0))
            & (py <= gt_boxes_ign_i[:, 3].unsqueeze(0))
        )

        assigned_labels[bg_idx[inside.any(dim=1)]] = -1