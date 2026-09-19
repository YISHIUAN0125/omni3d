from typing import Dict, Optional, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def bbox_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        mode = "ciou",
        eps: float = 1e-7,
) -> torch.Tensor:

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    b1_w = b1_x2 - b1_x1
    b1_h = b1_y2 - b1_y1
    b2_w = b2_x2 - b2_x1
    b2_h = b2_y2 - b2_y1

    # Area
    b1_area = (b1_w).clamp(min=0) * (b1_h).clamp(min=0)
    b2_area = (b2_w).clamp(min=0) * (b2_h).clamp(min=0)

    # Interaction
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # Inter area
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union area
    union_area = b1_area + b2_area -inter_area

    # IoU
    iou = inter_area / (union_area + eps)

    # Diagonal length
    outer_x1 = torch.min(b1_x1, b2_x1)
    outer_y1 = torch.min(b1_y1, b2_y1)
    outer_x2 = torch.max(b1_x2, b2_x2)
    outer_y2 = torch.max(b1_y2, b2_y2)

    c2 = (outer_x2 - outer_x1).pow(2) + (outer_y2 - outer_y1).pow(2) + eps
    rho2 = (((b2_x1 + b2_x2) / 2) - ((b1_x1 + b1_x2) / 2)).pow(2) + (((b2_y1 + b2_y2) / 2) - ((b1_y1 + b1_y2) / 2)).pow(2)

    v = (4 / math.pi**2) * ((b2_w / b2_h).atan() - (b1_w / b1_h).atan()).pow(2)

    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    ciou = iou - (rho2 / c2 + v * alpha)
    diou  = iou - rho2 / c2

    return {"iou": iou, "diou": diou, "ciou": ciou}[mode]


def pairwise_bbox_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    mode: str = "iou",
    eps: float = 1e-7,
) -> torch.Tensor:

    # boxes1: (N, 1), boxes2: (1, M)
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0].unsqueeze(0), boxes2[:, 1].unsqueeze(0), boxes2[:, 2].unsqueeze(0), boxes2[:, 3].unsqueeze(0)

    b1_w = (b1_x2 - b1_x1).clamp(min=0)
    b1_h = (b1_y2 - b1_y1).clamp(min=0)
    b2_w = (b2_x2 - b2_x1).clamp(min=0)
    b2_h = (b2_y2 - b2_y1).clamp(min=0)

    b1_area = b1_w * b1_h
    b2_area = b2_w * b2_h

    # (N, M)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union_area = b1_area + b2_area - inter_area

    iou = inter_area / (union_area + eps)
    if mode == "iou":
        return iou

    outer_x1 = torch.min(b1_x1, b2_x1)
    outer_y1 = torch.min(b1_y1, b2_y1)
    outer_x2 = torch.max(b1_x2, b2_x2)
    outer_y2 = torch.max(b1_y2, b2_y2)

    c2 = (outer_x2 - outer_x1).pow(2) + (outer_y2 - outer_y1).pow(2) + eps
    
    rho2 = (
        (((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) / 2).pow(2)
        + (((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) / 2).pow(2)
    )

    diou = iou - (rho2 / c2)
    if mode == "diou":
        return diou

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(b2_w / (b2_h + eps)) - torch.atan(b1_w / (b1_h + eps)), 2)

    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    ciou = diou - (v * alpha)
    return ciou

def bbox_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:

    loss = 1.0 - bbox_iou(boxes1, boxes2)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()

    return loss

class TSREvidenceLoss(nn.Module):
    """
    Optional regularization loss for TSR evidence maps.

    This is NOT the main supervision for TSR.

    Main supervision:
        depth loss -> depth evidence route
        dim loss   -> dim evidence route
        pose loss  -> pose evidence route

    This loss only prevents undesirable map degeneration.

    ROI input format:
        evidence_maps: [B, T, 1, H, W]

    Dense input format:
        evidence_maps: [B, T, 1, H, W]

    ROI mode:
        Evidence maps are spatial-softmax distributions.
        Diversity and minimum-entropy regularization are supported.

    Dense mode:
        Maps are independent sigmoid gates.
        Spatial-softmax entropy assumptions do not apply.
        Regularization is disabled by default.
    """

    def __init__(
        self,
        similarity_margin: float = 0.95,
        min_normalized_entropy: float = 0.20,
        diversity_weight: float = 1.0,
        entropy_weight: float = 0.0,
        dense_balance_weight: float = 0.0,
        dense_target_mean: float = 0.80,
        dense_total_variation_weight: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if not 0.0 <= similarity_margin <= 1.0:
            raise ValueError(
                "similarity_margin must be between 0 and 1."
            )

        if not 0.0 <= min_normalized_entropy <= 1.0:
            raise ValueError(
                "min_normalized_entropy must be between 0 and 1."
            )

        if not 0.0 <= dense_target_mean <= 1.0:
            raise ValueError(
                "dense_target_mean must be between 0 and 1."
            )

        if diversity_weight < 0.0:
            raise ValueError("diversity_weight cannot be negative.")

        if entropy_weight < 0.0:
            raise ValueError("entropy_weight cannot be negative.")

        if dense_balance_weight < 0.0:
            raise ValueError(
                "dense_balance_weight cannot be negative."
            )

        if dense_total_variation_weight < 0.0:
            raise ValueError(
                "dense_total_variation_weight cannot be negative."
            )

        self.similarity_margin = float(similarity_margin)
        self.min_normalized_entropy = float(
            min_normalized_entropy
        )

        self.diversity_weight = float(diversity_weight)
        self.entropy_weight = float(entropy_weight)

        self.dense_balance_weight = float(
            dense_balance_weight
        )
        self.dense_target_mean = float(dense_target_mean)

        self.dense_total_variation_weight = float(
            dense_total_variation_weight
        )

        self.eps = float(eps)

    def _validate_maps(
        self,
        evidence_maps: torch.Tensor,
    ) -> None:
        if not isinstance(evidence_maps, torch.Tensor):
            raise TypeError(
                "evidence_maps must be a torch.Tensor."
            )

        if evidence_maps.ndim != 5:
            raise ValueError(
                "Expected evidence_maps with shape "
                "[B, T, 1, H, W], but received "
                f"{tuple(evidence_maps.shape)}."
            )

        if evidence_maps.shape[2] != 1:
            raise ValueError(
                "The channel dimension of each evidence map "
                "must equal 1."
            )

        if evidence_maps.shape[-2] <= 0:
            raise ValueError("H must be positive.")

        if evidence_maps.shape[-1] <= 0:
            raise ValueError("W must be positive.")

        if not torch.isfinite(evidence_maps).all():
            raise ValueError(
                "evidence_maps contains NaN or Inf."
            )

    def _roi_diversity_loss(
        self,
        probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            probabilities: [B, T, HW]

        Returns:
            loss_diversity
            mean_off_diagonal_similarity
        """
        batch_size, num_tasks, _ = probabilities.shape

        if num_tasks <= 1:
            zero = probabilities.sum() * 0.0
            return zero, zero.detach()

        normalized_maps = F.normalize(
            probabilities,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        similarity = torch.bmm(
            normalized_maps,
            normalized_maps.transpose(1, 2),
        )

        # Only select the upper triangular entries.
        # This avoids counting both (depth, dim) and (dim, depth).
        upper_mask = torch.triu(
            torch.ones(
                num_tasks,
                num_tasks,
                dtype=torch.bool,
                device=similarity.device,
            ),
            diagonal=1,
        )

        off_diagonal_similarity = similarity[:, upper_mask]

        loss_diversity = F.relu(
            off_diagonal_similarity
            - self.similarity_margin
        ).mean()

        mean_similarity = (
            off_diagonal_similarity.mean().detach()
        )

        return loss_diversity, mean_similarity

    def _roi_entropy_loss(
        self,
        probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Minimum normalized entropy constraint.

        This does not maximize entropy indefinitely.

        It only penalizes evidence maps whose normalized entropy
        drops below min_normalized_entropy.

        Args:
            probabilities: [B, T, HW]

        Returns:
            loss_entropy
            mean_normalized_entropy
        """
        num_positions = probabilities.shape[-1]

        entropy = -(
            probabilities
            * torch.log(probabilities.clamp_min(self.eps))
        ).sum(dim=-1)

        if num_positions <= 1:
            normalized_entropy = torch.ones_like(entropy)
        else:
            max_entropy = torch.log(
                probabilities.new_tensor(
                    float(num_positions)
                )
            ).clamp_min(self.eps)

            normalized_entropy = entropy / max_entropy

        loss_entropy = F.relu(
            self.min_normalized_entropy
            - normalized_entropy
        ).mean()

        mean_normalized_entropy = (
            normalized_entropy.mean().detach()
        )

        return (
            loss_entropy,
            mean_normalized_entropy,
        )

    def _forward_roi(
        self,
        evidence_maps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_tasks, _, height, width = (
            evidence_maps.shape
        )

        flat_maps = evidence_maps.reshape(
            batch_size,
            num_tasks,
            height * width,
        )

        # TSR ROI maps already come from spatial softmax.
        # Renormalization is kept for numerical safety.
        probabilities = flat_maps.clamp_min(self.eps)

        probabilities = probabilities / (
            probabilities.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
        )

        loss_diversity, mean_similarity = (
            self._roi_diversity_loss(probabilities)
        )

        loss_entropy, mean_entropy = (
            self._roi_entropy_loss(probabilities)
        )

        total_loss = (
            self.diversity_weight * loss_diversity
            + self.entropy_weight * loss_entropy
        )

        return {
            # Only this tensor should be added to training loss.
            "loss_tsr_regularizer": total_loss,

            # The following are detached monitoring values.
            "loss_tsr_diversity": (
                loss_diversity.detach()
            ),
            "loss_tsr_entropy": (
                loss_entropy.detach()
            ),
            "tsr_map_similarity": mean_similarity,
            "tsr_normalized_entropy": mean_entropy,
            "tsr_map_max": (
                probabilities.max(dim=-1).values
                .mean()
                .detach()
            ),
            "tsr_map_min": (
                probabilities.min(dim=-1).values
                .mean()
                .detach()
            ),
        }

    def _dense_total_variation(
        self,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional spatial smoothness regularizer for dense sigmoid gates.

        gates: [B, T, 1, H, W]
        """
        height = gates.shape[-2]
        width = gates.shape[-1]

        zero = gates.sum() * 0.0

        if height > 1:
            variation_y = (
                gates[..., 1:, :]
                - gates[..., :-1, :]
            ).abs().mean()
        else:
            variation_y = zero

        if width > 1:
            variation_x = (
                gates[..., :, 1:]
                - gates[..., :, :-1]
            ).abs().mean()
        else:
            variation_x = zero

        return variation_x + variation_y

    def _forward_dense(
        self,
        evidence_maps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Dense mode maps are sigmoid gates, not spatial distributions.

        Therefore:
        - no spatial entropy loss
        - no forced task orthogonality by default

        Optional constraints:
        - gate mean warm-start regularization
        - total variation
        """
        gate_mean_per_task = evidence_maps.mean(
            dim=(0, 2, 3, 4)
        )

        target = torch.full_like(
            gate_mean_per_task,
            self.dense_target_mean,
        )

        loss_balance = F.smooth_l1_loss(
            gate_mean_per_task,
            target,
        )

        loss_total_variation = (
            self._dense_total_variation(evidence_maps)
        )

        total_loss = (
            self.dense_balance_weight * loss_balance
            + self.dense_total_variation_weight
            * loss_total_variation
        )

        return {
            "loss_tsr_regularizer": total_loss,
            "loss_tsr_dense_balance": (
                loss_balance.detach()
            ),
            "loss_tsr_dense_tv": (
                loss_total_variation.detach()
            ),
            "tsr_dense_gate_mean": (
                gate_mean_per_task.mean().detach()
            ),
            "tsr_dense_gate_min": (
                evidence_maps.min().detach()
            ),
            "tsr_dense_gate_max": (
                evidence_maps.max().detach()
            ),
        }

    def forward(
        self,
        evidence_maps: torch.Tensor,
        mode: str = "roi",
    ) -> Dict[str, torch.Tensor]:
        self._validate_maps(evidence_maps)

        if mode == "roi":
            return self._forward_roi(evidence_maps)

        if mode == "dense":
            return self._forward_dense(evidence_maps)

        raise ValueError(
            f"Unsupported mode={mode!r}. "
            "Expected 'roi' or 'dense'."
        )
