import math
import torch
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
