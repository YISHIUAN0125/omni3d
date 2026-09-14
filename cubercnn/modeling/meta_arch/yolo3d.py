from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import batched_nms
import detectron2.utils.comm as comm

from cubercnn import util as cubeutil
from cubercnn import vis

from ultralytics import YOLO
from ultralytics.utils.loss import E2EDetect3DLoss, Detect3DLoss
from ultralytics.utils import LOGGER

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class YOLO3DWrapper(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        yolo_yaml: str,
        is_e2e: bool,
        input_format: str,
        vis_period: int,
        pixel_mean: List[float],
        pixel_std: List[float],
        pretrained_weights,
        vis_nms_thresh: float = 0.5,
        metadata_name: str = "omni3d_model",
    ):
        super().__init__()
        self.yolo_yaml = yolo_yaml
        self.is_e2e = is_e2e
        self.input_format = input_format
        self.vis_period = vis_period
        self.vis_nms_thresh = vis_nms_thresh
        self.metadata_name = metadata_name

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.yolo_model = YOLO(yolo_yaml).model

        if pretrained_weights:
            self.load_yolo_pretrained(self.yolo_model, pretrained_weights)

        self.criterion = E2EDetect3DLoss(self.yolo_model) if is_e2e else Detect3DLoss(self.yolo_model)
        self.criterion.to(self.device)

    @classmethod
    def from_config(cls, cfg, priors=None):
    # cfg.MODEL.PIXEL_MEAN = [0.0, 0.0, 0.0]
    # cfg.MODEL.PIXEL_STD = [255.0, 255.0, 255.0]
        return {
            "yolo_yaml": cfg.MODEL.YOLO3D.ARCH,
            "is_e2e": cfg.MODEL.YOLO3D.IS_E2E,
            "input_format": cfg.MODEL.INPUT_FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "pretrained_weights": cfg.MODEL.PRETRAINED_WEIGHTS,
            "metadata_name": cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "omni3d_model",
        }

    @staticmethod
    def load_yolo_pretrained(yolo_model: nn.Module, pt_path: str):
        if not pt_path:
            return
        LOGGER.info(f"Loading YOLO Backbone + Neck weights from: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu")
        candidate = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if isinstance(candidate, nn.Module):
            src_sd = candidate.float().state_dict()
        elif isinstance(candidate, dict):
            src_sd = candidate
        else:
            src_sd = ckpt

        target_sd = yolo_model.state_dict()
        head_idx = len(yolo_model.model) - 1
        head_prefix = f"model.{head_idx}."

        filtered_sd = {}
        head_keys_skipped = 0
        mismatched_keys = 0
        for k, v in src_sd.items():
            if k.startswith(head_prefix):
                head_keys_skipped += 1
                continue
            if k in target_sd:
                if target_sd[k].shape == v.shape:
                    filtered_sd[k] = v
                else:
                    mismatched_keys += 1
                    LOGGER.warning(f"Shape mismatch, skip {k}: ckpt {v.shape} vs model {target_sd[k].shape}")

        yolo_model.load_state_dict(filtered_sd, strict=False)
        LOGGER.info(
            f"Transferred {len(filtered_sd)} tensors to Backbone+Neck. "
            f"Skipped {head_keys_skipped} Head tensors. "
            f"Head parameters remained randomly initialized."
        )

    def set_priors(self, priors: dict):
        head = self.yolo_model.model[-1]
        if hasattr(head, "set_priors"):
            head.set_priors(priors)
        else:
            LOGGER.warning("The model head has no 'set_priors' method implemented.")

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, Any]]):
        images, yolo_batch, pad_hw = self.preprocess_inputs(batched_inputs)

        if self.training:
            preds = self.yolo_model(images)
            total_loss, loss_items = self.criterion(preds, yolo_batch)

            if total_loss.dim() > 0 and total_loss.numel() == 4:
                losses = {
                    "loss_box": total_loss[0],
                    "loss_cls": total_loss[1],
                    "loss_dfl": total_loss[2],
                    "loss_3d": total_loss[3],
                }
            else:
                losses = {"loss_3d_total": total_loss.sum()}

            storage = get_event_storage()
            for k, v in loss_items.items():
                if isinstance(v, torch.Tensor):
                    storage.put_scalar(f"yolo_metrics/{k}", v.item())

            # 週期性可視化：只在主行程做，避免 DDP 每張卡都各自寫一份到 TensorBoard
            if self.vis_period > 0 and comm.is_main_process() and storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, images, pad_hw)

            return losses
        else:
            return self.inference(images, batched_inputs, pad_hw)

    def preprocess_inputs(self, batched_inputs: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, Any], Tuple[int, int]]:
        images_list = []
        max_h, max_w = 0, 0

        for x in batched_inputs:
            img = x["image"].to(self.device).float()
            if self.input_format == "BGR":
                img = img[[2, 1, 0], :, :]
            img = (img - self.pixel_mean) / self.pixel_std
            images_list.append(img)
            max_h = max(max_h, img.shape[1])
            max_w = max(max_w, img.shape[2])

        pad_h = int(np.ceil(max_h / 32.0) * 32)
        pad_w = int(np.ceil(max_w / 32.0) * 32)

        padded_imgs = []
        for img in images_list:
            dh = pad_h - img.shape[1]
            dw = pad_w - img.shape[2]
            padded_imgs.append(F.pad(img, (0, dw, 0, dh), value=0.0))
        images = torch.stack(padded_imgs, dim=0)

        batch_size = len(batched_inputs)
        batch_idx_list, cls_list, bboxes_list = [], [], []
        gt_boxes3d_list, gt_poses_list, gt_2d_list = [], [], []
        Ks_list, im_scales_list, im_scales_orig_list, im_scales_ratio_list = [], [], [], []

        for i, x in enumerate(batched_inputs):
            orig_h = x.get("height", x["image"].shape[1])
            curr_h = x["image"].shape[1]

            Ks_list.append(x["K"].to(self.device) if isinstance(x["K"], torch.Tensor) else torch.as_tensor(x["K"], dtype=torch.float32, device=self.device))
            im_scales_list.append(torch.tensor(curr_h, device=self.device, dtype=torch.float32))
            im_scales_orig_list.append(torch.tensor(orig_h, device=self.device, dtype=torch.float32))
            scale_ratio = float(orig_h) / float(curr_h) if curr_h > 0 else 1.0
            im_scales_ratio_list.append(torch.tensor(scale_ratio, device=self.device, dtype=torch.float32))

            if "instances" in x:
                instances = x["instances"]
                num_inst = len(instances)
                if num_inst > 0:
                    if not (hasattr(instances, "gt_boxes3D") and hasattr(instances, "gt_poses")):
                        raise ValueError(
                            f"batched_inputs[{i}] has {num_inst} 2D instances but is missing "
                            f"3D annotations (gt_boxes3D / gt_poses)."
                        )
                    batch_idx_list.append(torch.full((num_inst,), i, dtype=torch.long, device=self.device))
                    cls_list.append(instances.gt_classes.to(self.device))

                    box = instances.gt_boxes.tensor.to(self.device)
                    x1, y1, x2, y2 = box.unbind(-1)
                    cx = (x1 + x2) * 0.5 / pad_w
                    cy = (y1 + y2) * 0.5 / pad_h
                    bw = (x2 - x1) / pad_w
                    bh = (y2 - y1) / pad_h
                    bboxes_list.append(torch.stack([cx, cy, bw, bh], dim=-1))

                    gt_boxes3d_list.append(instances.gt_boxes3D.to(self.device))
                    gt_poses_list.append(instances.gt_poses.to(self.device))
                    gt_2d_list.append(instances.gt_boxes3D[:, :2].to(self.device))

        yolo_batch = {
            "batch_idx": torch.cat(batch_idx_list, dim=0) if batch_idx_list else torch.empty(0, dtype=torch.long, device=self.device),
            "cls": torch.cat(cls_list, dim=0) if cls_list else torch.empty(0, device=self.device),
            "bboxes": torch.cat(bboxes_list, dim=0) if bboxes_list else torch.empty((0, 4), device=self.device),
            "gt_boxes3d": torch.cat(gt_boxes3d_list, dim=0) if gt_boxes3d_list else torch.empty((0, 6), device=self.device),
            "gt_poses": torch.cat(gt_poses_list, dim=0) if gt_poses_list else torch.empty((0, 3, 3), device=self.device),
            "gt_2d": torch.cat(gt_2d_list, dim=0) if gt_2d_list else torch.empty((0, 2), device=self.device),
            "Ks": torch.stack(Ks_list, dim=0),
            "im_scales": torch.stack(im_scales_list, dim=0),
            "im_scales_orig": torch.stack(im_scales_orig_list, dim=0),
            "im_scales_ratio": torch.stack(im_scales_ratio_list, dim=0),
        }

        return images, yolo_batch, (pad_h, pad_w)

    @torch.no_grad()
    def inference(self, images: torch.Tensor, batched_inputs: List[Dict[str, Any]], pad_hw: Tuple[int, int]) -> List[Dict[str, Instances]]:
        pad_h, pad_w = pad_hw
        head = self.yolo_model.model[-1]
        out = self.yolo_model(images)

        if head.end2end:
            pred_2d, cube_preds, _ = out
        else:
            y, cube_feats, _ = out
            pred_2d, cube_preds = head.postprocess(y.permute(0, 2, 1), cube_feats)

        results = []
        for i, x in enumerate(batched_inputs):
            orig_h = x.get("height", x["image"].shape[1])
            orig_w = x.get("width", x["image"].shape[2])
            curr_h = x["image"].shape[1]
            curr_w = x["image"].shape[2]

            p2d = pred_2d[i]
            valid_mask = p2d[:, 4] > 0.05
            p2d = p2d[valid_mask]

            inst = Instances((orig_h, orig_w))
            if len(p2d) == 0:
                inst.pred_boxes = Boxes(torch.empty((0, 4), device=self.device))
                inst.scores = torch.empty(0, device=self.device)
                inst.pred_classes = torch.empty(0, dtype=torch.long, device=self.device)
                inst.pred_bbox3D = torch.empty((0, 8, 3), device=self.device)
                inst.pred_center_cam = torch.empty((0, 3), device=self.device)
                inst.pred_center_2D = torch.empty((0, 2), device=self.device)
                inst.pred_dimensions = torch.empty((0, 3), device=self.device)
                inst.pred_pose = torch.empty((0, 3, 3), device=self.device)
                results.append({"instances": inst})
                continue

            scale_x = orig_w / float(curr_w)
            scale_y = orig_h / float(curr_h)

            boxes_xyxy = p2d[:, :4].clone()
            boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]]).clamp(0, curr_w) * scale_x
            boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]]).clamp(0, curr_h) * scale_y

            inst.pred_boxes = Boxes(boxes_xyxy)
            inst.scores = p2d[:, 4]
            inst.pred_classes = p2d[:, 5].long()

            box_classes = inst.pred_classes
            src_boxes = p2d[:, :4]
            K_orig = x["K"].to(self.device) if isinstance(x["K"], torch.Tensor) else torch.as_tensor(x["K"], dtype=torch.float32, device=self.device)
            scale_ratio = float(orig_h) / float(curr_h)
            K_scaled = K_orig / scale_ratio
            K_scaled[-1, -1] = 1.0
            Ks_per_box = K_scaled.unsqueeze(0).repeat(len(p2d), 1, 1)

            focal_lengths = K_orig[1, 1].unsqueeze(0).repeat(len(p2d))
            im_scales = torch.tensor(curr_h, device=self.device).repeat(len(p2d))
            im_scales_orig = torch.tensor(orig_h, device=self.device).repeat(len(p2d))

            per_img_cube_preds = {k: v[i][valid_mask] for k, v in cube_preds.items()}
            cube_decoded = head.decode_cube(
                cube_preds=per_img_cube_preds,
                box_classes=box_classes,
                src_boxes=src_boxes,
                Ks_scaled_per_box=Ks_per_box,
                focal_lengths=focal_lengths,
                im_scales_orig=im_scales_orig,
                im_scales=im_scales,
            )

            box3d_cam = torch.cat([cube_decoded["center_cam"], cube_decoded["dims"]], dim=-1)
            inst.pred_bbox3D = cubeutil.get_cuboid_verts_faces(box3d_cam, cube_decoded["pose"])[0]
            inst.pred_center_cam = cube_decoded["center_cam"]
            # 3D 中心點投影到「原始影像」上的 2D 座標，不是相機空間 xyz —— cubercnn 的
            # instances_to_coco_json 要這個欄位。decode_cube 算出來的 center_2d 是在網路輸入
            # (resize 後、還沒還原) 的尺度，要乘回 scale_ratio 才對得上 pred_boxes 的原圖尺度，
            # 跟 DenseCubeHead.inference() 的 `pred_center_2D = cube_xy * im_scales_ratio` 同一套。
            inst.pred_center_2D = cube_decoded["center_2d"] * scale_ratio
            inst.pred_dimensions = cube_decoded["dims"]
            inst.pred_pose = cube_decoded["pose"]

            results.append({"instances": inst})

        return results

    def visualize_training(self, batched_inputs, images, pad_hw):
        """Draw GT vs predicted 2D boxes and 3D cuboids to TensorBoard.

        Reuses `self.inference` (this wrapper's own eval-mode path) instead of a
        cubercnn-style `self.head.inference(...)` call — our head has no such method,
        and `self.inference` already returns Instances in original-image scale with
        exactly the fields (`pred_boxes`, `scores`, `pred_center_cam`, `pred_dimensions`,
        `pred_pose`) this function needs.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        if not hasattr(self, "thing_classes"):
            self.thing_classes = MetadataCatalog.get(self.metadata_name).thing_classes
            self.num_classes = len(self.thing_classes)

        # 模型此刻在 train() 模式，preds 是還沒後處理的訓練輸出，
        # 借用 self.inference 需要暫時切到 eval 模式才能拿到解碼後的偵測結果
        was_training = self.yolo_model.training
        self.yolo_model.eval()
        with torch.no_grad():
            raw_results = self.inference(images, batched_inputs, pad_hw)
        if was_training:
            self.yolo_model.train()
        results = [r["instances"] for r in raw_results]

        for input_info, instances_i in zip(batched_inputs, results):
            img = input_info["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]])
            img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]])

            # 1. 2D GT vs 2D 預測框
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input_info["instances"].gt_boxes)
            anno_img = v_gt.get_image()

            if len(instances_i) == 0:
                break

            keep = batched_nms(
                instances_i.pred_boxes.tensor,
                instances_i.scores,
                torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device),
                self.vis_nms_thresh,
            )[:max_vis_prop]

            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(boxes=instances_i.pred_boxes[keep].tensor.cpu().numpy())
            prop_img = v_pred.get_image()

            vis_img_2d = np.concatenate((anno_img, prop_img), axis=1)
            vis_img_2d = vis_img_2d.transpose(2, 0, 1)
            storage.put_image("Left: GT 2D boxes; Right: Predicted 2D boxes", vis_img_2d)

            # 2. 3D GT vs 3D 預測立方體
            K = torch.tensor(input_info["K"], device=self.device)
            scale = input_info["height"] / img.shape[0]
            K_scaled = torch.tensor(
                [[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1.0]],
                dtype=torch.float32, device=self.device,
            ) @ K

            gts_per_image = input_info["instances"]
            gt_classes = gts_per_image.gt_classes
            fg = (gt_classes != -1) & (gt_classes < self.num_classes)
            gt_classes = gt_classes[fg]
            gt_class_names = [self.thing_classes[c] for c in gt_classes]
            gt_poses = gts_per_image.gt_poses[fg]
            gt_boxes3D = gts_per_image.gt_boxes3D[fg]

            fx, sx = (v.item() / scale for v in K[0, [0, 2]])
            fy, sy = (v.item() / scale for v in K[1, [1, 2]])
            gt_z = gt_boxes3D[:, 2]
            gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx) / fx
            gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy) / fy
            gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
            gt_boxes3D_XYZ_WHL = torch.cat((gt_center_3D, gt_boxes3D[:, 3:6]), dim=1)
            gt_colors = torch.tensor(
                [cubeutil.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))], device=self.device
            ) / 255.0
            gt_meshes = cubeutil.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)

            pred_xyzwhl = torch.cat(
                (instances_i.pred_center_cam[keep], instances_i.pred_dimensions[keep]), dim=1
            )
            pred_pose = instances_i.pred_pose[keep]
            pred_colors = torch.tensor(
                [cubeutil.get_color(i) for i in range(len(keep))], device=self.device
            ) / 255.0
            pred_classes = instances_i.pred_classes[keep]
            pred_scores = instances_i.scores[keep]
            pred_class_names = [
                "{} {:.2f}".format(self.thing_classes[c], s) for c, s in zip(pred_classes, pred_scores)
            ]
            pred_meshes = cubeutil.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

            pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]
            gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]

            img_3DPR = vis.draw_scene_view(
                img_3DPR, K_scaled.cpu().numpy(), pred_meshes, text=pred_class_names,
                mode="front", blend_weight=0.0, blend_weight_overlay=0.85,
            )
            img_3DGT = vis.draw_scene_view(
                img_3DGT, K_scaled.cpu().numpy(), gt_meshes, text=gt_class_names,
                mode="front", blend_weight=0.0, blend_weight_overlay=0.85,
            )

            vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)[:, :, [2, 1, 0]]
            vis_img_3d = vis_img_3d.astype(np.uint8).transpose(2, 0, 1)
            storage.put_image("Left: GT 3D cuboids; Right: Predicted 3D cuboids", vis_img_3d)

            break

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.criterion.to(self.pixel_mean.device)
        return ret

def build_yolo_wrapper(cfg, priors=None):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, priors=priors)
    if priors is not None:
        model.set_priors(priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model