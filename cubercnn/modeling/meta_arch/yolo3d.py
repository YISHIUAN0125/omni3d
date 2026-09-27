from typing import Dict, List, Tuple, Optional, Any
import logging
import os
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
    """Final debug wrapper aligned with Cube R-CNN/Omni3D geometry."""

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
        pretrained_weights: Optional[str],
        vis_nms_thresh: float = 0.5,
        metadata_name: str = "omni3d_model",
        debug_mode: bool = False,
        debug_interval: int = 50,
        debug_inference_images: int = 3,
        debug_save_dir: str = "./output/yolo3d_debug",
        inference_conf_threshold: float = 0.01,
    ):
        super().__init__()
        self.yolo_yaml = yolo_yaml
        self.is_e2e = bool(is_e2e)
        self.input_format = input_format
        self.vis_period = int(vis_period)
        self.vis_nms_thresh = float(vis_nms_thresh)
        self.metadata_name = metadata_name
        self.debug_mode = bool(debug_mode)
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_inference_images = max(int(debug_inference_images), 0)
        self.debug_save_dir = debug_save_dir
        self.inference_conf_threshold = float(inference_conf_threshold)
        self._debug_inference_counter = 0
        self._debug_saved_prediction = False

        if self.debug_mode and comm.is_main_process():
            os.makedirs(self.debug_save_dir, exist_ok=True)

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1), False)
        self.yolo_model = YOLO(yolo_yaml).model

        if pretrained_weights:
            LOGGER.info("Load yolo pertrain...")
            self.pretrained_load_info = self.load_yolo_pretrained(self.yolo_model, pretrained_weights)
        else:
            self.pretrained_load_info = None
            LOGGER.info("No pretrained YOLO weights supplied; training from random initialization.")

        head = self.yolo_model.model[-1]
        head_is_e2e = bool(getattr(head, "end2end", True))
        if self.is_e2e != head_is_e2e:
            raise ValueError(f"E2E mismatch: cfg={self.is_e2e}, head.end2end={head_is_e2e}")
        self.is_e2e = head_is_e2e
        self._build_criterion()

    @classmethod
    def from_config(cls, cfg, priors=None):
        return {
            "yolo_yaml": cfg.MODEL.YOLO3D.ARCH,
            "is_e2e": cfg.MODEL.YOLO3D.IS_E2E,
            "input_format": cfg.MODEL.INPUT_FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "pretrained_weights": cfg.MODEL.PRETRAINED_WEIGHTS,
            "metadata_name": cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "omni3d_model",
            "debug_mode": True,
            "debug_interval": 50,
            "debug_inference_images": 3,
            "debug_save_dir": os.path.join(getattr(cfg, "OUTPUT_DIR", "./output"), "yolo3d_debug"),
            "inference_conf_threshold": 0.01,
        }

    @staticmethod
    def load_yolo_pretrained(yolo_model: nn.Module, pt_path: str, min_coverage: float = 0.80) -> Dict[str, Any]:
        try:
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            candidate = next((ckpt[k] for k in ("ema", "model", "state_dict") if ckpt.get(k) is not None), ckpt)
        else:
            candidate = ckpt
        if isinstance(candidate, nn.Module):
            src = candidate.float().state_dict()
        elif isinstance(candidate, dict):
            src = candidate
        else:
            raise TypeError(f"Unsupported checkpoint: {type(candidate).__name__}")

        normalized = {}
        for key, value in src.items():
            while key.startswith("module."):
                key = key[7:]
            if isinstance(value, nn.Parameter):
                value = value.detach()
            if torch.is_tensor(value):
                normalized[key] = value

        target = yolo_model.state_dict()
        head_prefix = f"model.{len(yolo_model.model) - 1}."
        eligible = [k for k in target if not k.startswith(head_prefix)]
        filtered, missing, mismatch = {}, [], []
        for key in eligible:
            value = normalized.get(key)
            if value is None:
                missing.append(key)
            elif value.shape != target[key].shape:
                mismatch.append(key)
            else:
                filtered[key] = value
        yolo_model.load_state_dict(filtered, strict=False)
        coverage = len(filtered) / max(len(eligible), 1)
        LOGGER.info(f"Pretrained coverage={coverage:.2%}, missing={len(missing)}, mismatch={len(mismatch)}")
        if coverage < min_coverage:
            raise RuntimeError(f"Pretrained coverage too low: {coverage:.2%}")
        return {"loaded": len(filtered), "eligible": len(eligible), "coverage": coverage}

    @property
    def device(self):
        return self.pixel_mean.device

    def _build_criterion(self):
        self.criterion = E2EDetect3DLoss(self.yolo_model) if self.is_e2e else Detect3DLoss(self.yolo_model)
        self.criterion.to(self.device)

    def set_priors(self, priors: dict):
        head = self.yolo_model.model[-1]
        if not hasattr(head, "set_priors"):
            LOGGER.warning("Head has no set_priors method.")
            return
        head.set_priors(priors)
        self._build_criterion()

    @staticmethod
    def _summary(name: str, value: Any, n: int = 8) -> str:
        if not torch.is_tensor(value):
            return f"{name}: type={type(value).__name__}, value={value}"
        x = value.detach()
        s = f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
        if x.numel() == 0:
            return s + ", empty=True"
        if torch.is_floating_point(x):
            finite = torch.isfinite(x)
            s += f", finite={int(finite.sum())}/{x.numel()}, nan={int(torch.isnan(x).sum())}, inf={int(torch.isinf(x).sum())}"
            if finite.any():
                y = x[finite].float()
                s += f", min={y.min().item():.6g}, mean={y.mean().item():.6g}, max={y.max().item():.6g}"
        return s + f", first={x.reshape(-1)[:n].cpu().tolist()}"

    @staticmethod
    def _safe_scalar(value: Any) -> Optional[float]:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            value = value.detach().float().mean().cpu().item()
        elif isinstance(value, (float, int, np.number)):
            value = float(value)
        else:
            return None
        return float(value) if np.isfinite(value) else None

    def _iteration(self):
        try:
            return int(get_event_storage().iter)
        except Exception:
            return -1

    def _debug_train_now(self):
        it = self._iteration()
        return self.debug_mode and comm.is_main_process() and (it < 0 or it % self.debug_interval == 0)

    def _log_losses(self, total_loss, loss_items):
        storage = get_event_storage()
        scalar = self._safe_scalar(total_loss)
        if scalar is not None:
            storage.put_scalar("yolo_metrics/total_loss", scalar)
        if isinstance(loss_items, dict):
            iterable = loss_items.items()
        elif torch.is_tensor(loss_items):
            names = ["box", "cls", "dfl", "3d"]
            iterable = [(names[i] if i < len(names) else f"component_{i}", v) for i, v in enumerate(loss_items.reshape(-1))]
        else:
            iterable = []
        for key, value in iterable:
            scalar = self._safe_scalar(value)
            if scalar is not None:
                storage.put_scalar(f"yolo_metrics/{key}", scalar)

    def _loss_dict(self, total_loss):
        if not torch.is_tensor(total_loss):
            total_loss = torch.as_tensor(total_loss, dtype=torch.float32, device=self.device)
        flat = total_loss.reshape(-1)
        if flat.numel() == 1:
            return {"loss_yolo3d": flat[0]}
        if flat.numel() == 4:
            return {"loss_box": flat[0], "loss_cls": flat[1], "loss_dfl": flat[2], "loss_3d": flat[3]}
        return {"loss_yolo3d": flat.sum()}

    def _get_thing_classes(self) -> List[str]:
        """Return contiguous class names used by pred_classes/gt_classes."""
        if hasattr(self, "thing_classes"):
            return self.thing_classes

        metadata = MetadataCatalog.get(self.metadata_name)
        classes = list(getattr(metadata, "thing_classes", []) or [])

        # Some Omni3D training datasets keep classes on the shared metadata entry.
        if not classes and self.metadata_name != "omni3d_model":
            shared_metadata = MetadataCatalog.get("omni3d_model")
            classes = list(getattr(shared_metadata, "thing_classes", []) or [])

        head_nc = int(getattr(self.yolo_model.model[-1], "nc", len(classes)))
        if len(classes) < head_nc:
            LOGGER.warning(
                f"Metadata '{self.metadata_name}' provides {len(classes)} class names, "
                f"but the YOLO head has nc={head_nc}. Missing names use class_<id>."
            )
            classes.extend([f"class_{i}" for i in range(len(classes), head_nc)])

        self.thing_classes = classes
        return self.thing_classes

    def _class_name(self, class_id: Any) -> str:
        """Convert a contiguous class ID to a safe human-readable class name."""
        class_id = int(class_id.item()) if torch.is_tensor(class_id) else int(class_id)
        classes = self._get_thing_classes()
        if 0 <= class_id < len(classes):
            return str(classes[class_id])
        return f"class_{class_id}"

    def forward(self, batched_inputs: List[Dict[str, Any]]):
        images, batch, pad_hw = self.preprocess_inputs(batched_inputs)
        if not self.training:
            return self.inference(images, batched_inputs, pad_hw)

        preds = self.yolo_model(images)
        total_loss, loss_items = self.criterion(preds, batch)
        losses = self._loss_dict(total_loss)
        for key, value in losses.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {key}: {self._summary(key, value)}")
        self._log_losses(total_loss, loss_items)

        if self._debug_train_now():
            print("\n" + "=" * 100)
            print(f"YOLO3D TRAIN DEBUG iter={self._iteration()}")
            print(self._summary("images", images))
            for key in ("batch_idx", "cls", "bboxes", "gt_boxes3d", "gt_poses", "gt_2d", "Ks"):
                print(self._summary(key, batch[key]))
            print(self._summary("total_loss", total_loss))
            print(self._summary("loss_items", loss_items))
            for key, value in losses.items():
                print(self._summary(key, value))
            print("=" * 100 + "\n")

        storage = get_event_storage()
        if self.vis_period > 0 and comm.is_main_process() and storage.iter > 0 and storage.iter % self.vis_period == 0:
            self.visualize_training(batched_inputs, images, pad_hw)
        return losses

    def preprocess_inputs(self, batched_inputs: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, Any], Tuple[int, int]]:
        image_list, max_h, max_w = [], 0, 0
        for x in batched_inputs:
            image = x["image"].to(self.device).float()
            if self.input_format == "BGR":
                image = image[[2, 1, 0]]
            image = (image - self.pixel_mean) / self.pixel_std
            image_list.append(image)
            max_h, max_w = max(max_h, image.shape[1]), max(max_w, image.shape[2])

        pad_h, pad_w = int(np.ceil(max_h / 32) * 32), int(np.ceil(max_w / 32) * 32)
        images = torch.stack([F.pad(im, (0, pad_w-im.shape[2], 0, pad_h-im.shape[1]), value=0.0) for im in image_list])
        batch_idx, classes, bboxes, boxes3d, poses, centers2d = [], [], [], [], [], []
        Ks, scales, scales_orig, ratios = [], [], [], []

        for i, x in enumerate(batched_inputs):
            orig_h, orig_w = int(x.get("height", x["image"].shape[1])), int(x.get("width", x["image"].shape[2]))
            curr_h, curr_w = int(x["image"].shape[1]), int(x["image"].shape[2])
            K = torch.as_tensor(x["K"], dtype=torch.float32, device=self.device).clone()
            ratio_h = float(orig_h) / float(curr_h)
            Ks.append(K)
            scales.append(torch.tensor(curr_h, dtype=torch.float32, device=self.device))
            scales_orig.append(torch.tensor(orig_h, dtype=torch.float32, device=self.device))
            ratios.append(torch.tensor(ratio_h, dtype=torch.float32, device=self.device))

            if "instances" not in x:
                continue
            inst = x["instances"].to(self.device)
            if len(inst) == 0:
                continue
            for field in ("gt_boxes", "gt_classes", "gt_boxes3D", "gt_poses"):
                if not inst.has(field):
                    raise ValueError(f"Input {i} missing {field}")

            nc = getattr(self.yolo_model.model[-1], "nc", None)
            valid_gt = inst.gt_classes >= 0
            if nc is not None:
                valid_gt &= inst.gt_classes < int(nc)
            if self._debug_train_now() and (~valid_gt).any():
                print(f"PREPROCESS DEBUG image={i}: filtered {int((~valid_gt).sum())}/{len(inst)} invalid classes")
            inst = inst[valid_gt]
            if len(inst) == 0:
                continue

            n = len(inst)
            batch_idx.append(torch.full((n,), i, dtype=torch.long, device=self.device))
            classes.append(inst.gt_classes.long())
            box = inst.gt_boxes.tensor.float()
            x1, y1, x2, y2 = box.unbind(-1)
            bboxes.append(torch.stack(((x1+x2)*0.5/pad_w, (y1+y2)*0.5/pad_h, (x2-x1)/pad_w, (y2-y1)/pad_h), -1))

            b3d = inst.gt_boxes3D.float()
            pose = inst.gt_poses.float()
            if b3d.ndim != 2 or b3d.shape[1] < 9:
                raise ValueError(f"Expected gt_boxes3D [N,>=9], got {tuple(b3d.shape)}")
            if not torch.isfinite(b3d).all() or not torch.isfinite(pose).all():
                raise FloatingPointError("Non-finite 3D annotation")
            boxes3d.append(b3d)
            poses.append(pose)

            # Cube R-CNN contract: columns 0:2 are projected center in mapper-resized pixels.
            center = b3d[:, 0:2].clone()
            if inst.has("gt_center_2D"):
                explicit = inst.gt_center_2D.float()
                if torch.isfinite(explicit).all() and explicit.shape == center.shape:
                    center = explicit
            if (center.abs() > torch.tensor([pad_w*10.0, pad_h*10.0], device=self.device)).any():
                raise ValueError(f"Projected centers far outside image: {center[:10].cpu().tolist()}")
            centers2d.append(center)

        cat = lambda xs, shape, dtype: torch.cat(xs, 0) if xs else torch.empty(shape, dtype=dtype, device=self.device)
        dim3d = int(boxes3d[0].shape[1]) if boxes3d else 9
        batch = {
            "batch_idx": cat(batch_idx, (0,), torch.long),
            "cls": cat(classes, (0,), torch.long),
            "bboxes": cat(bboxes, (0, 4), torch.float32),
            "gt_boxes3d": cat(boxes3d, (0, dim3d), torch.float32),
            "gt_poses": cat(poses, (0, 3, 3), torch.float32),
            "gt_2d": cat(centers2d, (0, 2), torch.float32),
            "Ks": torch.stack(Ks),
            "im_scales": torch.stack(scales),
            "im_scales_orig": torch.stack(scales_orig),
            "im_scales_ratio": torch.stack(ratios),
        }
        return images, batch, (pad_h, pad_w)

    def _empty_instances(self, h, w):
        inst = Instances((h, w))
        inst.pred_boxes = Boxes(torch.empty((0, 4), device=self.device))
        inst.scores = torch.empty(0, device=self.device)
        inst.pred_classes = torch.empty(0, dtype=torch.long, device=self.device)
        inst.pred_bbox3D = torch.empty((0, 8, 3), device=self.device)
        inst.pred_center_cam = torch.empty((0, 3), device=self.device)
        inst.pred_center_2D = torch.empty((0, 2), device=self.device)
        inst.pred_dimensions = torch.empty((0, 3), device=self.device)
        inst.pred_pose = torch.empty((0, 3, 3), device=self.device)
        return inst

    @torch.no_grad()
    def inference(self, images, batched_inputs, pad_hw):
        head = self.yolo_model.model[-1]
        out = self.yolo_model(images)
        if head.end2end:
            pred_2d, cube_preds, *_ = out
        else:
            y, dense_cube_preds, *_ = out
            pred_2d, cube_preds = head.postprocess(y.permute(0, 2, 1), dense_cube_preds)

        debug = self.debug_mode and comm.is_main_process() and self._debug_inference_counter < self.debug_inference_images
        results = []
        for i, x in enumerate(batched_inputs):
            orig_h, orig_w = int(x.get("height", x["image"].shape[1])), int(x.get("width", x["image"].shape[2]))
            curr_h, curr_w = int(x["image"].shape[1]), int(x["image"].shape[2])
            raw = pred_2d[i]
            valid_conf = torch.isfinite(raw[:, :6]).all(1) & (raw[:, 4] > self.inference_conf_threshold)
            p2d = raw[valid_conf]
            if debug:
                print(f"INFERENCE image={i}: raw={len(raw)}, confidence_valid={int(valid_conf.sum())}")
                print(self._summary("raw_scores", raw[:, 4]))
            if len(p2d) == 0:
                results.append({"instances": self._empty_instances(orig_h, orig_w)})
                continue

            to_orig_x, to_orig_y = orig_w/float(curr_w), orig_h/float(curr_h)
            to_resize_x, to_resize_y = curr_w/max(float(orig_w), 1.0), curr_h/max(float(orig_h), 1.0)
            if abs(to_resize_x - to_resize_y) > 1e-3:
                LOGGER.warning(f"Non-isotropic resize: x={to_resize_x:.6f}, y={to_resize_y:.6f}")

            boxes_resized = p2d[:, :4].clone()
            boxes_original = boxes_resized.clone()
            boxes_original[:, [0, 2]] = boxes_original[:, [0, 2]].clamp(0, curr_w) * to_orig_x
            boxes_original[:, [1, 3]] = boxes_original[:, [1, 3]].clamp(0, curr_h) * to_orig_y
            pred_classes = p2d[:, 5].long()

            K_orig = torch.as_tensor(x["K"], dtype=torch.float32, device=self.device).clone()
            K_scaled = K_orig.clone()
            K_scaled[0, 0] *= to_resize_x
            K_scaled[0, 2] *= to_resize_x
            K_scaled[1, 1] *= to_resize_y
            K_scaled[1, 2] *= to_resize_y
            K_scaled[2] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            Ks_box = K_scaled.unsqueeze(0).expand(len(p2d), -1, -1).contiguous()
            focal = K_orig[1, 1].reshape(1).expand(len(p2d)).contiguous()
            im_scale = torch.full((len(p2d),), float(curr_h), device=self.device)
            im_orig = torch.full((len(p2d),), float(orig_h), device=self.device)

            per_cube = {}
            for key, value in cube_preds.items():
                candidate = value[i]
                if candidate.shape[0] != valid_conf.shape[0]:
                    raise ValueError(f"2D/cube misalignment for {key}: {candidate.shape[0]} vs {valid_conf.shape[0]}")
                per_cube[key] = candidate[valid_conf]

            decoded = head.decode_cube(
                cube_preds=per_cube, box_classes=pred_classes, src_boxes=boxes_resized,
                Ks_scaled_per_box=Ks_box, focal_lengths=focal,
                im_scales_orig=im_orig, im_scales=im_scale,
            )
            center, dims_raw, pose, center2d = decoded["center_cam"], decoded["dims"], decoded["pose"], decoded["center_2d"]
            geometry_valid = (
                torch.isfinite(center).all(1) & torch.isfinite(dims_raw).all(1)
                & torch.isfinite(pose).all((1, 2)) & torch.isfinite(center2d).all(1)
                & (center[:, 2] > 1e-4) & (dims_raw > 1e-4).all(1)
            )
            eye = torch.eye(3, dtype=pose.dtype, device=pose.device).unsqueeze(0)
            ortho_error = torch.linalg.norm(pose.transpose(1, 2) @ pose - eye, dim=(1, 2))
            determinant = torch.det(pose)

            if debug:
                print("K original:\n", K_orig.cpu())
                print("K scaled:\n", K_scaled.cpu())
                print(self._summary("center_2d_resized", center2d))
                print(self._summary("center_cam", center))
                print(self._summary("dims", dims_raw))
                print(self._summary("pose_orthogonal_error", ortho_error))
                print(self._summary("pose_determinant", determinant))
                print(f"geometry_valid={int(geometry_valid.sum())}/{len(geometry_valid)}")

            if self.debug_mode and comm.is_main_process() and not self._debug_saved_prediction:
                torch.save({
                    "image_id": x.get("image_id"), "file_name": x.get("file_name"),
                    "orig_hw": (orig_h, orig_w), "curr_hw": (curr_h, curr_w), "pad_hw": pad_hw,
                    "K_original": K_orig.cpu(), "K_scaled": K_scaled.cpu(),
                    "pred_2d_raw": raw.cpu(), "confidence_valid_mask": valid_conf.cpu(),
                    "geometry_valid_mask": geometry_valid.cpu(),
                    "cube_preds": {k: v.cpu() for k, v in per_cube.items()},
                    "cube_decoded": {k: v.cpu() for k, v in decoded.items() if torch.is_tensor(v)},
                    "pose_orthogonal_error": ortho_error.cpu(), "pose_determinant": determinant.cpu(),
                }, os.path.join(self.debug_save_dir, "debug_prediction.pt"))
                self._debug_saved_prediction = True

            if not geometry_valid.any():
                results.append({"instances": self._empty_instances(orig_h, orig_w)})
                continue

            p2d, boxes_original, pred_classes = p2d[geometry_valid], boxes_original[geometry_valid], pred_classes[geometry_valid]
            center, dims, pose, center2d = center[geometry_valid], dims_raw[geometry_valid].clamp_min(0.01), pose[geometry_valid], center2d[geometry_valid]
            inst = Instances((orig_h, orig_w))
            inst.pred_boxes = Boxes(boxes_original.detach())
            inst.scores = p2d[:, 4].detach()
            inst.pred_classes = pred_classes.detach()
            inst.pred_center_cam = center.detach()
            inst.pred_dimensions = dims.detach()
            inst.pred_pose = pose.detach()
            inst.pred_center_2D = (center2d * torch.tensor([to_orig_x, to_orig_y], device=self.device)).detach()
            inst.pred_bbox3D = cubeutil.get_cuboid_verts_faces(torch.cat((center, dims), -1), pose)[0].detach()
            if debug:
                print(f"evaluator_fields={list(inst.get_fields().keys())}")
            results.append({"instances": inst})

        if debug:
            self._debug_inference_counter += 1
        return results

    @torch.no_grad()
    def visualize_training(self, batched_inputs, images, pad_hw):
        storage = get_event_storage()
        was_training = self.yolo_model.training
        try:
            self.yolo_model.eval()
            result = self.inference(images, batched_inputs, pad_hw)[0]["instances"]
            info = batched_inputs[0]
            image = info["image"].detach().cpu().permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = convert_image_to_rgb(image, self.input_format)
            curr_h, curr_w = image.shape[:2]
            orig_h, orig_w = int(info.get("height", curr_h)), int(info.get("width", curr_w))

            gt_v = Visualizer(image, None)
            gt_img = gt_v.overlay_instances(boxes=info["instances"].gt_boxes).get_image()
            keep = torch.empty(0, dtype=torch.long, device=self.device)
            pred_img = image.copy()
            if len(result):
                keep = batched_nms(result.pred_boxes.tensor, result.scores, torch.zeros_like(result.pred_classes), self.vis_nms_thresh)[:20]
                boxes = result.pred_boxes.tensor[keep].clone()
                boxes[:, [0, 2]] *= curr_w/max(float(orig_w), 1.0)
                boxes[:, [1, 3]] *= curr_h/max(float(orig_h), 1.0)
                labels = [f"{self._class_name(c)} {float(s):.2f}" for c, s in zip(result.pred_classes[keep], result.scores[keep])]
                pred_img = Visualizer(image, None).overlay_instances(boxes=boxes.cpu().numpy(), labels=labels).get_image()
            storage.put_image("YOLO3D/2D_Left_GT_Right_Pred", np.concatenate((gt_img, pred_img), 1).astype(np.uint8).transpose(2, 0, 1))

            K_orig = torch.as_tensor(info["K"], dtype=torch.float32, device=self.device).clone()
            sx, sy = curr_w/max(float(orig_w), 1.0), curr_h/max(float(orig_h), 1.0)
            K_scaled = K_orig.clone()
            K_scaled[0, 0] *= sx; K_scaled[0, 2] *= sx
            K_scaled[1, 1] *= sy; K_scaled[1, 2] *= sy
            K_scaled[2] = torch.tensor([0.0, 0.0, 1.0], device=self.device)

            gt = info["instances"].to(self.device)
            nc = getattr(self.yolo_model.model[-1], "nc", 10**9)
            valid_gt = (gt.gt_classes >= 0) & (gt.gt_classes < int(nc))
            b3d, gt_pose, gt_cls = gt.gt_boxes3D[valid_gt], gt.gt_poses[valid_gt], gt.gt_classes[valid_gt]
            gt_mesh_list, gt_names = [], []
            if len(b3d):
                u, v, z = b3d[:, 0], b3d[:, 1], b3d[:, 2]
                x3d = z * (u-K_scaled[0, 2]) / K_scaled[0, 0].clamp_min(1e-6)
                y3d = z * (v-K_scaled[1, 2]) / K_scaled[1, 1].clamp_min(1e-6)
                xyzwhl = torch.cat((torch.stack((x3d, y3d, z), -1), b3d[:, 3:6].clamp_min(0.01)), -1)
                colors = torch.tensor([cubeutil.get_color(j) for j in range(len(xyzwhl))], device=self.device) / 255.0
                meshes = cubeutil.mesh_cuboid(xyzwhl, gt_pose, colors)
                gt_mesh_list = [meshes.__getitem__(j).detach() for j in range(len(meshes))]
                gt_names = [self._class_name(c) for c in gt_cls]

            pred_mesh_list, pred_names = [], []
            if len(keep):
                xyzwhl = torch.cat((result.pred_center_cam[keep], result.pred_dimensions[keep]), -1)
                colors = torch.tensor([cubeutil.get_color(j) for j in range(len(keep))], device=self.device) / 255.0
                meshes = cubeutil.mesh_cuboid(xyzwhl, result.pred_pose[keep], colors)
                pred_mesh_list = [meshes.__getitem__(j).detach() for j in range(len(meshes))]
                pred_names = [f"{self._class_name(c)} {float(s):.2f}" for c, s in zip(result.pred_classes[keep], result.scores[keep])]

            gt_3d, pred_3d = np.ascontiguousarray(image.copy()), np.ascontiguousarray(image.copy())
            if gt_mesh_list:
                gt_3d = vis.draw_scene_view(gt_3d, K_scaled.cpu().numpy(), gt_mesh_list, text=gt_names, mode="front", blend_weight=0.0, blend_weight_overlay=0.85)
            if pred_mesh_list:
                pred_3d = vis.draw_scene_view(pred_3d, K_scaled.cpu().numpy(), pred_mesh_list, text=pred_names, mode="front", blend_weight=0.0, blend_weight_overlay=0.85)
            storage.put_image("YOLO3D/3D_Left_GT_Right_Pred", np.concatenate((gt_3d, pred_3d), 1).astype(np.uint8).transpose(2, 0, 1))
            LOGGER.info(f"TensorBoard YOLO3D images written at iter={storage.iter}")
        except Exception as exc:
            LOGGER.exception(f"YOLO3D visualization failed at iter={storage.iter}: {exc}")
        finally:
            if was_training:
                self.yolo_model.train()

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        if hasattr(self, "criterion"):
            self.criterion.to(self.pixel_mean.device)
        return ret


def build_yolo_wrapper(cfg, priors=None):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model_cls = META_ARCH_REGISTRY.get(meta_arch)

    kwargs = model_cls.from_config(cfg, priors=priors)
    model = model_cls(**kwargs)
    
    if priors is not None:
        model.set_priors(priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model