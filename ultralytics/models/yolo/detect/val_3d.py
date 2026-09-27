"""Ultralytics validator adapted to the official Cube R-CNN Omni3D evaluator."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, ops
import cubercnn.vis.logperf as utils_logperf
from collections import OrderedDict, abc
import logging, sys
from ultralytics.utils.cube_utils import get_cuboid_verts_faces

# 關鍵：配置 cubercnn 日誌流，確保 logperf 產出的四張大表直接輸出至終端控制台
_c_logger = logging.getLogger("cubercnn")
_c_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in _c_logger.handlers):
    _h = logging.StreamHandler(sys.stdout)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _c_logger.addHandler(_h)

_lp_logger = logging.getLogger("cubercnn.vis.logperf")
_lp_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in _lp_logger.handlers):
    _h2 = logging.StreamHandler(sys.stdout)
    _h2.setLevel(logging.INFO)
    _h2.setFormatter(logging.Formatter("%(message)s"))
    _lp_logger.addHandler(_h2)
    _lp_logger.propagate = False


STAT_NAMES_2D = (
    "AP", "AP50", "AP75", "AP95", "APs", "APm", "APl",
    "AR-1", "AR-10", "AR-100", "ARs", "ARm", "ARl",
)

STAT_NAMES_3D = (
    "AP", "AP15", "AP25", "AP50", "AP-near", "AP-medium", "AP-far",
    "AR-1", "AR-10", "AR-100", "AR-near", "AR-medium", "AR-far",
)


def is_ignore(anno: dict, filter_settings: dict, image_height: int) -> bool:
    """判斷標註是否應被忽略，與官方 Cube R-CNN 規則完全一致。"""
    if anno.get("behind_camera", False) or not bool(anno.get("valid3D", True)):
        return True

    dims = anno.get("dimensions", [0, 0, 0])
    if dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0:
        return True
    center = anno.get("center_cam", [0, 0, 0])
    if center[2] > filter_settings.get("max_depth", 512.0):
        return True
    if anno.get("lidar_pts", 1) == 0 or anno.get("segmentation_pts", 1) == 0:
        return True
    if anno.get("depth_error", 0) > 0.5:
        return True

    tight = anno.get("bbox2D_tight")
    trunc = anno.get("bbox2D_trunc")
    proj = anno.get("bbox2D_proj")

    if filter_settings.get("modal_2D_boxes") and tight and tight[0] != -1:
        bbox2D = [tight[0], tight[1], tight[2] - tight[0], tight[3] - tight[1]]
    elif filter_settings.get("trunc_2D_boxes") and trunc and not all(v == -1 for v in trunc):
        bbox2D = [trunc[0], trunc[1], trunc[2] - trunc[0], trunc[3] - trunc[1]]
    elif proj and proj[0] != -1:
        bbox2D = [proj[0], proj[1], proj[2] - proj[0], proj[3] - proj[1]]
    else:
        bbox2D = anno.get("bbox", [0, 0, 0, 0])

    if bbox2D[3] <= filter_settings.get("min_height_thres", 0.0) * image_height:
        return True
    if bbox2D[3] >= filter_settings.get("max_height_thres", 1.5) * image_height:
        return True

    trunc_val = anno.get("truncation", -1)
    if trunc_val >= 0 and trunc_val >= filter_settings.get("truncation_thres", 0.99):
        return True
    vis_val = anno.get("visibility", -1)
    if vis_val >= 0 and vis_val <= filter_settings.get("visibility_thres", 0.01):
        return True

    if "ignore_names" in filter_settings and anno.get("category_name") in filter_settings["ignore_names"]:
        return True

    return False


class Omni3DValidationGT(COCO):
    """專為驗證設計的等價 GT 載入器，主動衍生 bbox, bbox3D, depth, area 欄位並消除 ID 碰撞。"""

    def __init__(self, annotation_files: list[str], filter_settings: dict, dataset_idx_offset: int = 0):
        super().__init__()
        self.dataset = {"images": [], "annotations": [], "categories": []}
        self.anns = {}
        self.imgs = {}
        self.cats = {}
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)

        if isinstance(annotation_files, (str, Path)):
            annotation_files = [annotation_files]

        cats_master = {}
        valid_anns = []
        im_height_map = {}

        for dataset_idx, jf in enumerate(annotation_files):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            id_offset = (dataset_idx + dataset_idx_offset) * 1_000_000

            for cat in data.get("categories", []):
                cats_master[cat["id"]] = cat

            for im in data.get("images", []):
                im_copy = dict(im)
                im_copy["id"] = int(im["id"]) + id_offset
                self.dataset["images"].append(im_copy)
                im_height_map[im_copy["id"]] = im_copy["height"]

            for anno in data.get("annotations", []):
                ann = dict(anno)
                ann["id"] = int(ann["id"]) + id_offset
                ann["image_id"] = int(ann["image_id"]) + id_offset

                im_h = im_height_map[ann["image_id"]]
                ignore = is_ignore(ann, filter_settings, im_h)

                tight = anno.get("bbox2D_tight")
                trunc = anno.get("bbox2D_trunc")
                proj = anno.get("bbox2D_proj")

                if filter_settings.get("modal_2D_boxes") and tight and tight[0] != -1:
                    bbox2D = [tight[0], tight[1], tight[2] - tight[0], tight[3] - tight[1]]
                elif filter_settings.get("trunc_2D_boxes") and trunc and not all(v == -1 for v in trunc):
                    bbox2D = [trunc[0], trunc[1], trunc[2] - trunc[0], trunc[3] - trunc[1]]
                elif proj and proj[0] != -1:
                    bbox2D = [proj[0], proj[1], proj[2] - proj[0], proj[3] - proj[1]]
                else:
                    bbox2D = ann.get("bbox", [0, 0, 0, 0])

                w, h = float(bbox2D[2]), float(bbox2D[3])
                ann["area"] = float(w * h)
                ann["iscrowd"] = 0
                ann["ignore"] = int(ignore)
                ann["ignore2D"] = int(ignore)
                ann["ignore3D"] = int(ignore)
                ann["bbox"] = bbox2D
                ann["bbox3D"] = ann.get("bbox3D_cam")
                center_cam = ann.get("center_cam", [0, 0, 0])
                ann["depth"] = float(center_cam[2])

                category_name = ann.get("category_name")
                if not filter_settings.get("category_names") or category_name in filter_settings["category_names"]:
                    valid_anns.append(ann)

        self.dataset["categories"] = sorted(cats_master.values(), key=lambda c: c["id"])
        self.dataset["annotations"] = valid_anns
        self.createIndex()


class _MetricsKeysShim:
    """Minimal stand-in for stock DetMetrics so `self.metrics.keys` works in BaseTrainer."""

    def __init__(self, validator: "Detection3DValidator"):
        self._validator = validator

    @property
    def keys(self) -> list[str]:
        return [key for key in self._validator.metric_keys if key != "fitness"]


class Detection3DValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = getattr(self.args, "task", "detect3d") or "detect3d"
        self.metrics = _MetricsKeysShim(self)
        self.predictions: list[dict[str, Any]] = []
        self.metric_results: dict[str, float] = {}
        self.head = None
        self.loss = defaultdict(float)
        self._id_map: dict[int, int] = {}
        self._contiguous_to_raw: dict[int, int] = {}
        self._json_files: list[str] = []
        self._gt: Omni3DValidationGT | None = None

    @property
    def metric_keys(self) -> list[str]:
        return [
            "metrics/AP2D", "metrics/AP2D50", "metrics/AP3D", "metrics/AP3D15",
            "metrics/AP3D25", "metrics/AP3D50", "metrics/AP3D-N",
            "metrics/AP3D-M", "metrics/AP3D-F", "fitness",
        ]

    def init_metrics(self, model):
        native = model.model if getattr(model, "format", None) == "pt" else model
        candidates = [
            m for m in native.modules()
            if hasattr(m, "decode_cube") and hasattr(m, "cube_branch")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one Detect3D head, found {len(candidates)}")
        self.head = candidates[0]
        self.names = model.names
        self.nc = len(self.names)
        self.seen = 0
        self.predictions = []
        self.metric_results = {}
        self.loss = defaultdict(float)
        self._gt = None

        dataset = self.dataloader.dataset
        self._json_files = list(dataset.json_files)
        self._id_map = dict(dataset.id_map)
        self._contiguous_to_raw = {v: k for k, v in self._id_map.items()}

    def get_desc(self):
        return ("%22s" + "%11s" * 5) % (
            "Class", "Images", "Preds", "AP2D", "AP3D", "AP3D50"
        )

    def preprocess(self, batch):
        non_blocking = self.device.type not in {"cpu", "mps"}
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device, non_blocking=non_blocking)
        batch["img"] = batch["img"].float() / 255.0
        return batch

    def postprocess(self, preds):
        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
            return preds

        detections, cube_preds = preds[0], preds[1]
        if detections.ndim == 3 and detections.shape[1] < detections.shape[2]:
            detections, cube_preds = self.head.postprocess(detections.permute(0, 2, 1), cube_preds)

        output = []
        for image_index in range(detections.shape[0]):
            det = detections[image_index]
            valid = torch.isfinite(det[:, :6]).all(1) & (det[:, 4] >= self.args.conf)
            det = det[valid]
            selected_cube = {name: value[image_index][valid] for name, value in cube_preds.items()}
            box_format = self.data.get("detect3d_pred_box_format", "xywh")
            boxes = ops.xywh2xyxy(det[:, :4]) if box_format == "xywh" else det[:, :4]
            output.append({
                "bboxes": boxes,
                "conf": det[:, 4],
                "cls": det[:, 5].long(),
                "cube_preds": selected_cube,
            })
        return output

    def _decode_image(self, pred, batch, image_index):
        n = len(pred["conf"])
        if n == 0:
            return None
        classes = pred["cls"].long()
        boxes = pred["bboxes"]
        K = batch["K"][image_index].unsqueeze(0).expand(n, -1, -1)
        K_orig = batch["K_orig"][image_index].unsqueeze(0).expand(n, -1, -1)

        orig_scale = batch["im_scales_orig"][image_index].unsqueeze(0).expand(n, -1)
        input_scale = batch["im_scales"][image_index].unsqueeze(0).expand(n, -1)

        decoded = self.head.decode_cube(
            cube_preds=pred["cube_preds"],
            box_classes=classes,
            src_boxes=boxes,
            Ks_scaled_per_box=K,
            focal_lengths=K_orig[:, 1, 1],
            im_scales_orig=orig_scale,
            im_scales=input_scale,
        )
        return decoded

    def update_metrics(self, preds, batch):
        for image_index, pred in enumerate(preds):
            self.seen += 1
            if len(pred["conf"]) == 0:
                continue
            decoded = self._decode_image(pred, batch, image_index)
            boxes3d = torch.cat((decoded["center_cam"], decoded["dims"]), dim=1)

            vertices, _ = get_cuboid_verts_faces(boxes3d, decoded["pose"])

            scale = batch["im_scales_orig"][image_index]
            ori_shape = (int(scale[0]), int(scale[1]))
            boxes_original = ops.scale_boxes(batch["img"].shape[2:], pred["bboxes"].clone(), ori_shape)
            image_id = int(batch["image_id"][image_index])

            for row in range(len(pred["conf"])):
                cls = int(pred["cls"][row])
                x1, y1, x2, y2 = boxes_original[row].tolist()
                w = max(x2 - x1, 0.0)
                h = max(y2 - y1, 0.0)
                depth_val = float(decoded["center_cam"][row, 2].item())

                self.predictions.append({
                    "image_id": image_id,
                    "category_id": int(self._contiguous_to_raw[cls]),
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "score": float(pred["conf"][row]),
                    "depth": depth_val,
                    "bbox3D": vertices[row].detach().cpu().tolist(),
                    "center_cam": decoded["center_cam"][row].detach().cpu().tolist(),
                    "dimensions": decoded["dims"][row].detach().cpu().tolist(),
                    "pose": decoded["pose"][row].detach().cpu().tolist(),
                })

    def gather_stats(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world = torch.distributed.get_world_size()
        gathered = [None] * world if RANK == 0 else None
        torch.distributed.gather_object(self.predictions, gathered, dst=0)
        if RANK == 0:
            self.predictions = [item for rank_items in gathered for item in rank_items]
        else:
            self.predictions = []

    def _get_gt(self, json_files: list[str] | None = None, dataset_idx_offset: int = 0) -> Omni3DValidationGT:
        target_files = json_files or self._json_files
        data_dict = getattr(self, "data", {}) or {}
        filter_settings = dict(
            data_dict.get("val_filter_settings")
            or data_dict.get("filter_settings")
            or {}
        )
        filter_settings.setdefault("category_names", [self.names[i] for i in range(self.nc)])
        filter_settings.setdefault("ignore_names", [])
        filter_settings.setdefault("truncation_thres", 0.75)
        filter_settings.setdefault("visibility_thres", 0.33333)
        filter_settings.setdefault("min_height_thres", 0.00)
        filter_settings.setdefault("max_height_thres", 1.50)
        filter_settings.setdefault("modal_2D_boxes", False)
        filter_settings.setdefault("trunc_2D_boxes", True)
        filter_settings.setdefault("max_depth", 512.0)

        return Omni3DValidationGT(target_files, filter_settings, dataset_idx_offset=dataset_idx_offset)

    @staticmethod
    def _extract_category_aps(evaluator, category_ids, contiguous_to_name):
        results_cat = OrderedDict()
        if not evaluator or not getattr(evaluator, "eval", None):
            return results_cat

        precision = evaluator.eval.get("precision")
        if precision is None:
            return results_cat

        for idx, cat_id in enumerate(category_ids):
            name = contiguous_to_name.get(cat_id, f"cat_{cat_id}")
            vals = precision[:, :, idx, 0, -1]
            valid = vals[vals > -1]
            ap = float(np.mean(valid) * 100.0) if valid.size else float("nan")
            results_cat[name] = ap
        return results_cat

    def _evaluate_subset(self, json_file: str, preds: list[dict], dataset_name: str, dataset_idx: int = 0):
        try:
            from cubercnn.evaluation.omni3d_evaluation import Omni3Deval
        except ImportError:
            from cubercnn.evaluation.omni3d_evaluation import Omni3DEval as Omni3Deval

        gt = self._get_gt([json_file], dataset_idx_offset=dataset_idx)
        if len(preds) == 0:
            return None, None, {}, {}

        dt = gt.loadRes(preds)
        eval_2d = Omni3Deval(gt, dt, iouType="bbox", mode="2D")
        eval_2d.params.catIds = sorted(self._id_map.keys())
        eval_2d.evaluate()
        eval_2d.accumulate()
        eval_2d.summarize()

        eval_3d = Omni3Deval(gt, dt, iouType="bbox", mode="3D")
        eval_3d.params.catIds = sorted(self._id_map.keys())
        eval_3d.evaluate()
        eval_3d.accumulate()
        eval_3d.summarize()

        cat_ids = sorted(self._id_map.keys())
        c2name = {cid: self.names[self._id_map[cid]] for cid in cat_ids}
        cat_aps_2d = self._extract_category_aps(eval_2d, cat_ids, c2name)
        cat_aps_3d = self._extract_category_aps(eval_3d, cat_ids, c2name)

        return eval_2d, eval_3d, cat_aps_2d, cat_aps_3d

    def get_stats(self):
        self.gather_stats()

        if RANK not in {-1, 0}:
            return {}

        try:
            from cubercnn.evaluation.omni3d_evaluation import Omni3Deval
        except ImportError:
            from cubercnn.evaluation.omni3d_evaluation import Omni3DEval as Omni3Deval

        results_analysis = OrderedDict()
        results_omni3d = OrderedDict()
        iter_label = "final"

        # -------------------------------------------------------------
        # 1. 依據資料集分別進行評估並輸出單一資料集直方圖
        # -------------------------------------------------------------
        for idx, json_path in enumerate(self._json_files):
            d_name = Path(json_path).stem
            id_min = idx * 1_000_000
            id_max = (idx + 1) * 1_000_000
            subset_preds = [p for p in self.predictions if id_min <= p["image_id"] < id_max]

            e2d, e3d, c_2d, c_3d = self._evaluate_subset(json_path, subset_preds, d_name, dataset_idx=idx)

            if e2d is not None and e3d is not None:
                single_results_cat = OrderedDict()
                for c_name in c_2d:
                    v2, v3 = c_2d.get(c_name, np.nan), c_3d.get(c_name, np.nan)
                    if not (np.isnan(v2) and np.isnan(v3)):
                        single_results_cat[c_name] = {"AP2D": v2, "AP3D": v3}

                # 表一：個別資料集類別表現
                if single_results_cat:
                    utils_logperf.print_ap_category_histogram(d_name, single_results_cat)

                s2d = np.asarray(e2d.stats, dtype=float)
                s3d = np.asarray(e3d.stats, dtype=float)
                ap2d_val = float(s2d[0] * 100.0) if s2d.size else 0.0
                ap3d_val = float(s3d[0] * 100.0) if s3d.size else 0.0

                results_analysis[d_name] = {
                    "iters": iter_label,
                    "AP2D": ap2d_val,
                    "AP3D": ap3d_val,
                    "AP3D@15": float(s3d[1] * 100.0) if s3d.size > 1 else np.nan,
                    "AP3D@25": float(s3d[2] * 100.0) if s3d.size > 2 else np.nan,
                    "AP3D@50": float(s3d[3] * 100.0) if s3d.size > 3 else np.nan,
                    "AP3D-N": float(s3d[4] * 100.0) if s3d.size > 4 else np.nan,
                    "AP3D-M": float(s3d[5] * 100.0) if s3d.size > 5 else np.nan,
                    "AP3D-F": float(s3d[6] * 100.0) if s3d.size > 6 else np.nan,
                }
                results_omni3d[d_name] = {
                    "iters": iter_label,
                    "AP2D": ap2d_val,
                    "AP3D": ap3d_val,
                }

        # -------------------------------------------------------------
        # 2. 全部資料集合併評估 (<Concat>)
        # -------------------------------------------------------------
        gt_all = self._get_gt(self._json_files)
        dt_all = gt_all.loadRes(self.predictions) if len(self.predictions) > 0 else None

        if dt_all:
            eval_2d_all = Omni3Deval(gt_all, dt_all, iouType="bbox", mode="2D")
            eval_2d_all.params.catIds = sorted(self._id_map.keys())
            eval_2d_all.evaluate()
            eval_2d_all.accumulate()
            eval_2d_all.summarize()

            eval_3d_all = Omni3Deval(gt_all, dt_all, iouType="bbox", mode="3D")
            eval_3d_all.params.catIds = sorted(self._id_map.keys())
            eval_3d_all.evaluate()
            eval_3d_all.accumulate()
            eval_3d_all.summarize()

            s2d_all = np.asarray(eval_2d_all.stats, dtype=float)
            s3d_all = np.asarray(eval_3d_all.stats, dtype=float)
            concat_ap2d = float(s2d_all[0] * 100.0)
            concat_ap3d = float(s3d_all[0] * 100.0)

            cat_ids = sorted(self._id_map.keys())
            c2name = {cid: self.names[self._id_map[cid]] for cid in cat_ids}
            c_2d_all = self._extract_category_aps(eval_2d_all, cat_ids, c2name)
            c_3d_all = self._extract_category_aps(eval_3d_all, cat_ids, c2name)

            concat_results_cat = OrderedDict()
            for c_name in c_2d_all:
                concat_results_cat[c_name] = {
                    "AP2D": c_2d_all.get(c_name, np.nan),
                    "AP3D": c_3d_all.get(c_name, np.nan),
                }

            # 表二：<Concat> 完整 38 類大表
            utils_logperf.print_ap_category_histogram("<Concat>", concat_results_cat)

            results_analysis["<Concat>"] = {
                "iters": iter_label,
                "AP2D": concat_ap2d,
                "AP3D": concat_ap3d,
                "AP3D@15": float(s3d_all[1] * 100.0),
                "AP3D@25": float(s3d_all[2] * 100.0),
                "AP3D@50": float(s3d_all[3] * 100.0),
                "AP3D-N": float(s3d_all[4] * 100.0),
                "AP3D-M": float(s3d_all[5] * 100.0),
                "AP3D-F": float(s3d_all[6] * 100.0),
            }

            # 表三：多資料集性能分析大表
            utils_logperf.print_ap_analysis_histogram(results_analysis)

            # 表四：Omni3D 官方基準大表
            results_omni3d["Omni3D_Out"] = {"iters": iter_label, "AP2D": np.nan, "AP3D": np.nan}
            results_omni3d["Omni3D_In"] = {"iters": iter_label, "AP2D": concat_ap2d, "AP3D": concat_ap3d}
            results_omni3d["Omni3D"] = {"iters": iter_label, "AP2D": np.nan, "AP3D": np.nan}
            utils_logperf.print_ap_omni_histogram(results_omni3d)

            # 強制刷新終端緩衝區
            sys.stdout.flush()

            self.metric_results = {
                "metrics/AP2D": float(s2d_all[0]),
                "metrics/AP2D50": float(s2d_all[1]),
                "metrics/AP3D": float(s3d_all[0]),
                "metrics/AP3D15": float(s3d_all[1]),
                "metrics/AP3D25": float(s3d_all[2]),
                "metrics/AP3D50": float(s3d_all[3]),
                "metrics/AP3D-N": float(s3d_all[4]),
                "metrics/AP3D-M": float(s3d_all[5]),
                "metrics/AP3D-F": float(s3d_all[6]),
                "fitness": float(s3d_all[0]),
            }
        else:
            self.metric_results = {k: 0.0 for k in self.metric_keys}

        if self.loss:
            num_batches = max(len(self.dataloader), 1)
            for k, val in self.loss.items():
                v = float(val.item() if torch.is_tensor(val) else val) / num_batches
                self.metric_results[f"val/{k}"] = round(v, 5)

        return self.metric_results

    def print_results(self):
        pass

    def finalize_metrics(self):
        pass