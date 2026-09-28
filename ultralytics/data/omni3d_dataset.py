from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch

from ultralytics.utils.instance import Instances
from ultralytics.utils.cube_utils import pick_2d_box, is_ignore
from .augment import BaseTransform, Compose, Format, LetterBox
from .base import BaseDataset

CACHE_VERSION = "1.0.7"

DEFAULT_FILTER_SETTINGS = {
    "category_names": [],
    "ignore_names": [],
    "truncation_thres": 0.99,
    "visibility_thres": 0.01,
    "min_height_thres": 0.00,
    "max_height_thres": 1.50,
    "modal_2D_boxes": False,
    "trunc_2D_boxes": True,
    "max_depth": 512.0,
}


# -------------------------------------------------------------
# 3D Transform
# -------------------------------------------------------------
_FLIP_M1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
_FLIP_M2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

def mirror_rotation(R: np.ndarray) -> np.ndarray:
    return _FLIP_M1 @ R @ _FLIP_M2


class RandomFlip3D(BaseTransform):
    """Horizontal flip with image and 2D/3D annotation"""
    def __init__(self, p: float = 0.5, mirror_center_x: bool = True):
        self.p = p
        self.mirror_center_x = mirror_center_x

    def __call__(self, labels: dict) -> dict:
        if random.random() >= self.p:
            return labels

        img = labels["img"]
        w = img.shape[1]
        labels["img"] = np.ascontiguousarray(img[:, ::-1])

        # Use normalized image to flip
        inst = labels["instances"]
        inst.fliplr(1 if inst.normalized else w)

        K = labels["K"].copy()
        K[0, 2] = w - K[0, 2]
        labels["K"] = K

        R_cam = labels["R_cam"]
        if len(R_cam):
            labels["R_cam"] = np.stack([mirror_rotation(R) for R in R_cam])

        if self.mirror_center_x and len(labels["center_cam"]):
            center_cam = labels["center_cam"].copy()
            center_cam[:, 0] *= -1
            labels["center_cam"] = center_cam

        center_2d = labels.get("center_2D")
        if center_2d is not None and len(center_2d):
            center_2d = center_2d.copy()
            center_2d[:, 0] = w - center_2d[:, 0]
            labels["center_2D"] = center_2d

        return labels


class LetterBox3D(LetterBox):
    """Pad image and annotations to target size"""
    def __call__(self, labels: dict) -> dict:
        h0, w0 = labels["img"].shape[:2]
        labels = super().__call__(labels)

        new_h, new_w = self.new_shape if isinstance(self.new_shape, tuple) else (self.new_shape, self.new_shape)
        r = min(new_h / h0, new_w / w0)
        if not self.scaleup:
            r = min(r, 1.0)
        new_unpad_w, new_unpad_h = round(w0 * r), round(h0 * r)
        dw, dh = (new_w - new_unpad_w) / 2, (new_h - new_unpad_h) / 2
        left, top = round(dw - 0.1), round(dh - 0.1)

        K = labels["K"].copy()
        K[0, 0] *= r
        K[1, 1] *= r
        K[0, 2] = K[0, 2] * r + left
        K[1, 2] = K[1, 2] * r + top
        labels["K"] = K

        center_2d = labels.get("center_2D")
        if center_2d is not None and len(center_2d):
            center_2d = center_2d.copy()
            center_2d[:, 0] = center_2d[:, 0] * r + left
            center_2d[:, 1] = center_2d[:, 1] * r + top
            labels["center_2D"] = center_2d

        labels["im_scales"] = np.array([r, r], dtype=np.float32)
        return labels


class Albumentations3D(BaseTransform):
    def __init__(self, p: float = 1.0, transforms: list | None = None) -> None:
        self.p = p
        self.transform = None

        try:
            import os
            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
            import albumentations as A

            T = (
                [
                    A.CLAHE(p=0.01),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
                ]
                if transforms is None
                else transforms
            )

            for t in T:
                if isinstance(t, A.DualTransform):
                    raise ValueError(
                        f"[Albumentations3D] 檢測到空間幾何變換算子: {t.__class__.__name__}。"
                        "在 3D 任務中請勿混入空間算子，以避免破壞相機內參 K！"
                    )

            self.transform = A.Compose(T)
        except ImportError:
            pass

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.transform is None or random.random() >= self.p:
            return labels

        im = labels["img"]
        if im.shape[2] not in {1, 3}:
            return labels

        labels["img"] = self.transform(image=im)["image"]
        return labels


def _parse_one_json(json_file: str, filter_settings: dict, id_map: dict, dataset_idx: int = 0) -> dict:
    with open(json_file) as f:
        data = json.load(f)

    images = data["images"]
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    file_paths, widths, heights, Ks, image_ids = [], [], [], [], []
    ann_image_idx, ann_cat_id = [], []
    ann_bbox, ann_dims, ann_center, ann_R, ann_center2d, ann_ignore = [], [], [], [], [], []

    # ID shift 避免多資料集碰撞
    id_offset = dataset_idx * 1_000_000

    for img_idx, im in enumerate(images):
        file_paths.append(im["file_path"])
        image_ids.append(int(im["id"]) + id_offset)
        widths.append(im["width"])
        heights.append(im["height"])
        Ks.append(im["K"])

        for anno in anns_by_img.get(im["id"], []):
            cat_id = anno["category_id"]
            if cat_id not in id_map and anno.get("category_name") not in filter_settings.get("ignore_names", []):
                continue

            box2d = pick_2d_box(anno, filter_settings)
            # 剔除無效的 2D 框結構
            if box2d is None or len(box2d) != 4:
                continue

            # 寬或高小於等於 0，直接略過
            if box2d[2] <= 0.0 or box2d[3] <= 0.0:
                continue

            cx3d, cy3d, z3d = anno["center_cam"]
            k_img = im["K"]
            safe_z = z3d if abs(z3d) > 1e-6 else (1e-6 if z3d >= 0 else -1e-6)
            proj_x = k_img[0][0] * cx3d / safe_z + k_img[0][2]
            proj_y = k_img[1][1] * cy3d / safe_z + k_img[1][2]
            center_2d = [proj_x, proj_y]

            ignore = is_ignore(anno, filter_settings, im["height"])

            # 防禦：退化框（寬或高小於 2 像素）標記為 ignore，避免邊界孤立標籤
            if box2d[2] < 12.0 or box2d[3] < 12.0:
                ignore = True

            mapped_cat_id = id_map.get(cat_id, -1)
            if mapped_cat_id < 0:
                ignore = True

            ann_image_idx.append(img_idx)
            ann_cat_id.append(-1 if ignore else mapped_cat_id)
            ann_bbox.append(box2d)
            ann_dims.append(anno["dimensions"])
            ann_center.append(anno["center_cam"])
            ann_R.append(np.array(anno["R_cam"], dtype=np.float32).reshape(-1))
            ann_center2d.append(center_2d)
            ann_ignore.append(ignore)

    n_ann = len(ann_image_idx)
    return {
        "file_path": np.array(file_paths, dtype=object),
        "image_id": np.array(image_ids, dtype=np.int64),
        "width": np.array(widths, dtype=np.int32),
        "height": np.array(heights, dtype=np.int32),
        "K": np.array(Ks, dtype=np.float32).reshape(-1, 3, 3),
        "ann_image_idx": np.array(ann_image_idx, dtype=np.int32),
        "ann_cat_id": np.array(ann_cat_id, dtype=np.int16),
        "ann_bbox": np.array(ann_bbox, dtype=np.float32).reshape(n_ann, 4),
        "ann_dims": np.array(ann_dims, dtype=np.float32).reshape(n_ann, 3),
        "ann_center": np.array(ann_center, dtype=np.float32).reshape(n_ann, 3),
        "ann_R": np.array(ann_R, dtype=np.float32).reshape(n_ann, 9),
        "ann_center2d": np.array(ann_center2d, dtype=np.float32).reshape(n_ann, 2),
        "ann_ignore": np.array(ann_ignore, dtype=bool),
    }


def _cache_path_for(json_file: str, filter_settings: dict, id_map: dict, dataset_idx: int, cache_dir: Path) -> Path:
    st = os.stat(json_file)
    key = (
        f"{json_file}:{st.st_mtime}:{st.st_size}:"
        f"{json.dumps(filter_settings, sort_keys=True)}:"
        f"{json.dumps(id_map, sort_keys=True)}:"
        f"{dataset_idx}:{CACHE_VERSION}"
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return cache_dir / f"{Path(json_file).stem}.{digest}.npz"


def _load_or_parse_one_json(json_file: str, filter_settings: dict, id_map: dict, cache_dir: Path, dataset_idx: int = 0) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path_for(json_file, filter_settings, id_map, dataset_idx, cache_dir)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}

    arrays = _parse_one_json(json_file, filter_settings, id_map, dataset_idx=dataset_idx)
    np.savez_compressed(cache_file, **arrays)
    return arrays


class Omni3DDataset(BaseDataset):
    def __init__(self, *args, json_files, img_path, id_map,
                 filter_settings=None, cache_dir=None, filter_empty=True,
                 fliplr_p=0.0, mirror_center_x=False, **kwargs):
        self.json_files = json_files
        self.id_map = id_map
        self.filter_settings = {**DEFAULT_FILTER_SETTINGS, **(filter_settings or {})}
        self.filter_empty = filter_empty
        self.cache_dir = Path(cache_dir or Path(json_files[0]).parent)
        self.fliplr_p = fliplr_p
        self.mirror_center_x = mirror_center_x

        self._per_json = [
            _load_or_parse_one_json(jf, self.filter_settings, self.id_map, self.cache_dir, dataset_idx=i)
            for i, jf in enumerate(json_files)
        ]
        self._entries = self._build_entries()

        kwargs.setdefault("rect", False)
        super().__init__(*args, img_path=img_path, **kwargs)

    def _build_entries(self):
        entries = []
        for arrs in self._per_json:
            by_img = defaultdict(list)
            for i, img_idx in enumerate(arrs["ann_image_idx"]):
                by_img[int(img_idx)].append(i)

            for img_idx in range(len(arrs["file_path"])):
                idxs = by_img.get(img_idx, [])
                
                # 關鍵落實：只有具備「非 ignore、類別合法、寬高 >= 2」的有效目標時，才算有效影像
                has_valid = any(
                    (not arrs["ann_ignore"][i])
                    and (arrs["ann_cat_id"][i] >= 0)
                    and (arrs["ann_bbox"][i][2] >= 12.0)
                    and (arrs["ann_bbox"][i][3] >= 12.0)
                    for i in idxs
                )
                
                # 當開啟 filter_empty 時，徹底排除純背景影像
                if self.filter_empty and not has_valid:
                    continue

                entries.append({
                    "file_path": str(arrs["file_path"][img_idx]),
                    "image_id": int(arrs["image_id"][img_idx]),
                    "width": int(arrs["width"][img_idx]),
                    "height": int(arrs["height"][img_idx]),
                    "K": arrs["K"][img_idx].copy(),
                    "cls": arrs["ann_cat_id"][idxs].astype(np.float32).reshape(-1, 1)
                        if idxs else np.zeros((0, 1), np.float32),
                    "bbox": arrs["ann_bbox"][idxs] if idxs else np.zeros((0, 4), np.float32),
                    "dimensions": arrs["ann_dims"][idxs] if idxs else np.zeros((0, 3), np.float32),
                    "center_cam": arrs["ann_center"][idxs] if idxs else np.zeros((0, 3), np.float32),
                    "R_cam": arrs["ann_R"][idxs].reshape(-1, 3, 3) if idxs else np.zeros((0, 3, 3), np.float32),
                    "center_2D": arrs["ann_center2d"][idxs] if idxs else np.zeros((0, 2), np.float32),
                    "ignore": arrs["ann_ignore"][idxs] if idxs else np.zeros((0,), bool),
                })
        return entries

    def get_img_files(self, img_path):
        return [str(Path(img_path) / e["file_path"]) for e in self._entries]

    def get_labels(self):
        labels = []
        for i, e in enumerate(self._entries):
            n = len(e["bbox"])
            if n:
                x, y, w, h = e["bbox"][:, 0], e["bbox"][:, 1], e["bbox"][:, 2], e["bbox"][:, 3]
                cx = np.clip((x + w / 2.0) / e["width"], 0.0, 1.0)
                cy = np.clip((y + h / 2.0) / e["height"], 0.0, 1.0)
                w_norm = np.clip(w / e["width"], 0.0, 1.0)
                h_norm = np.clip(h / e["height"], 0.0, 1.0)
                bboxes = np.stack([cx, cy, w_norm, h_norm], axis=1).astype(np.float32)
            else:
                bboxes = np.zeros((0, 4), np.float32)

            labels.append({
                "im_file": self.im_files[i],
                "shape": (e["height"], e["width"]),
                "image_id": e["image_id"],
                "cls": e["cls"],
                "bboxes": bboxes,
                "normalized": True,
                "bbox_format": "xywh",
                "K": e["K"],
                "dimensions": e["dimensions"],
                "center_cam": e["center_cam"],
                "R_cam": e["R_cam"],
                "center_2D": e["center_2D"],
                "ignore": e["ignore"],
                "K_orig": e["K"].copy(),
                "im_scales_orig": np.array([e["height"], e["width"]], dtype=np.float32),
            })
        return labels

    def update_labels_info(self, label):
        bboxes = label.pop("bboxes")
        label["instances"] = Instances(bboxes, bbox_format="xywh", normalized=True)
        return label

    def build_transforms(self, hyp=None):
        transforms = Compose([
            Albumentations3D(p=1.0 if self.augment else 0.0),
            RandomFlip3D(p=self.fliplr_p if self.augment else 0.0, mirror_center_x=self.mirror_center_x),
            LetterBox3D(new_shape=(self.imgsz, self.imgsz), scaleup=self.augment),
        ])
        transforms.append(Format(bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False))
        return transforms

    @staticmethod
    def collate_fn(batch):
        if not batch:
            raise ValueError("[Omni3D] collate_fn received an empty batch")

        new_batch = {
            "img": torch.stack([torch.as_tensor(s["img"]) for s in batch], 0),
            "K": torch.stack([torch.as_tensor(s["K"], dtype=torch.float32) for s in batch], 0),
            "K_orig": torch.stack([torch.as_tensor(s["K_orig"], dtype=torch.float32) for s in batch], 0),
            "im_scales": torch.stack([torch.as_tensor(s["im_scales"], dtype=torch.float32) for s in batch], 0),
            "im_scales_orig": torch.stack([torch.as_tensor(s["im_scales_orig"], dtype=torch.float32) for s in batch], 0),
        }

        flat = {"cls": [], "bboxes": [], "dimensions": [], "center_cam": [], "R_cam": [], "center_2D": []}
        batch_indices, ignored_bboxes, ignored_batch_indices = [], [], []

        for image_idx, sample in enumerate(batch):
            cls = torch.as_tensor(sample["cls"], dtype=torch.float32).reshape(-1, 1)
            bboxes = torch.as_tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
            dimensions = torch.as_tensor(sample["dimensions"], dtype=torch.float32).reshape(-1, 3)
            center_cam = torch.as_tensor(sample["center_cam"], dtype=torch.float32).reshape(-1, 3)
            R_cam = torch.as_tensor(sample["R_cam"], dtype=torch.float32).reshape(-1, 3, 3)
            center_2d = torch.as_tensor(sample["center_2D"], dtype=torch.float32).reshape(-1, 2)
            ignore = torch.as_tensor(sample["ignore"], dtype=torch.bool).reshape(-1)

            valid = (~ignore) & (cls[:, 0] >= 0)
            ignored = ~valid

            flat["cls"].append(cls[valid])
            flat["bboxes"].append(bboxes[valid])
            flat["dimensions"].append(dimensions[valid])
            flat["center_cam"].append(center_cam[valid])
            flat["R_cam"].append(R_cam[valid])
            flat["center_2D"].append(center_2d[valid])
            batch_indices.append(torch.full((int(valid.sum()),), image_idx, dtype=torch.long))

            ignored_bboxes.append(bboxes[ignored])
            ignored_batch_indices.append(torch.full((int(ignored.sum()),), image_idx, dtype=torch.long))

        empty_shapes = {
            "cls": (0, 1), "bboxes": (0, 4), "dimensions": (0, 3),
            "center_cam": (0, 3), "R_cam": (0, 3, 3), "center_2D": (0, 2),
        }
        for key, parts in flat.items():
            new_batch[key] = torch.cat(parts, 0) if parts else torch.empty(empty_shapes[key])

        new_batch["batch_idx"] = torch.cat(batch_indices, 0) if batch_indices else torch.empty((0,), dtype=torch.long)
        new_batch["gt_boxes3D"] = torch.cat([new_batch["center_cam"], new_batch["dimensions"]], dim=1)
        new_batch["gt_poses"] = new_batch["R_cam"]
        new_batch["gt_2D"] = new_batch["center_2D"]

        handled = {
            "img", "K", "K_orig", "im_scales", "im_scales_orig",
            "cls", "bboxes", "dimensions", "center_cam", "R_cam", "center_2D", "ignore",
            "batch_idx", "gt_boxes3D", "gt_poses", "gt_2D",
        }
        all_keys = set().union(*(s.keys() for s in batch))
        for key in sorted(all_keys - handled):
            new_batch[key] = [s.get(key) for s in batch]

        return new_batch