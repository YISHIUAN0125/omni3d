"""Source-close Cube R-CNN prior computation adapted only at the input boundary.

The arithmetic, resize rule, virtual-depth transform, category statistics,
scale-cluster initialization/update, and low-sample fallback follow Cube R-CNN.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch


def approx_eval_resolution(h, w, scale_min=0, scale_max=1e10):
    orig_h = h
    sf = scale_min / min(h, w)
    h *= sf; w *= sf
    sf = min(scale_max / max(h, w), 1.0)
    h *= sf; w *= sf
    return h, w, h / orig_h


def compute_virtual_scale_from_focal_spaces(f, H, f0, H0):
    return H0 * f / (f0 * H)


class Omni3DPriorDatasetAdapter:
    """COCO-like adapter expected by the original Cube R-CNN prior routine."""
    def __init__(self, json_files, id_map, filter_settings):
        self.imgs, self._anns = {}, []
        next_image_id = 1
        for dataset_id, path in enumerate(map(Path, json_files)):
            data = json.loads(path.read_text(encoding="utf-8"))
            cats = {int(c["id"]): c.get("name", str(c["id"])) for c in data.get("categories", [])}
            image_remap = {}
            for image in data["images"]:
                new_id = next_image_id; next_image_id += 1
                image_remap[int(image["id"])] = new_id
                self.imgs[new_id] = dict(image)
            for ann in data["annotations"]:
                raw = int(ann["category_id"])
                if raw not in id_map:
                    continue
                item = dict(ann)
                item["id"] = len(self._anns) + 1
                item["image_id"] = image_remap[int(ann["image_id"])]
                item["dataset_id"] = dataset_id
                item["category_name"] = cats.get(raw, str(raw)).lower()
                item["category_id"] = int(id_map[raw])
                image = self.imgs[item["image_id"]]
                item["ignore"] = _official_ignore_equivalent(item, filter_settings, int(image["height"]))
                self._anns.append(item)
    def getAnnIds(self):
        return [a["id"] for a in self._anns]
    def loadAnns(self, ids):
        wanted = set(ids)
        return [a for a in self._anns if a["id"] in wanted]


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _official_box(ann, settings, allow_missing=False):
    if settings.get("modal_2D_boxes") and "bbox2D_tight" in ann and ann["bbox2D_tight"][0] != -1:
        return _xyxy_to_xywh(ann["bbox2D_tight"])
    if settings.get("trunc_2D_boxes") and "bbox2D_trunc" in ann and not np.all([v == -1 for v in ann["bbox2D_trunc"]]):
        return _xyxy_to_xywh(ann["bbox2D_trunc"])
    if "bbox2D_proj" in ann:
        return _xyxy_to_xywh(ann["bbox2D_proj"])
    return None if allow_missing else ann.get("bbox")


def _official_ignore_equivalent(ann, settings, image_h):
    ignore = bool(ann.get("behind_camera", False)) or not bool(ann.get("valid3D", True))
    if ignore:
        return True
    d, c = ann["dimensions"], ann["center_cam"]
    ignore |= d[0] <= 0 or d[1] <= 0 or d[2] <= 0
    ignore |= c[2] > settings.get("max_depth", 1e8)
    ignore |= ann.get("lidar_pts", 1) == 0 or ann.get("segmentation_pts", 1) == 0
    ignore |= ann.get("depth_error", 0) > 0.5
    box = _official_box(ann, settings, allow_missing=True) or ann.get("bbox")
    ignore |= box[3] <= settings.get("min_height_thres", 0.0) * image_h
    ignore |= box[3] >= settings.get("max_height_thres", 1.5) * image_h
    trunc, vis = ann.get("truncation", -1), ann.get("visibility", -1)
    ignore |= trunc >= 0 and trunc >= settings.get("truncation_thres", 0.99)
    ignore |= vis >= 0 and vis <= settings.get("visibility_thres", 0.01)
    ignore |= ann.get("category_name") in settings.get("ignore_names", [])
    return bool(ignore)


def make_cfg(virtual_depth, virtual_focal, test_scale_min, test_scale_max,
             cluster_bins, anchor_sizes, modal_2d=False, trunc_2d=False):
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            ROI_CUBE_HEAD=SimpleNamespace(VIRTUAL_DEPTH=virtual_depth, VIRTUAL_FOCAL=virtual_focal, CLUSTER_BINS=cluster_bins),
            ANCHOR_GENERATOR=SimpleNamespace(SIZES=anchor_sizes),
        ),
        INPUT=SimpleNamespace(MIN_SIZE_TEST=test_scale_min, MAX_SIZE_TEST=test_scale_max),
        DATASETS=SimpleNamespace(MODAL_2D_BOXES=modal_2d, TRUNC_2D_BOXES=trunc_2d),
    )


def compute_priors(cfg, datasets, category_names, max_cluster_rounds=1000, min_points_for_std=5):
    """Cube R-CNN compute_priors with only MetadataCatalog removed."""
    anns = datasets.loadAnns(datasets.getAnnIds())
    data_raw = []
    virtual_depth = cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH
    virtual_focal = cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL
    test_scale_min, test_scale_max = cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
    for ann in anns:
        category_name = ann["category_name"].lower()
        image_id = ann["image_id"]
        fy = datasets.imgs[image_id]["K"][1][1]
        im_h, im_w = datasets.imgs[image_id]["height"], datasets.imgs[image_id]["width"]
        f = 2 * fy / im_h
        box = _official_box(ann, {"modal_2D_boxes": cfg.DATASETS.MODAL_2D_BOXES, "trunc_2D_boxes": cfg.DATASETS.TRUNC_2D_BOXES}, allow_missing=True)
        if box is None:
            continue
        x, y, w, h = box
        x3d, y3d, z3d = ann["center_cam"]
        w3d, h3d, l3d = ann["dimensions"]
        test_h, test_w, sf = approx_eval_resolution(im_h, im_w, test_scale_min, test_scale_max)
        h *= sf; w *= sf
        if virtual_depth:
            virtual_to_real = compute_virtual_scale_from_focal_spaces(fy, im_h, virtual_focal, test_h)
            z3d *= 1.0 / virtual_to_real
        scale = np.sqrt(w ** 2 + h ** 2)
        data_raw.append([category_name, ann["ignore"], ann["dataset_id"], f, x3d, y3d, z3d, w3d, h3d, l3d, scale])
    df = pd.DataFrame(data_raw, columns=["category","ignore","dataset","f","x3d","y3d","z3d","w3d","h3d","l3d","scale"])
    df = df.loc[df.ignore == False]
    priors_y3d = [df.y3d.mean(), df.y3d.std()]
    priors_z3d = [df.z3d.mean(), df.z3d.std()]
    n_bins = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
    priors_bins, priors_dims_per_cat, priors_z3d_per_cat, priors_y3d_per_cat = [], [], [], []
    for cat in category_names:
        df_cat = df.loc[df.category == cat]
        n = len(df_cat)
        if n > 0:
            priors_dims_per_cat.append([[df_cat.w3d.mean(),df_cat.h3d.mean(),df_cat.l3d.mean()],[df_cat.w3d.std(),df_cat.h3d.std(),df_cat.l3d.std()]])
            priors_z3d_per_cat.append([df_cat.z3d.mean(),df_cat.z3d.std()])
            priors_y3d_per_cat.append([df_cat.y3d.mean(),df_cat.y3d.std()])
        else:
            priors_dims_per_cat.append([[1.,1.,1.],[1.,1.,1.]])
            priors_z3d_per_cat.append([50,50]); priors_y3d_per_cat.append([1,10])
        def cluster_mean(scales, assignments, bins, quality):
            result=[]
            for b in range(bins):
                inside=assignments==b
                if inside.sum()<min_points_for_std:
                    inside[quality[:,b].topk(min_points_for_std)[1]]=True
                result.append(scales[inside].mean().item())
            return torch.FloatTensor(result)
        if n_bins > 1:
            if n < min_points_for_std:
                max_scale=cfg.MODEL.ANCHOR_GENERATOR.SIZES[-1][-1]; min_scale=cfg.MODEL.ANCHOR_GENERATOR.SIZES[0][0]
                base=(max_scale/min_scale)**(1/(n_bins-1)); clusters=np.array([min_scale*(base**i) for i in range(n_bins)])
                priors_bins.append((cat,clusters.tolist(),[[1,10] for _ in range(n_bins)]))
            else:
                scales=torch.FloatTensor(df_cat.scale.values)
                max_scale=scales.max(); min_scale=scales.min(); base=(max_scale/min_scale)**(1/(n_bins-1))
                clusters=torch.FloatTensor([min_scale*(base**i) for i in range(n_bins)])
                best=-np.inf
                for _ in range(max_cluster_rounds):
                    quality=-(clusters.unsqueeze(0)-scales.unsqueeze(1)).abs(); scores,new_assign=quality.max(1); score=scores.mean().item()
                    if np.round(score,5)>best:
                        best=score; assignments=new_assign; clusters=cluster_mean(scales,assignments,n_bins,quality)
                    else: break
                stats=[]
                for b in range(n_bins):
                    inside=assignments==b
                    if inside.sum()<min_points_for_std: inside[quality[:,b].topk(min_points_for_std)[1]]=True
                    inside=inside.numpy(); stats.append([df_cat.z3d[inside].mean(),df_cat.z3d[inside].std()])
                priors_bins.append((cat,clusters.numpy().tolist(),stats))
    return {"priors_dims_per_cat":priors_dims_per_cat,"priors_z3d_per_cat":priors_z3d_per_cat,"priors_y3d_per_cat":priors_y3d_per_cat,"priors_bins":priors_bins,"priors_y3d":priors_y3d,"priors_z3d":priors_z3d}
