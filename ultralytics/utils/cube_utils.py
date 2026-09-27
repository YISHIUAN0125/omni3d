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
from typing import Any
from ultralytics.utils import LOGGER

def _valid_xyxy(box) -> bool:
    if box is None or len(box) != 4:
        return False
    if any(v is None or v == -1 for v in box):
        return False
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def pick_2d_box(anno: dict, filter_settings: dict):
    tight = anno.get("bbox2D_tight")
    if filter_settings.get("modal_2D_boxes") and _valid_xyxy(tight):
        return _xyxy_to_xywh(tight)

    trunc = anno.get("bbox2D_trunc")
    if filter_settings.get("trunc_2D_boxes") and _valid_xyxy(trunc):
        return _xyxy_to_xywh(trunc)

    proj = anno.get("bbox2D_proj")
    if _valid_xyxy(proj):
        return _xyxy_to_xywh(proj)

    return anno.get("bbox")

def is_ignore(anno: dict, filter_settings: dict, image_height: int) -> bool:
    if anno.get("behind_camera") or not bool(anno.get("valid3D", True)):
        return True

    dims = anno["dimensions"]
    if dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0:
        return True
    if anno["center_cam"][2] > filter_settings["max_depth"]:
        return True
    if anno.get("lidar_pts", 1) == 0 or anno.get("segmentation_pts", 1) == 0:
        return True
    if anno.get("depth_error", 0) > 0.5:
        return True

    box2d = pick_2d_box(anno, filter_settings)
    if box2d is None:
        return True
    box_h = box2d[3]
    if box_h <= filter_settings["min_height_thres"] * image_height:
        return True
    if box_h >= filter_settings["max_height_thres"] * image_height:
        return True

    trunc = anno.get("truncation", -1)
    if trunc >= 0 and trunc >= filter_settings["truncation_thres"]:
        return True
    vis = anno.get("visibility", -1)
    if vis >= 0 and vis <= filter_settings["visibility_thres"]:
        return True
    if anno.get("category_name") in filter_settings.get("ignore_names", []):
        return True

    return False

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
                item["ignore"] = is_ignore(item, filter_settings, int(image["height"]))
                self._anns.append(item)
    def getAnnIds(self):
        return [a["id"] for a in self._anns]
    def loadAnns(self, ids):
        wanted = set(ids)
        return [a for a in self._anns if a["id"] in wanted]


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


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


def _resolve_json_path(path_candidate: str | Path, base_dir: str | Path | None = None) -> Path | None:
    """解析並確認 JSON 檔案的絕對或相對路徑。"""
    p = Path(path_candidate)
    if p.is_file():
        return p
    if base_dir:
        combined = Path(base_dir) / p
        if combined.is_file():
            return combined
    return None


def read_id_map(data: dict[str, Any]) -> dict[int, int]:
    """
    依據 Cube R-CNN 官方規則自動解析並構建類別映射表 (id_map) 與類別名稱字典 (names)。
    
    規則：
    1. 若 YAML 明確宣告 `auto_id_map: false` 且已提供完整的 `id_map` 字典，則直接尊重手動設定。
    2. 否則（預設行為，或 id_map 為 'auto' / None / 缺失時）：
       - 自動從 `stats.json` 或 `train_json` 讀取 category 定義。
       - 若設定了 `category_names` 或 `names`，則自動過濾出目標子集。
       - 嚴格依照原始 category_id 進行數值升冪排序 (sorted by raw ID)。
       - 連續指派 0 ~ N-1，並自動同步覆寫 data['names'], data['id_map'], data['nc']。
    """
    manual_id_map = data.get("id_map")
    auto_mode = data.get("auto_id_map", True)

    # 1. 使用者明確關閉自動推導，且提供了手動字典時退回靜態映射
    if isinstance(manual_id_map, dict) and not auto_mode and manual_id_map:
        return {int(k): int(v) for k, v in manual_id_map.items()}

    base_path = data.get("path", "")
    categories_found: list[dict] = []

    # 2. 優先嘗試從 stats.json 讀取類別清單 (Cube R-CNN 官方首選)
    stats_candidates = [
        data.get("stats_json"),
        data.get("priors_file"),
        Path(base_path) / "stats.json" if base_path else None,
        Path("datasets/Omni3D/stats.json"),
    ]
    for sc in stats_candidates:
        if sc is None:
            continue
        p = _resolve_json_path(sc, base_path)
        if p:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                if isinstance(sdata.get("categories"), list) and sdata["categories"]:
                    categories_found = sdata["categories"]
                    LOGGER.info(f"[Omni3D] 成功自 {p.name} 載入類別資訊 (共 {len(categories_found)} 類)。")
                    break
            except Exception as e:
                LOGGER.debug(f"[Omni3D] 讀取 {p} 失敗: {e}")

    # 3. 若 stats.json 不存在，直接自 train_json / val_json 讀取 COCO 格式標註
    if not categories_found:
        json_sources = (
            data.get("train_json")
            or data.get("train_jsons")
            or data.get("val_json")
            or data.get("train")
        )
        if isinstance(json_sources, (str, Path)):
            json_sources = [json_sources]
        elif isinstance(json_sources, (list, tuple)):
            json_sources = list(json_sources)
        else:
            json_sources = []

        for jf in json_sources:
            p = _resolve_json_path(jf, base_path)
            if p:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        jdata = json.load(f)
                    if isinstance(jdata.get("categories"), list) and jdata["categories"]:
                        categories_found = jdata["categories"]
                        LOGGER.info(f"[Omni3D] 成功自標註檔 {p.name} 擷取類別定義 (共 {len(categories_found)} 類)。")
                        break
                except Exception as e:
                    LOGGER.debug(f"[Omni3D] 讀取 {p} 失敗: {e}")

    # 若完全無法自資料集讀取，則回退至 YAML 內建的 id_map
    if not categories_found:
        if isinstance(manual_id_map, dict) and manual_id_map:
            LOGGER.warning("[Omni3D] 無法自動讀取類別標籤，退回使用 YAML 現存的靜態 id_map。")
            return {int(k): int(v) for k, v in manual_id_map.items()}
        raise FileNotFoundError(
            "無法自 JSON 或 stats.json 讀取 categories 定義，且 YAML 未提供有效的 id_map。"
        )

    # 4. 目標類別篩選 (若 filter_settings 或 YAML names 有指定子集)
    target_names: set[str] | None = None
    filter_settings = data.get("filter_settings") or {}
    if filter_settings.get("category_names"):
        target_names = {str(n).strip() for n in filter_settings["category_names"]}
    elif data.get("names"):
        raw_names = data["names"]
        if isinstance(raw_names, dict):
            target_names = {str(v).strip() for v in raw_names.values()}
        elif isinstance(raw_names, (list, tuple)):
            target_names = {str(v).strip() for v in raw_names}

    selected_cats: list[dict] = []
    seen_raw_ids = set()

    for cat in categories_found:
        raw_id = int(cat["id"])
        raw_name = str(cat["name"]).strip()
        if raw_id in seen_raw_ids:
            continue

        if target_names is None:
            selected_cats.append(cat)
            seen_raw_ids.add(raw_id)
        else:
            # 支援不分大小寫比對
            if raw_name in target_names or raw_name.lower() in {t.lower() for t in target_names}:
                selected_cats.append(cat)
                seen_raw_ids.add(raw_id)

    if not selected_cats:
        selected_cats = categories_found

    # 5. Cube R-CNN 核心演算法：依照 raw_id 數值嚴格由小至大排序
    selected_cats = sorted(selected_cats, key=lambda c: int(c["id"]))

    # 6. 自動派生連續索引 (0 ~ N-1)
    auto_id_map: dict[int, int] = {}
    auto_names: dict[int, str] = {}

    for idx, c in enumerate(selected_cats):
        raw_id = int(c["id"])
        c_name = str(c["name"]).strip()
        auto_id_map[raw_id] = idx
        auto_names[idx] = c_name

    # 7. 回寫並覆寫 data 字典，確保 Trainer、Validator 與模型 Head 完全同步
    data["id_map"] = auto_id_map
    data["names"] = auto_names
    data["nc"] = len(auto_names)

    LOGGER.info(
        f"[Omni3D] 已按 Cube R-CNN 規範自動構建映射表: 共 {len(auto_id_map)} 個類別 "
        f"(原始 ID 範圍: {min(auto_id_map.keys())} ~ {max(auto_id_map.keys())})"
    )
    return auto_id_map


# 提供別名，相容既有的 build_id_map 調用
build_id_map = read_id_map


def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()

def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces from a 3D cuboid representation without CUDA advanced indexing.
    Args:
        box3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
    Returns:
        verts: the 3D vertices of the cuboid in camera space [N, 8, 3]
        faces: the vertex indices per face [N, 12, 3]
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1]

    # 確保型態與設備一致
    box3d = to_float_tensor(box3d)
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)
    device = box3d.device

    centers = box3d[:, :3]
    w = box3d[:, 3]
    h = box3d[:, 4]
    l = box3d[:, 5]

    # 採用純張量堆疊，嚴格對齊原版 8 個頂點的幾何定義，徹底避開 CUDA Indexing.cu 崩潰
    # v0: [-l/2, -h/2, -w/2]
    # v1: [ l/2, -h/2, -w/2]
    # v2: [ l/2,  h/2, -w/2]
    # v3: [-l/2,  h/2, -w/2]
    # v4: [-l/2, -h/2,  w/2]
    # v5: [ l/2, -h/2,  w/2]
    # v6: [ l/2,  h/2,  w/2]
    # v7: [-l/2,  h/2,  w/2]
    x_local = torch.stack([-l, l, l, -l, -l, l, l, -l], dim=1) * 0.5
    y_local = torch.stack([-h, -h, h, h, -h, -h, h, h], dim=1) * 0.5
    z_local = torch.stack([-w, -w, -w, -w, w, w, w, w], dim=1) * 0.5

    # [n, 3, 8]
    verts = torch.stack([x_local, y_local, z_local], dim=1)

    if R is not None:
        verts = R @ verts

    # 平移至相機坐標系中心
    verts[:, 0:1, :] += centers[:, 0:1, None]
    verts[:, 1:2, :] += centers[:, 1:2, None]
    verts[:, 2:3, :] += centers[:, 2:3, None]

    verts = verts.transpose(1, 2)  # [n, 8, 3]

    faces = torch.tensor([
        [0, 1, 2], # front TR
        [2, 3, 0], # front BL

        [1, 5, 6], # right TR
        [6, 2, 1], # right BL

        [4, 0, 3], # left TR
        [3, 7, 4], # left BL

        [5, 4, 7], # back TR
        [7, 6, 5], # back BL

        [4, 5, 1], # top TR
        [1, 0, 4], # top BL

        [3, 2, 6], # bottom TR
        [6, 7, 3], # bottom BL
    ], dtype=torch.float32, device=device).unsqueeze(0).repeat([n, 1, 1])

    if squeeze:
        verts = verts.squeeze(0)
        faces = faces.squeeze(0)

    return verts, faces