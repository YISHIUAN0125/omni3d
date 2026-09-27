"""Omni3D utility functions aligned with Cube R-CNN official dataset protocols."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER


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