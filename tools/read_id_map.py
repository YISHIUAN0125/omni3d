import json
from pathlib import Path
from ultralytics.utils import YAML

# 讀取你的 YAML 配置
yaml_path = "configs/omni3d_38_classes.yaml"
data = YAML.load(yaml_path)
names = data["names"]  # 0: 'stationery', 1: 'sink', ...

# 讀取 SUN-RGBD 的 JSON 標註
json_file = "datasets/Omni3D/SUNRGBD_train.json"
with open(json_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

json_categories = raw_data.get("categories", [])
print(f"JSON 中總共定義了 {len(json_categories)} 個類別。\n")

# 建立 name -> model_class_idx 反向映射
name_to_model_idx = {v.strip().lower(): k for k, v in names.items()}

# 建立真正的 raw_category_id -> model_class_idx
real_id_map = {}
unmatched_in_json = []

for cat in json_categories:
    raw_id = cat["id"]
    cat_name = cat["name"].strip().lower()
    if cat_name in name_to_model_idx:
        model_idx = name_to_model_idx[cat_name]
        real_id_map[raw_id] = model_idx
    else:
        unmatched_in_json.append((raw_id, cat_name))

print("=" * 50)
print("請將以下正確的 id_map 貼入 configs/omni3d_38_classes.yaml：")
print("=" * 50)
print("id_map:")
for raw_id in sorted(real_id_map.keys()):
    matched_name = names[real_id_map[raw_id]]
    print(f"  {raw_id}: {real_id_map[raw_id]}  # {matched_name}")

print("\n未匹配到 38 類的 JSON 類別 (將自動被忽略):", len(unmatched_in_json))