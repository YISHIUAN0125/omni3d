import torch
from ultralytics.data import (
    build_category_id_map, 
    build_dataloader, 
    Omni3DDataset)

id_map = build_category_id_map(
    "datasets/Omni3D/stats.json",
    category_names=['stationery', 'sink', 'table', 'floor mat', 'bottle', 'bookcase', 'bin', 'blinds', 'pillow', 'bicycle', 'refrigerator', 'night stand', 'chair', 'sofa', 'books', 'oven', 'towel', 'cabinet', 'window', 'curtain', 'bathtub', 'laptop', 'desk', 'television', 'clothes', 'stove', 'cup', 'shelves', 'box', 'shoes', 'mirror', 'door', 'picture', 'lamp', 'machine', 'counter', 'bed', 'toilet'],
)

dataset = Omni3DDataset(
    json_files=["datasets/Omni3D/SUNRGBD_train.json"],
    img_path="datasets",
    id_map=id_map,
    imgsz=640,
    augment=True,
    fliplr_p=0.5,
    batch_size=16,
)
loader = build_dataloader(dataset, batch=16, workers=0, shuffle=True)
batch = next(iter(loader))
print({k: (v.shape if torch.is_tensor(v) else type(v)) for k, v in batch.items()})