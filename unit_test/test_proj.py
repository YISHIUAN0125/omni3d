import json, numpy as np
from ultralytics.utils.cube_utils import cuboid_corners, project_points, box_iou_xyxy
from PIL import Image

data = json.load(open("datasets/Omni3D/Hypersim_train.json"))
imgs = {im["id"]: im for im in data["images"]}
ious, sizes_bad = [], 0
for a in data["annotations"][:3000]:
    p = a.get("bbox2D_proj")
    if not p or p[0] == -1 or a.get("behind_camera"):
        continue
    K = imgs[a["image_id"]]["K"]
    cor = cuboid_corners([a["center_cam"]], [a["dimensions"]], [a["R_cam"]])[0]
    if (cor[:, 2] <= 0.1).any():
        continue
    uv = project_points(K, cor).numpy()
    box = [uv[:,0].min(), uv[:,1].min(), uv[:,0].max(), uv[:,1].max()]
    ious.append(box_iou_xyxy(box, p))
print(len(ious), np.percentile(ious, [5, 25, 50, 75, 95]))

# 診斷：JSON 記載的尺寸 vs 實際檔案尺寸
bad = 0
for im in list(imgs.values())[:200]:
    w, h = Image.open(f"datasets/{im['file_path']}").size
    if (w, h) != (im["width"], im["height"]):
        bad += 1
print("尺寸不一致的影像:", bad, "/ 200")