from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data import Omni3DDataset, build_category_id_map
from ultralytics.data.augment import Compose, Format, LetterBox
import ultralytics.data.omni3d_dataset as ud


# ------------------------------------------------------------------
# 1. LetterBox3D:同步 K 的縮放 + padding offset
#    (跟聊天紀錄裡討論的版本一致,這裡直接內嵌,方便單檔案獨立驗證)
# ------------------------------------------------------------------
class LetterBox3D(LetterBox):
    """LetterBox 只會 resize + pad 影像,不會動 K;這裡在同一次呼叫裡
    把 letterbox 的縮放/位移同步套到 K 上。僅適用於 auto=False,
    scale_fill=False 的預設呼叫方式(目前 build_transforms 的用法)。
    """

    def __call__(self, labels: dict) -> dict:
        h0, w0 = labels["img"].shape[:2]

        labels = super().__call__(labels)

        new_shape = self.new_shape
        new_h, new_w = new_shape if isinstance(new_shape, tuple) else (new_shape, new_shape)
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

        return labels


def _patched_build_transforms(self, hyp=None):
    """monkeypatch Omni3DDataset.build_transforms,把 LetterBox 換成 LetterBox3D。
    如果你已經手動改過原始檔,這段等於重複套用一次同樣的 transforms,結果不變。
    """
    transforms = Compose(
        [
            ud.RandomFlip3D(
                p=self.fliplr_p if self.augment else 0.0,
                mirror_center_x=self.mirror_center_x,
            ),
            LetterBox3D(new_shape=(self.imgsz, self.imgsz), scaleup=self.augment),
        ]
    )
    transforms.append(
        Format(bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False)
    )
    return transforms

Omni3DDataset.build_transforms = _patched_build_transforms


# ------------------------------------------------------------------
# 2. 3D box 投影工具
# ------------------------------------------------------------------
_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 上面四邊
    (4, 5), (5, 6), (6, 7), (7, 4),  # 下面四邊
    (0, 4), (1, 5), (2, 6), (3, 7),  # 直的四邊
]


def get_3d_box_corners(dim: np.ndarray, center: np.ndarray, R: np.ndarray) -> np.ndarray:
    """dim=(w,h,l), center=(x,y,z) in camera coords, R: 3x3 camera-space rotation.
    回傳 8x3 的 corner 座標(camera coords)。"""
    w, h, l = dim
    x = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    y = np.array([h, h, h, h, -h, -h, -h, -h]) / 2
    z = np.array([l, -l, -l, l, l, -l, -l, l]) / 2
    corners = np.stack([x, y, z], axis=0)  # 3x8
    corners = R @ corners + center.reshape(3, 1)
    return corners.T  # 8x3


def project_points(pts3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    proj = (K @ pts3d.T).T
    proj[:, :2] /= np.clip(proj[:, 2:3], 1e-6, None)
    return proj[:, :2]


def draw_3d_boxes(img, dims, centers, Rs, K, ignore, color=(255, 255, 0)):
    for dim, center, R, ig in zip(dims, centers, Rs, ignore):
        if ig:
            continue
        corners = get_3d_box_corners(np.asarray(dim), np.asarray(center), np.asarray(R))
        # 相機後方的點投影沒有意義,整個 box 跳過
        if (corners[:, 2] <= 0).any():
            continue
        pts2d = project_points(corners, K)
        for i, j in _EDGES:
            p1 = tuple(np.round(pts2d[i]).astype(int))
            p2 = tuple(np.round(pts2d[j]).astype(int))
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    return img


def draw_2d_boxes(img, bboxes_norm, w, h, ignore, color=(0, 255, 0)):
    for b, ig in zip(bboxes_norm, ignore):
        if ig:
            continue
        cx, cy, bw, bh = b
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    return img

# ------------------------------------------------------------------
# 3. 把單一 sample(dataset[idx] 的輸出)畫成一張 BGR uint8 圖
# ------------------------------------------------------------------
def render_sample(sample: dict) -> np.ndarray:
    img = sample["img"]
    if torch.is_tensor(img):
        img = img.numpy()
    if img.shape[0] in (1, 3):  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 註:Format 內部可能隨機把 RGB/BGR 對調(self.bgr 機率),
    # 所以顏色色調不一定精準對應原圖,但不影響 box 對齊位置的檢查。

    h, w = img.shape[:2]
    K = np.asarray(sample["K"])
    dims = np.asarray(sample["dimensions"])
    centers = np.asarray(sample["center_cam"])
    Rs = np.asarray(sample["R_cam"]).reshape(-1, 3, 3)
    ignore = np.asarray(sample["ignore"])
    bboxes = sample["bboxes"] if "bboxes" in sample else sample["instances"].bboxes
    if torch.is_tensor(bboxes):
        bboxes = bboxes.numpy()

    draw_2d_boxes(img, bboxes, w, h, ignore)
    draw_3d_boxes(img, dims, centers, Rs, K, ignore)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="dataset 裡要檢查的樣本 index")
    ap.add_argument("--out", type=str, default="unit_test/flip_compare.png")
    ap.add_argument("--stats_json", type=str, default="datasets/Omni3D/stats.json")
    ap.add_argument("--train_json", type=str, default="datasets/Omni3D/SUNRGBD_train.json")
    ap.add_argument("--img_root", type=str, default="datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    category_names = [
        "stationery", "sink", "table", "floor mat", "bottle", "bookcase", "bin",
        "blinds", "pillow", "bicycle", "refrigerator", "night stand", "chair",
        "sofa", "books", "oven", "towel", "cabinet", "window", "curtain",
        "bathtub", "laptop", "desk", "television", "clothes", "stove", "cup",
        "shelves", "box", "shoes", "mirror", "door", "picture", "lamp",
        "machine", "counter", "bed", "toilet",
    ]
    id_map = build_category_id_map(args.stats_json, category_names=category_names)

    common_kwargs = dict(
        json_files=[args.train_json],
        img_path=args.img_root,
        id_map=id_map,
        imgsz=args.imgsz,
        augment=True,      # 一定要 True,否則 fliplr_p 不會生效(見 build_transforms)
        mirror_center_x=True,
        batch_size=16,
    )

    ds_noflip = Omni3DDataset(**common_kwargs, fliplr_p=0.0)
    ds_flip = Omni3DDataset(**common_kwargs, fliplr_p=1.0)  # p=1.0 -> 強制一定翻轉

    sample_before = ds_noflip[args.idx]
    sample_after = ds_flip[args.idx]

    img_before = render_sample(sample_before)
    img_after = render_sample(sample_after)

    cv2.putText(img_before, "no flip", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img_after, "flipped", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    combined = np.concatenate([img_before, img_after], axis=1)
    out_path = Path(args.out)
    cv2.imwrite(str(out_path), combined)
    print(f"saved -> {out_path.resolve()}")
    print(f"idx={args.idx}  n_instances={len(sample_before['ignore'])}")


if __name__ == "__main__":
    main()