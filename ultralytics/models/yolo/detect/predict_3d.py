from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import LOGGER, ops, colorstr
from ultralytics.utils.cube_utils import get_cuboid_verts_faces, CUBOID_EDGES, project_points

class Detection3DResult:
    """輕量結果容器，只滿足 BasePredictor.stream_inference 需要的介面。"""
    def __init__(self, pred, decoded, verts, meta, orig_img, path):
        self.pred = pred
        self.decoded = decoded
        self.verts = verts
        self.meta = meta
        self.orig_img = orig_img
        self.path = path
        self.speed = {}          # stream_inference 會寫入這個
        self.save_dir = None     # write_results 裡有用到 result.save_dir


class Detection3DPredictor(BasePredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self._K_override = overrides.pop("K", None)
        if self._K_override is not None:
            self._K_override = np.asarray(self._K_override, dtype=np.float32).reshape(3, 3)
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.head = None
        self._meta_batch: list[dict] = []

    def _get_K_for_image(self, h: int, w: int) -> np.ndarray:
        if self._K_override is not None:
            return self._K_override.copy()
        focal_length = 4.0 * h / 2
        px, py = w / 2, h / 2
        return np.array([[focal_length, 0.0, px], [0.0, focal_length, py], [0.0, 0.0, 1.0]], dtype=np.float32)

    def setup_model(self, model, verbose = True):
        super().setup_model(model, verbose)
        native = self.model.model if hasattr(self.model, "model") else self.model
        candidates = [m for m in native.modules() if hasattr(m, "decode_cube") and hasattr(m, "cube_branch")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expect one CubeHead find {len(candidates)}")
        self.head = candidates[0]

    def preprocess(self, im0s: list[np.ndarray]) -> torch.Tensor:
        self._meta_batch = []
        imgsz = self.imgsz[0] if isinstance(self.imgsz, (list, tuple)) else self.imgsz

        batch_imgs = []
        for im0 in im0s:
            H0, W0 = im0.shape[:2]
            K_orig = self._get_K_for_image(H0, W0)

            r_load = imgsz / max(H0, W0)
            w0, h0 = min(round(W0 * r_load), imgsz), min(round(H0 * r_load), imgsz)
            im_resized = cv2.resize(im0, (w0, h0), interpolation=cv2.INTER_LINEAR)

            sx, sy = w0 / W0, h0 / H0
            dw, dh = (imgsz - w0) / 2, (imgsz - h0) / 2
            left, top = round(dw - 0.1), round(dh - 0.1)
            right, bottom = imgsz - w0 - left, imgsz - h0 - top
            im_padded = cv2.copyMakeBorder(
                im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )

            K_net = K_orig.copy().astype(np.float32)
            K_net[0, 0] *= sx
            K_net[1, 1] *= sy
            K_net[0, 2] = K_net[0, 2] * sx + left
            K_net[1, 2] = K_net[1, 2] * sy + top

            self._meta_batch.append({
                "K_orig": K_orig, "K_net": K_net,
                "left": left, "top": top,
                "im_scales": np.array([h0 / H0, w0 / W0], dtype=np.float32),
                "im_scales_orig": np.array([H0, W0], dtype=np.float32),
            })
            batch_imgs.append(im_padded)

        im = torch.from_numpy(np.stack(batch_imgs)).to(self.device)
        im = im.permute(0, 3, 1, 2).flip(1).contiguous()  # BHWC BGR -> BCHW RGB
        im = (im.half() if self.model.fp16 else im.float()) / 255.0
        return im


    def postprocess(self, preds, img, orig_imgs):
        detections, cube_preds = preds[0], preds[1]
        if detections.ndim == 3 and detections.shape[1] < detections.shape[2]:
            detections, cube_preds = self.head.postprocess(detections.permute(0, 2, 1), cube_preds)

        results = []
        for i in range(detections.shape[0]):
            det = detections[i]
            valid = torch.isfinite(det[:, :6]).all(1) & (det[:, 4] >= self.args.conf)
            det = det[valid]
            selected_cube = {name: value[i][valid] for name, value in cube_preds.items()}
            boxes = ops.xywh2xyxy(det[:, :4])

            pred = {"bboxes": boxes, "conf": det[:, 4], "cls": det[:, 5].long(), "cube_preds": selected_cube}
            meta = self._meta_batch[i]

            decoded, verts = None, None
            n = len(pred["conf"])
            if n > 0:
                device = det.device
                K = torch.as_tensor(meta["K_net"], device=device).unsqueeze(0).expand(n, -1, -1)
                K_orig_t = torch.as_tensor(meta["K_orig"], device=device).unsqueeze(0).expand(n, -1, -1)
                im_scales = torch.as_tensor(meta["im_scales"], device=device).unsqueeze(0).expand(n, -1)
                im_scales_orig = torch.as_tensor(meta["im_scales_orig"], device=device).unsqueeze(0).expand(n, -1)

                decoded = self.head.decode_cube(
                    cube_preds=pred["cube_preds"],
                    box_classes=pred["cls"],
                    src_boxes=pred["bboxes"],
                    Ks_scaled_per_box=K,
                    focal_lengths=K_orig_t[:, 1, 1],
                    im_scales_orig=im_scales_orig,
                    im_scales=im_scales,
                )
                boxes3d = torch.cat((decoded["center_cam"], decoded["dims"]), dim=1)
                verts, _ = get_cuboid_verts_faces(boxes3d, decoded["pose"])

            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0][i] if self.batch else ""
            results.append(Detection3DResult(pred, decoded, verts, meta, orig_img, path))
        return results


    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        r = self.results[i]
        pred, decoded, verts, meta = r.pred, r.decoded, r.verts, r.meta

        im0 = self.batch[1][i].copy()
        n_det = len(pred["conf"])
        string = f"{im.shape[2]}x{im.shape[3]} {n_det} objects, "

        left, top = meta["left"], meta["top"]
        sx, sy = meta["im_scales"][1], meta["im_scales"][0]
        boxes = pred["bboxes"].detach().cpu().numpy()
        confs = pred["conf"].detach().cpu().numpy()
        clss = pred["cls"].detach().cpu().numpy()
        K_net_t = torch.as_tensor(meta["K_net"], device=pred["bboxes"].device)

        for j in range(n_det):
            x1, y1, x2, y2 = boxes[j]
            x1, x2 = (np.array([x1, x2]) - left) / sx
            y1, y2 = (np.array([y1, y2]) - top) / sy
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            # label = f"{self.model.names[int(clss[j])]} {confs[j]:.2f}"
            label = f"{self.model.names[int(clss[j])]} {confs[j]:.2f}"
            if decoded is not None:
                z_val = decoded["center_cam"][j, 2].item()
                label += f" z={z_val:.2f}m"
                if "conf" in decoded:
                    label += f" c3d={decoded['conf'][j].item():.2f}"
            cv2.putText(im0, label, (int(x1), max(int(y1) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.putText(im0, label, (int(x1), max(int(y1) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if verts is not None:
                uv = project_points(K_net_t, verts[j]).detach().cpu().numpy()
                uv[:, 0] = (uv[:, 0] - left) / sx
                uv[:, 1] = (uv[:, 1] - top) / sy
                for a, b in CUBOID_EDGES:
                    cv2.line(im0, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (255, 128, 0), 1)

        self.plotted_img = im0

        if self.args.save:
            save_path = str(self.save_dir / p.name)
            cv2.imwrite(save_path, im0)
            string += f"saved to {colorstr('bold', save_path)}"

        if self.args.show:
            self.show(str(p))

        return string