from __future__ import annotations

import json
import math
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import build_dataloader
from ultralytics.data.omni3d_dataset import Omni3DDataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val_3d import Detection3DValidator
from ultralytics.nn.tasks import Detection3DModel
from ultralytics.utils import LOGGER, RANK, colorstr, YAML
import ultralytics.utils.plotting as ul_plot
from ultralytics.utils.checks import check_file
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first, unwrap_model

from .train_omni3d_util import build_id_map
from ultralytics.utils.cube_utils import compute_priors, Omni3DPriorDatasetAdapter, make_cfg


def plot_results_3d(file="path/to/results.csv", dir="", on_plot=None):
    """自適應動態網格繪圖，支援 30+ 個 3D 欄位，徹底解決 28-axis 越界錯誤。"""
    import pandas as pd
    import matplotlib.pyplot as plt

    save_dir = Path(file).parent if file else Path(dir)
    file = Path(file)
    if not file.exists():
        return

    try:
        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        plot_cols = [c for c in df.columns if c != "epoch"]
        n = len(plot_cols)
        if n == 0:
            return

        n_cols = 5
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.2), tight_layout=True)
        ax = axes.ravel() if hasattr(axes, "ravel") else [axes]

        x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))
        for i, col in enumerate(plot_cols):
            y = df[col]
            ax[i].plot(x, y, linewidth=1.5, marker="o", markersize=2)
            ax[i].set_title(col, fontsize=8)
            ax[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(i + 1, len(ax)):
            fig.delaxes(ax[j])

        save_path = save_dir / "results.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        if on_plot:
            on_plot(save_path)
    except Exception as e:
        LOGGER.warning(f"[Plotting] Failed to plot 3D results: {e}")


# 全局替換繪圖函式，攔截所有回呼 (Callbacks)
ul_plot.plot_results = plot_results_3d


SINGLE_BRANCH_LOSS_NAMES = (
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "loss_3d_xy",
    "loss_3d_dims",
    "loss_3d_z",
    "loss_3d_pose",
    "loss_3d_joint",
    "loss_3d_uncert",
    "loss_3d",
)


class Detection3DTrainer(DetectionTrainer):
    """Ultralytics trainer for flat TAL targets plus Cube R-CNN 3D annotations."""

    REQUIRED_BATCH_KEYS = {
        "img",
        "batch_idx",
        "cls",
        "bboxes",
        "gt_boxes3D",
        "gt_poses",
        "gt_2D",
        "K",
        "K_orig",
        "im_scales",
        "im_scales_orig",
    }

    DEVICE_TENSOR_KEYS = (
        "batch_idx",
        "cls",
        "bboxes",
        "dimensions",
        "center_cam",
        "R_cam",
        "gt_boxes3D",
        "gt_poses",
        "gt_2D",
        "K",
        "K_orig",
        "im_scales",
        "im_scales_orig",
        "ignore",
        "ignored_bboxes",
        "ignored_batch_idx",
    )

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        overrides = dict(overrides or {})

        # 關鍵修復：在交給父類別前先 pop 掉自定義參數，避開 check_dict_alignment 白名單檢查
        self.custom_val_period = overrides.pop("val_period", None)

        if float(overrides.get("multi_scale", 0.0) or 0.0) > 0:
            raise ValueError(
                "Native multi_scale does not synchronize camera matrices (K). Set multi_scale=0.0."
            )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.loss_names = SINGLE_BRANCH_LOSS_NAMES

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = Detection3DModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    # ------------------------------------------------------------------
    # Data YAML loading
    # ------------------------------------------------------------------
    def get_dataset(self):
        data_path = check_file(self.args.data)
        data = YAML.load(data_path, append_filename=True)

        required = ("nc", "names")
        missing = [key for key in required if key not in data]
        if missing:
            raise SyntaxError(f"{data_path} is missing required key(s): {missing}.")

        self.data = data
        dataset_path = data.get("path", "")
        self._json_files("train", dataset_path)
        self._json_files("val", dataset_path)
        self._id_map()

        placeholder = self._image_root(dataset_path) if self.data.get("image_root") else dataset_path
        data.setdefault("train", placeholder)
        data.setdefault("val", placeholder)

        return data

    @staticmethod
    def _as_list(value: Any) -> list:
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @staticmethod
    def _looks_like_json(value: Any) -> bool:
        values = value if isinstance(value, (list, tuple)) else [value]
        return bool(values) and all(str(item).lower().endswith(".json") for item in values)

    def _json_files(self, mode: str, dataset_path: str | list[str]) -> list[str]:
        candidates = (
            self.data.get(f"{mode}_json"),
            self.data.get(f"{mode}_jsons"),
            dataset_path if self._looks_like_json(dataset_path) else None,
        )
        for value in candidates:
            files = [str(item) for item in self._as_list(value) if item]
            if files:
                return files
        raise KeyError(f"No Omni3D JSON configured for mode={mode!r}.")

    def _image_root(self, dataset_path: str | list[str]) -> str:
        for key in ("image_root", "images", "img_path"):
            if self.data.get(key):
                return str(self.data[key])
        if not self._looks_like_json(dataset_path):
            return str(dataset_path)
        raise KeyError("Omni3D image root is missing. Add `image_root` to data YAML.")

    def _id_map(self) -> dict[int, int]:
        return build_id_map(self.data)

    def set_model_attributes(self):
        super().set_model_attributes()
        m = unwrap_model(self.model)
        head = getattr(m, "model", [None])[-1]

        if not hasattr(head, "set_priors"):
            return

        if getattr(head, "priors_initialized", torch.tensor(False)).item():
            LOGGER.info("[Detect3D] 先驗已存在於載入的權重中，跳過注入。")
            return

        priors_source = (
            self.data.get("priors")
            or self.data.get("priors_file")
            or self.data.get("stats_json")
        )
        priors_data = None

        if isinstance(priors_source, dict):
            priors_data = priors_source
        elif isinstance(priors_source, str) and Path(priors_source).is_file():
            p_path = Path(priors_source)
            try:
                if p_path.suffix == ".json":
                    with open(p_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if "priors_dims_per_cat" in raw:
                        priors_data = raw
                    else:
                        LOGGER.info(f"[Detect3D] 檢測到 stats.json ({p_path.name})，正在計算先驗...")
                        train_jsons = self._json_files("train", self.data.get("path", ""))
                        filter_settings = dict(self.data.get("filter_settings") or {})
                        adapter = Omni3DPriorDatasetAdapter(train_jsons, self._id_map(), filter_settings)
                        prior_cfg = make_cfg(
                            virtual_depth=head.virtual_depth,
                            virtual_focal=head.virtual_focal,
                            test_scale_min=self.args.imgsz,
                            test_scale_max=self.args.imgsz,
                            cluster_bins=head.cluster_bins,
                            anchor_sizes=[[16, 32, 64], [64, 128, 256], [256, 512, 1024]],
                            modal_2d=filter_settings.get("modal_2D_boxes", False),
                            trunc_2d=filter_settings.get("trunc_2D_boxes", False),
                        )
                        cat_names = [self.data["names"][i] for i in range(self.data["nc"])]
                        priors_data = compute_priors(prior_cfg, adapter, cat_names)
            except Exception as e:
                LOGGER.warning(f"[Detect3D] 讀取先驗失敗: {e}，跳過 set_priors。")
                priors_data = None

        if priors_data is not None:
            head.set_priors(priors_data)
            LOGGER.info("[Detect3D] 成功載入並設定先驗。")
        else:
            LOGGER.info("[Detect3D] 未提供有效先驗檔案，平滑關閉先驗引導。")
            head.disable_priors()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        json_files = self._json_files(mode, img_path)
        image_root = self._image_root(img_path)

        if mode == "val":
            filter_settings = dict(
                self.data.get("val_filter_settings")
                or self.data.get("filter_settings")
                or {}
            )
        else:
            filter_settings = dict(self.data.get("filter_settings") or {})

        flip_probability = float(getattr(self.args, "fliplr", 0.0)) if mode == "train" else 0.0

        dataset = Omni3DDataset(
            img_path=image_root,
            json_files=json_files,
            id_map=self._id_map(),
            filter_settings=filter_settings,
            cache_dir=self.data.get("cache_dir"),
            filter_empty=bool(self.data.get("filter_empty", True)),
            imgsz=self.args.imgsz,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            batch_size=batch,
            stride=stride,
            pad=0.0,
            cache=self.args.cache,
            prefix=colorstr(f"{mode}: "),
            classes=getattr(self.args, "classes", None),
            fraction=getattr(self.args, "fraction", 1.0) if mode == "train" else 1.0,
            fliplr_p=flip_probability,
            mirror_center_x=True,
        )
        return dataset

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)

        shuffle = mode == "train"
        workers = int(self.args.workers) if mode == "train" else int(self.args.workers) * 2
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=workers,
            shuffle=shuffle,
            rank=rank,
            drop_last=bool(getattr(self.args, "compile", False)) and mode == "train",
            pin_memory=getattr(self.args, "pin_memory", True),
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        non_blocking = self.device.type not in {"cpu", "mps"}
        batch["img"] = batch["img"].to(self.device, non_blocking=non_blocking).float().div_(255.0)

        for key in self.DEVICE_TENSOR_KEYS:
            value = batch.get(key)
            if torch.is_tensor(value):
                batch[key] = value.to(self.device, non_blocking=non_blocking)

        return batch

    @staticmethod
    def _metric_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().mean().cpu().item())
        return float(value)

    def label_loss_items(self, loss_items=None, prefix="train"):
        names = tuple(self.loss_names or SINGLE_BRANCH_LOSS_NAMES)
        if loss_items is None:
            return [f"{prefix}/{name}" for name in names]

        if isinstance(loss_items, dict):
            return {
                f"{prefix}/{key}": round(self._metric_float(loss_items.get(key, 0.0)), 5)
                for key in names
            }

        return {
            f"{prefix}/{key}": round(self._metric_float(value), 5)
            for key, value in zip(names, loss_items)
        }

    def progress_string(self):
        names = tuple(self.loss_names or SINGLE_BRANCH_LOSS_NAMES)
        return ("\n" + "%11s" * (4 + len(names))) % (
            "Epoch",
            "GPU_mem",
            *names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        if "im_file" not in batch:
            return
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        labels = self.train_loader.dataset.labels
        boxes = [item["bboxes"] for item in labels if len(item["bboxes"])]
        classes = [item["cls"] for item in labels if len(item["cls"])]
        if not boxes:
            return
        plot_labels(
            np.concatenate(boxes, axis=0),
            np.concatenate(classes, axis=0).squeeze(),
            names=self.data["names"],
            save_dir=self.save_dir,
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """覆寫 Trainer 的 plot_metrics，調用自適應 3D 繪圖。"""
        plot_results_3d(file=self.csv, on_plot=self.on_plot)

    # ------------------------------------------------------------------
    # Validation & Final Evaluation
    # ------------------------------------------------------------------
    def get_validator(self):
        self.loss_names = tuple(self.loss_names or SINGLE_BRANCH_LOSS_NAMES)
        return Detection3DValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        """支援自定義 val_period 驗證週期，大幅節省 3D 評估時間。"""
        if not self.args.val:
            loss = getattr(self, "loss", None)
            fallback_fitness = -self._metric_float(loss) if loss is not None else 0.0
            return {}, fallback_fitness

        # 優先從 self.custom_val_period 或 data.yaml 中讀取 val_period (預設為 1，即每個 epoch 都驗證)
        val_period = int(
            getattr(self, "custom_val_period", None)
            or self.data.get("val_period", 1)
            or 1
        )

        current_epoch = self.epoch + 1
        is_final_epoch = current_epoch >= self.epochs

        # 若設定了週期，且當前不是週期倍數、也不是最後一個 epoch，則跳過本次驗證
        if val_period > 1 and (current_epoch % val_period != 0) and not is_final_epoch:
            LOGGER.info(f"Epoch {current_epoch}/{self.epochs}: 跳過 3D 評估 (val_period={val_period})")
            last_fitness = getattr(self, "fitness", 0.0)
            last_metrics = getattr(self, "metrics", {})
            return last_metrics, last_fitness

        return super().validate()

    def final_eval(self):
        if not self.args.val:
            LOGGER.info("Skipping final evaluation because val=False")
            return None

        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.validator.args.save_json = False

                    try:
                        ckpt = torch.load(f, map_location=self.device, weights_only=False)
                    except TypeError:
                        ckpt = torch.load(f, map_location=self.device)

                    best_model = ckpt.get("ema") or ckpt.get("model")
                    best_model = best_model.float().to(self.device)
                    self.validator.init_metrics(best_model)

                    self.metrics = self.validator(trainer=self)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

        return self.metrics

    def auto_batch(self):
        raise RuntimeError("AutoBatch is disabled for 3D model baseline. Set batch explicitly.")