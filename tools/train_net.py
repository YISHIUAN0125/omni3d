# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import gc
import inspect
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

import detectron2.engine.defaults as d2_defaults
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.engine import (
    DefaultTrainer,
    HookBase,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn import data, util, vis
from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    DatasetMapper3D,
    build_detection_test_loader,
    build_detection_train_loader,
    simple_register,
)
from cubercnn.evaluation import Omni3DEvaluationHelper, inference_on_dataset
from cubercnn.modeling.meta_arch import build_model, build_yolo_wrapper
from cubercnn.solver import (
    PeriodicCheckpointerOnlyOne,
    build_optimizer,
    build_optimizer_yolo3d,
    freeze_bn,
)

MAX_TRAINING_ATTEMPTS = 10


class ModelDivergedError(RuntimeError):
    """Model diverged error"""


_EPOCH_SIZE_CACHE = {}

def compute_epoch_size(cfg):
    """Calculate the number of samples per epoch according to the sampler type."""

    names = list(cfg.DATASETS.TRAIN)
    if not names:
        return 0.0

    sampler = getattr(cfg.DATALOADER, "SAMPLER_TRAIN", "TrainingSampler")
    repeat_thr = getattr(cfg.DATALOADER, "REPEAT_THRESHOLD", 0.0)
    subset_ratio = getattr(cfg.DATALOADER, "RANDOM_SUBSET_RATIO", 1.0)
    key = (tuple(names), sampler, repeat_thr, subset_ratio)
    if key in _EPOCH_SIZE_CACHE:
        return _EPOCH_SIZE_CACHE[key]

    dataset_dicts = []
    for name in names:
        dataset_dicts.extend(DatasetCatalog.get(name))
    n = float(len(dataset_dicts))

    if sampler == "TrainingSampler":
        size = n
    elif sampler == "RepeatFactorTrainingSampler":
        try:
            rf = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dicts, repeat_thr
            )
            size = float(rf.sum().item())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Unable to compute repeat factor（{e!r}, fallback to the size of datasets {n:.0f}.")
            size = n

    logger.info(f"[epoch size] sampler={sampler}, datasets size={n:.0f}, avalible epoch samples≈{size:.1f}")
    _EPOCH_SIZE_CACHE[key] = size
    return size

def compute_iters_per_epoch(cfg) -> int:
    size = compute_epoch_size(cfg)
    return max(math.ceil(size / cfg.SOLVER.IMS_PER_BATCH), 1)

def _rescale_lr_schedule(cfg, old_max, new_max):
    steps = tuple(cfg.SOLVER.STEPS)
    if steps:
        ratio = new_max / old_max
        scaled = []
        for s in steps:
            v = int(round(s * ratio))
            if 0 < v < new_max and (not scaled or v > scaled[-1]):
                scaled.append(v)
            else:
                logger.warning(f"LR STEP {s} scaling to {v} needs to be incremented and < MAX_ITER={new_max}), and has been discarded.")
        cfg.SOLVER.STEPS = tuple(scaled)
        logger.info(f"[LR rescale] MAX_ITER {old_max}->{new_max}, STEPS {steps} -> {tuple(scaled)}")

    warmup = cfg.SOLVER.WARMUP_ITERS
    first = cfg.SOLVER.STEPS[0] if len(cfg.SOLVER.STEPS) > 0 else new_max
    if warmup >= first:
        logger.warning(f"WARMUP_ITERS={warmup} >= firsr decay point/MAX_ITER={first}, please check warmup setting.")

def resolve_epoch_iter_config(cfg, mode: str = "iter") -> int:
    iters_per_epoch = compute_iters_per_epoch(cfg)
    old_max = cfg.SOLVER.MAX_ITER

    if mode == "epoch":
        if cfg.MODEL.YOLO3D.EPOCHS <= 0:
            raise ValueError("mode='epoch' need to set cfg.MODEL.YOLO3D.EPOCHS > 0 first.")
        new_max = max(int(round(cfg.MODEL.YOLO3D.EPOCHS * iters_per_epoch)), 1)
        if getattr(cfg.MODEL.YOLO3D, "RESCALE_LR_STEPS", True) and old_max > 0 and new_max != old_max:
            _rescale_lr_schedule(cfg, old_max, new_max)
        cfg.SOLVER.MAX_ITER = new_max
    elif mode == "iter":
        if cfg.SOLVER.MAX_ITER <= 0:
            raise ValueError("mode='iter' need to set cfg.SOLVER.MAX_ITER > 0 first.")
        cfg.MODEL.YOLO3D.EPOCHS = cfg.SOLVER.MAX_ITER / iters_per_epoch
    else:
        raise ValueError(f"cfg.MODEL.YOLO3D.TIME_UNIT need to be 'epoch' or 'iter', got {mode!r}")

    logger.info(
        f"[epoch/iter conversion] iters_per_epoch={iters_per_epoch}，"
        f"MAX_ITER={cfg.SOLVER.MAX_ITER}, YOLO epochs={cfg.MODEL.YOLO3D.EPOCHS:.2f}"
    )
    return iters_per_epoch


# -----------------------------------------------------------
# Criterion update hook
# -----------------------------------------------------------
def _get_criterion(model):
    """
    Unpack DDP and get criterion：
    model.criterion / model.module.criterion / model.(module.)model.criterion
    """
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "criterion"):
        return raw_model.criterion
    if hasattr(raw_model, "model") and hasattr(raw_model.model, "criterion"):
        return raw_model.model.criterion
    return None

class CriterionScheduler:
    def __init__(self, criterion, iters_per_epoch: int, epoch_type: str = "auto"):
        self.criterion = criterion
        self.iters_per_epoch = iters_per_epoch

        params = inspect.signature(criterion.update).parameters
        self.names = {n for n in ("step", "max_iter", "epoch", "max_epoch") if n in params}

        if epoch_type == "auto":
            ann = params["epoch"].annotation if "epoch" in params else inspect.Parameter.empty
            self.epoch_as_int = ann in (int, "int")
        else:
            self.epoch_as_int = epoch_type == "int"

        self.n_positional = 0
        if not self.names:
            self.n_positional = len([
                p for p in params.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            ])

    def __call__(self, step: int, max_iter: int, floor_epoch: bool = False):
        ipe = self.iters_per_epoch
        epoch = step // ipe if floor_epoch else step / ipe
        if self.epoch_as_int:
            epoch = int(epoch)
        max_epoch = max_iter / ipe

        if self.names:
            values = {"step": step, "max_iter": max_iter, "epoch": epoch, "max_epoch": max_epoch}
            self.criterion.update(**{k: values[k] for k in self.names})
        elif self.n_positional >= 2:
            self.criterion.update(epoch, max_epoch)
        elif self.n_positional == 1:
            self.criterion.update(epoch)
        else:
            self.criterion.update()

class CriterionUpdateHook(HookBase):
    """Dynamic update criterion: TAL assigner o2m o2o weights etc."""
    def __init__(self, iters_per_epoch: int, mode: str = "step", epoch_type: str = "auto"):
        assert mode in ("step", "epoch"), f"CRITERION_UPDATE_MODE need to be 'step' or 'epoch', got {mode!r}"
        self.iters_per_epoch = iters_per_epoch
        self.mode = mode
        self.epoch_type = epoch_type
        self._updater = None

    def _sync(self, step: int):
        self._updater(step, self.trainer.max_iter, floor_epoch=(self.mode == "epoch"))

    def before_train(self):
        criterion = _get_criterion(self.trainer.train_model)
        if criterion is None or not hasattr(criterion, "update"):
            if comm.is_main_process():
                logger.warning("[CriterionUpdateHook] Can't find criterion.update(), ignore this when train cubercnn or any model based on detectron2")
            return

        self._updater = CriterionScheduler(criterion, self.iters_per_epoch, self.epoch_type)
        cur_step = self.trainer.iter  # 已完成步數（fresh=0，resume=start_iter）
        self._sync(cur_step)
        if comm.is_main_process():
            logger.info(
                f"[Criterion Sync] step {cur_step} (epoch {cur_step / self.iters_per_epoch:.2f}),"
                f"mode={self.mode}, update param={sorted(self._updater.names) or f'positional×{self._updater.n_positional}'}"
            )

    def after_step(self):
        if self._updater is None:
            return
        cur_step = self.trainer.iter + 1
        is_epoch_boundary = (cur_step % self.iters_per_epoch == 0)

        if self.mode == "step" or is_epoch_boundary:
            self._sync(cur_step)
        if is_epoch_boundary and comm.is_main_process():
            logger.info(f"[Epoch Update] Epoch {cur_step // self.iters_per_epoch} finished (step {cur_step})")

# -----------------------------------------------------------
# Build model
# -----------------------------------------------------------
def build_model_for_cfg(cfg, priors=None):
    if cfg.MODEL.META_ARCHITECTURE == "YOLO3DWrapper":
        model = build_yolo_wrapper(cfg)
    else:
        model = build_model(cfg, priors=priors)
    if not cfg.MODEL.USE_BN:
        freeze_bn(model)
    return model

def _gather_loss_keys(loss_dict):
    """Take the union of the loss keys for all ranks 
    (this only needs to be called once in the first step, 
    but all ranks need to be called simultaneously)."""
    gathered = comm.all_gather(sorted(loss_dict.keys()))
    return sorted(set().union(*gathered))

def _all_reduce_mean_dict(loss_dict, keys):
    """
    Returning (total, per_key_dict), all ranks receive a consistent average.
    The vector length is fixed at 1 + len(keys). Missing keys in a rank are padded with 0s,
    to avoid all_reduce hangs caused by different tensor lengths across ranks.
    Difference judgment uses the perpetually existing scalar `total`.
    """
    ref = next(iter(loss_dict.values()))
    zero = torch.zeros((), device=ref.device)
    total = sum(v.detach().float().sum() for v in loss_dict.values())
    vals = torch.stack(
        [total] + [loss_dict[k].detach().float().sum() if k in loss_dict else zero for k in keys]
    )
    world = comm.get_world_size()
    if world > 1:
        dist.all_reduce(vals)
        vals = vals / world
    out = vals.cpu().tolist()
    return out[0], dict(zip(keys, out[1:]))

def _has_bad_grad(model) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return False
    norms = torch.stack([g.detach().float().norm() for g in grads])
    return not bool(torch.isfinite(norms).all())

def _log_bad_grad_name(model):
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            logger.warning(f"Non-finite gradient in {name}")
            return


_DDP_PATCHED = False

def _maybe_patch_ddp_find_unused(cfg):
    """
    If cfg.MODEL.DDP_FIND_UNUSED is True (default is False), 
    the DefaultTrainer will include find_unused_parameters=True 
    when creating a DDP. It will only patch once.
    """
    global _DDP_PATCHED
    if _DDP_PATCHED:
        return
    if comm.get_world_size() > 1 and getattr(cfg.MODEL, "DDP_FIND_UNUSED", False):
        d2_defaults.create_ddp_model = functools.partial(
            d2_defaults.create_ddp_model, find_unused_parameters=True
        )
        _DDP_PATCHED = True
        logger.info("[DDP] Enabled find_unused_parameters=True")


# -----------------------------------------------------------
# Build model
# -----------------------------------------------------------
def do_test(cfg, model, iteration="final"):
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    filter_settings["visibility_thres"] = cfg.TEST.VISIBILITY_THRES
    filter_settings["truncation_thres"] = cfg.TEST.TRUNCATION_THRES
    filter_settings["min_height_thres"] = 0.0625
    filter_settings["max_depth"] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", f"iter_{iteration}")

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test,
        filter_settings,
        output_folder,
        iter_label=iteration,
        only_2d=only_2d,
    )

    for dataset_name in dataset_names_test:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        results_json = inference_on_dataset(model, data_loader)

        if comm.is_main_process():
            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.save_predictions(dataset_name)
            eval_helper.evaluate(dataset_name)

            instances = torch.load(
                os.path.join(output_folder, dataset_name, "instances_predictions.pth"),
                weights_only=False,
            )
            log_str = vis.visualize_from_instances(
                instances,
                data_loader.dataset,
                dataset_name,
                cfg.INPUT.MIN_SIZE_TEST,
                os.path.join(output_folder, dataset_name),
                MetadataCatalog.get("omni3d_model").thing_classes,
                iteration,
            )
            logger.info(log_str)

    if comm.is_main_process():
        eval_helper.summarize_all()

class PeriodicEvalHook(HookBase):
    def __init__(self, eval_period: int, do_test_fn):
        self.period = eval_period
        self._do_test = do_test_fn

    def after_step(self):
        if self.period <= 0:
            return
        cur_step = self.trainer.iter + 1
        if cur_step >= self.trainer.max_iter:
            return
        if cur_step % self.period != 0:
            return

        if comm.is_main_process():
            logger.info(f"[Periodic Eval] start eval at step {cur_step}.")
        self._do_test(self.trainer.cfg, self.trainer.train_model, iteration=cur_step)
        comm.synchronize()

# -----------------------------------------------------------
# Trainer
# -----------------------------------------------------------
class CubeYOLOUnifiedTrainer(DefaultTrainer):
    def __init__(self, cfg, priors=None, dataset_id_to_unknown_cats=None, dataset_id_to_src=None):
        self.priors = priors
        self.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats
        self.dataset_id_to_src = dataset_id_to_src

        self.iters_per_epoch = compute_iters_per_epoch(cfg)

        # Loss of explosion-proof switches and statistical status
        self.tolerance = 4.0
        self.gamma = 0.02
        self.recent_loss = None
        self.iterations_explode = 0
        self.iterations_success = 0
        self._loss_keys = None

        if getattr(getattr(cfg.SOLVER, "AMP", None), "ENABLED", False):
            logger.warning("Custom run_step doesn't AMP, please turn off SOLVER.AMP.ENABLED")

        _maybe_patch_ddp_find_unused(cfg)

        super().__init__(cfg)

    @property
    def _core(self):
        return getattr(self, "_trainer", self)

    @property
    def train_model(self):
        return self._core.model

    # ---- builders ----
    def build_model(self, cfg):
        return build_model_for_cfg(cfg, priors=self.priors)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.MODEL.META_ARCHITECTURE == "YOLO3DWrapper":
            return build_optimizer_yolo3d(cfg, model)
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_train_loader(self, cfg):
        data_mapper = DatasetMapper3D(cfg, is_train=True)
        data_mapper.dataset_id_to_unknown_cats = self.dataset_id_to_unknown_cats
        return build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=self.dataset_id_to_src)

    def build_hooks(self):
        update_mode = getattr(self.cfg.MODEL.YOLO3D, "CRITERION_UPDATE_MODE", "step")
        epoch_type = getattr(self.cfg.MODEL.YOLO3D, "CRITERION_EPOCH_TYPE", "auto")

        new_hooks = []
        for h in super().build_hooks():
            if isinstance(h, hooks.EvalHook):
                continue
            if isinstance(h, hooks.PeriodicCheckpointer):
                new_hooks.append(
                    PeriodicCheckpointerOnlyOne(
                        self.checkpointer,
                        self.cfg.SOLVER.CHECKPOINT_PERIOD,
                        max_iter=self.max_iter,
                    )
                )
                continue
            new_hooks.append(h)

        eval_period = self.cfg.TEST.EVAL_PERIOD
        if eval_period > 0 and self.cfg.DATASETS.TEST:
            new_hooks.insert(0, PeriodicEvalHook(eval_period, do_test))

        new_hooks.insert(0, CriterionUpdateHook(self.iters_per_epoch, mode=update_mode, epoch_type=epoch_type))
        return new_hooks

    def run_step(self):
        core = self._core
        model, optimizer = core.model, core.optimizer
        assert model.training, "[Trainer] Model was changed to eval mode!"
        storage = get_event_storage()
        stabilize = self.cfg.MODEL.STABILIZE

        t0 = time.perf_counter()
        batch = next(core._data_loader_iter)
        data_time = time.perf_counter() - t0

        loss_dict = model(batch)
        if isinstance(loss_dict, torch.Tensor):
            loss_dict = {"total_loss": loss_dict}
        losses = sum(loss_dict.values())

        if self._loss_keys is None:
            self._loss_keys = _gather_loss_keys(loss_dict)
        losses_reduced, loss_values = _all_reduce_mean_dict(loss_dict, self._loss_keys)
        finite = math.isfinite(losses_reduced)

        if self.recent_loss is None and finite:
            self.recent_loss = losses_reduced * 2.0

        diverging = stabilize > 0 and (
            not finite
            or (self.recent_loss is not None and losses_reduced > self.recent_loss * self.tolerance)
        )
        if diverging:
            logger.warning(
                f"Skipping gradient update due to abnormal loss {losses_reduced:.4f} "
                f"vs. rolling mean {self.recent_loss}"
            )
        elif finite:
            self.recent_loss = (
                losses_reduced if self.recent_loss is None
                else self.recent_loss * (1 - self.gamma) + losses_reduced * self.gamma
            )

        optimizer.zero_grad()
        losses.backward()

        # Grad NaN / Inf check
        if stabilize > 0 and not diverging and _has_bad_grad(model):
            diverging = True
            if comm.is_main_process():
                _log_bad_grad_name(model)
            logger.warning("Gradient explosion detected, skipping update.")

        # Multi GPU sync
        flag = torch.tensor(float(diverging), device=losses.device)
        if comm.get_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        diverging = flag.item() > 0

        metrics = {"data_time": data_time}
        if finite:
            metrics.update(loss_values)
            metrics["total_loss"] = losses_reduced
        storage.put_scalars(**metrics)

        if diverging:
            optimizer.zero_grad()
            self.iterations_explode += 1
        else:
            optimizer.step()
            self.iterations_success += 1

        # Breaker: If the counts are consistent across all ranks, a circuit breaker will be triggered synchronously.
        total_iters = self.iterations_success + self.iterations_explode
        if (
            stabilize > 0
            and total_iters > self.cfg.SOLVER.CHECKPOINT_PERIOD * 0.5
            and (self.iterations_explode / total_iters) >= stabilize
        ):
            raise ModelDivergedError(
                f"Model diverged: Discarded {self.iterations_explode}/{total_iters} steps (>= {stabilize})"
            )

# -----------------------------------------------------------
# Data preporcess
# -----------------------------------------------------------
def _resolve_config_path(args) -> str:
    config_file = args.config_file
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)
    return config_file

def setup(args):
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    cfg.merge_from_file(_resolve_config_path(args))
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")

    filter_settings = data.get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True)

    for dataset_name in cfg.DATASETS.TEST:
        if dataset_name not in cfg.DATASETS.TRAIN:
            simple_register(dataset_name, filter_settings, filter_empty=False)

    return cfg

def _register_model_metadata_for_eval(args, cfg):
    category_path = os.path.join(cfg.OUTPUT_DIR, "category_meta.json")
    if not os.path.isfile(category_path):
        raise FileNotFoundError(
            f"eval-only can't find category_meta.json：{category_path}。"
        )

    with open(category_path, "r") as f:
        metadata = json.load(f)

    meta = MetadataCatalog.get("omni3d_model")
    meta.thing_classes = metadata["thing_classes"]
    meta.thing_dataset_id_to_contiguous_id = {
        int(k): v for k, v in metadata["thing_dataset_id_to_contiguous_id"].items()
    }
    logger.info(f"[eval-only] loaded category metadata from {category_path}.")

def main(args):
    cfg = setup(args)
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    priors = None

    if args.eval_only:
        _register_model_metadata_for_eval(args, cfg)
        model = build_model_for_cfg(cfg, priors=None)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    dataset_paths = [os.path.join("datasets", "Omni3D", name + ".json") for name in cfg.DATASETS.TRAIN]
    datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)
    data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

    infos = datasets.dataset["info"]
    if isinstance(infos, dict):
        infos = [datasets.dataset["info"]]

    dataset_id_to_unknown_cats = {}
    possible_categories = set(range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
    dataset_id_to_src = {}

    thing_map = MetadataCatalog.get("omni3d_model").thing_dataset_id_to_contiguous_id
    for info in infos:
        dataset_id = info["id"]
        if dataset_id not in dataset_id_to_src:
            dataset_id_to_src[dataset_id] = info["source"]

        known_ids = {thing_map[i] for i in info["known_category_ids"] if i in thing_map}
        dataset_id_to_unknown_cats[dataset_id] = possible_categories - known_ids

    if cfg.MODEL.META_ARCHITECTURE != "YOLO3DWrapper":
        priors = util.compute_priors(cfg, datasets)

    if cfg.MODEL.META_ARCHITECTURE == "YOLO3DWrapper" and cfg.DATASETS.TRAIN:
        cfg.defrost()
        resolve_epoch_iter_config(cfg, mode=getattr(cfg.MODEL.YOLO3D, "TIME_UNIT", "iter"))
        cfg.freeze()

    # 發散重啟迴圈
    base_seed = cfg.SEED
    for attempt in range(1, MAX_TRAINING_ATTEMPTS + 1):
        # 重試時換 seed（所有 rank 的 attempt 序列一致，seed 也一致）。
        # SEED < 0 代表隨機，不需處理。實際是否改變資料順序取決於 sampler 是否讀 cfg.SEED。
        if attempt > 1 and base_seed >= 0:
            cfg.defrost()
            cfg.SEED = base_seed + attempt
            cfg.freeze()

        trainer = CubeYOLOUnifiedTrainer(
            cfg,
            priors=priors,
            dataset_id_to_unknown_cats=dataset_id_to_unknown_cats,
            dataset_id_to_src=dataset_id_to_src,
        )
        # 第二次起強制從最近的 checkpoint 續訓
        trainer.resume_or_load(resume=(args.resume or attempt > 1))

        diverged = False
        try:
            trainer.train()
        except ModelDivergedError as e:
            diverged = True
            logger.warning(f"[attempt {attempt}/{MAX_TRAINING_ATTEMPTS}] {e}. Restarting from last checkpoint...")

        if not diverged:
            return do_test(cfg, trainer.train_model)

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    raise RuntimeError("Training failed: maximum divergence restart attempts reached.")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )