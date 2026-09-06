#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import os
import psutil
import torch
import tqdm
from fvcore.common.timer import Timer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data import DatasetFromList, MetadataCatalog
from detectron2.engine import AMPTrainer, SimpleTrainer, default_argument_parser, hooks, launch, default_setup
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

from cubercnn.solver import build_optimizer
from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    DatasetMapper3D,
    build_detection_train_loader,
    build_detection_test_loader,
    simple_register
)
from cubercnn import util, data
import cubercnn.modeling.meta_arch


logger = logging.getLogger("detectron2")

def setup(args):
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.SOLVER.TYPE = "SGD"

    config_file = args.config_file
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.SOLVER.BASE_LR = 0.001 
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")
    
    # Register Omni3D train and test set
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True)
    
    for dataset_name in cfg.DATASETS.TEST:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            simple_register(dataset_name, filter_settings, filter_empty=False)
            
    return cfg

def setup_cubercnn_metadata(cfg, args):
    """Load MetadataCatalog"""
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    
    # Directly fetch category_meta.json from MODEL.WEIGHTS dir
    weight_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
    category_path = os.path.join(weight_dir, 'category_meta.json')
    
    datasets = None
    
    if args.task == "eval" and os.path.exists(category_path):
        metadata = util.load_json(category_path)
        MetadataCatalog.get('omni3d_model').thing_classes = metadata['thing_classes']
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = {
            int(key): val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()
        }
    else:
        dataset_paths = [os.path.join('datasets', 'Omni3D', name + '.json') for name in cfg.DATASETS.TRAIN]
        datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)
        data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

    if datasets is None:
        dataset_paths = [os.path.join('datasets', 'Omni3D', name + '.json') for name in cfg.DATASETS.TRAIN]
        datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)
        
    priors = util.compute_priors(cfg, datasets)
    
    dataset_id_to_unknown_cats = {}
    dataset_id_to_src = {}
    
    infos = datasets.dataset['info']
    if type(infos) == dict:
        infos = [datasets.dataset['info']]

    dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
    possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))

    for info in infos:
        dataset_id = info['id']
        dataset_id_to_src[dataset_id] = info['source']
        known_category_training_ids = set()
        for cat_id in info['known_category_ids']:
            if cat_id in dataset_id_to_contiguous_id:
                known_category_training_ids.add(dataset_id_to_contiguous_id[cat_id])
        dataset_id_to_unknown_cats[dataset_id] = possible_categories - known_category_training_ids

    return priors, dataset_id_to_unknown_cats, dataset_id_to_src

def RAM_msg():
    vram = psutil.virtual_memory()
    return "RAM Usage: {:.2f}/{:.2f} GB".format(
        (vram.total - vram.available) / 1024**3, vram.total / 1024**3
    )

def benchmark_train(args):
    cfg = setup(args)
    priors, dataset_id_to_unknown_cats, dataset_id_to_src = setup_cubercnn_metadata(cfg, args)
    
    model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    logger.info("Model:\n{}".format(model))
    
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )
        
    optimizer = build_optimizer(cfg, model)
    checkpointer = DetectionCheckpointer(model, optimizer=optimizer)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.freeze()
    
    data_mapper = DatasetMapper3D(cfg, is_train=True)
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats
    data_loader = build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src)
    dummy_data = list(itertools.islice(data_loader, 100))

    def f():
        data = DatasetFromList(dummy_data, copy=False, serialize=False)
        while True:
            yield from data

    max_iter = 400
    trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, f(), optimizer)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.PeriodicWriter([CommonMetricPrinter(max_iter)]),
            hooks.TorchProfiler(
                lambda trainer: trainer.iter == max_iter - 1,
                cfg.OUTPUT_DIR,
                save_tensorboard=True,
            ),
        ]
    )
    trainer.train(1, max_iter)

@torch.no_grad()
def benchmark_eval(args):
    cfg = setup(args)
    priors, _, _ = setup_cubercnn_metadata(cfg, args)
    model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    model.eval()
    logger.info("Model:\n{}".format(model))
    dummy_data = DatasetFromList(list(itertools.islice(data_loader, 100)), copy=False)

    def f():
        while True:
            yield from dummy_data

    for k in range(5):  # warmup
        model(dummy_data[k])

    torch.cuda.reset_peak_memory_stats()

    max_iter = 300
    timer = Timer()
    with tqdm.tqdm(total=max_iter) as pbar:
        for idx, d in enumerate(f()):
            if idx == max_iter:
                break
            model(d)
            pbar.update()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    logger.info("{} iters in {} seconds.".format(max_iter, timer.seconds()))
    logger.info("Peak GPU Memory Usage: {:.2f} MB".format(peak_mem_mb))

def main() -> None:
    parser = default_argument_parser()
    parser.add_argument("--task", choices=["train", "eval"], required=True)
    args = parser.parse_args()
    assert not args.eval_only

    logger.info("Environment info:\n" + collect_env_info())
    
    if args.task == "train":
        f = benchmark_train
    elif args.task == "eval":
        f = benchmark_eval
        assert args.num_gpus == 1 and args.num_machines == 1
        
    launch(
        f,
        args.num_gpus,
        args.num_machines,
        args.machine_rank,
        args.dist_url,
        args=(args,),
    )

if __name__ == "__main__":
    main()
