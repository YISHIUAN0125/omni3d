# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.config import CfgNode as CN


def get_cfg_defaults(cfg):
    # -------------------------------------------------------------------------
    # Dataset & Dataloader Options
    # -------------------------------------------------------------------------
    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []

    # The category names which will be treated as ignore
    # e.g., not counting as background during training or false positives during eval
    cfg.DATASETS.IGNORE_NAMES = []

    # Should the datasets appear with the same probability in batches
    cfg.DATALOADER.BALANCE_DATASETS = False

    # Thresholds for truncation and visibility
    cfg.DATASETS.TRUNCATION_THRES = 0.99
    cfg.DATASETS.VISIBILITY_THRES = 0.01
    cfg.DATASETS.MIN_HEIGHT_THRES = 0.00
    cfg.DATASETS.MAX_DEPTH = 1e8

    # Whether modal 2D boxes should be loaded, or full 3D projected boxes
    cfg.DATASETS.MODAL_2D_BOXES = False

    # Whether truncated 2D boxes should be loaded, or full 3D projected boxes
    cfg.DATASETS.TRUNC_2D_BOXES = True

    # -------------------------------------------------------------------------
    # General Model & Backbone Options
    # -------------------------------------------------------------------------
    # Threshold used for matching and filtering boxes inside ignore regions
    cfg.MODEL.RPN.IGNORE_THRESHOLD = 0.5
    cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY = 'IoUness'

    # Batch Normalization control
    cfg.MODEL.USE_BN = True

    # Gradient stabilization threshold (0.0 is off, 0.01 recommended)
    cfg.MODEL.STABILIZE = 0.01

    # DLA Backbone settings
    cfg.MODEL.DLA = CN()
    cfg.MODEL.DLA.TYPE = 'dla34'
    cfg.MODEL.DLA.TRICKS = False

    cfg.MODEL.RESNETS.TORCHVISION = True
    cfg.MODEL.WEIGHTS_PRETRAIN = ''

    # -------------------------------------------------------------------------
    # 2-Stage ROI Cube Head Options (Original Baseline)
    # -------------------------------------------------------------------------
    cfg.MODEL.ROI_CUBE_HEAD = CN()
    cfg.MODEL.ROI_CUBE_HEAD.NAME = "CubeHead"
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE = "ROIAlignV2"

    cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_CUBE_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_CUBE_HEAD.FC_DIM = 1024

    cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE = "direct"
    cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE = "6d"
    cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT = False
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH = True
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL = 512.0
    cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS = True
    cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS = 1
    cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE = True
    cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE = True
    cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC = True
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED = True
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC = 'exp'
    cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES = 0.0

    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT = 1.0

    # -------------------------------------------------------------------------
    # Single-Stage Dense Cube Head Options (DenseCubeHead + SimOTAAssigner)
    # -------------------------------------------------------------------------
    cfg.MODEL.DENSE_HEAD = CN()
    cfg.MODEL.DENSE_HEAD.NAME = "DenseCubeHead"
    cfg.MODEL.DENSE_HEAD.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
    cfg.MODEL.DENSE_HEAD.NUM_CLASSES = 80
    cfg.MODEL.DENSE_HEAD.NUM_CONVS = 4
    cfg.MODEL.DENSE_HEAD.CONV_DIM = 256
    cfg.MODEL.DENSE_HEAD.POSE_TYPE = "6d"               # ['6d', 'euler', 'quaternion']
    cfg.MODEL.DENSE_HEAD.USE_CONFIDENCE = True          # Predict uncertainty confidence
    cfg.MODEL.DENSE_HEAD.CLUSTER_BINS = 1               # Number of depth clusters (1 = disabled)
    cfg.MODEL.DENSE_HEAD.PRIOR_PROB = 0.01              # Sigmoid focal loss prior initialization
    cfg.MODEL.DENSE_HEAD.FOCAL_ALPHA = 0.25
    cfg.MODEL.DENSE_HEAD.FOCAL_GAMMA = 2.0

    # SimOTA Assigner settings (assigner.py)
    cfg.MODEL.DENSE_HEAD.ASSIGNER = "SimOTAAssigner"
    cfg.MODEL.DENSE_HEAD.CENTER_RADIUS = 2.5            # Center sampling radius in strides
    cfg.MODEL.DENSE_HEAD.CANDIDATE_TOPK = 10            # Top-k candidate IoU sum for dynamic K
    cfg.MODEL.DENSE_HEAD.CLS_COST_WEIGHT = 1.0          # Classification cost weight
    cfg.MODEL.DENSE_HEAD.IOU_COST_WEIGHT = 3.0          # 2D IoU cost weight

    # Dense Head Loss Weights
    cfg.MODEL.DENSE_HEAD.LOSS_W_3D = 1.0                # Overall 3D loss scaling
    cfg.MODEL.DENSE_HEAD.LOSS_W_CLS = 1.0               # Focal loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_BOX2D = 1.0             # GIoU loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_CENTERNESS = 1.0        # Centerness BCE loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_XY = 1.0                # Projected 2D center XY loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_Z = 1.0                 # Depth Z loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_DIMS = 1.0              # 3D dimensions loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_POSE = 1.0              # 3D rotation pose loss weight
    cfg.MODEL.DENSE_HEAD.LOSS_W_JOINT = 0.0             # 3D full cuboid joint loss weight

    # 3D Geometric & Prior settings
    cfg.MODEL.DENSE_HEAD.Z_TYPE = "log"                 # ['direct', 'sigmoid', 'log', 'clusters']
    cfg.MODEL.DENSE_HEAD.DIMS_PRIORS_ENABLED = True
    cfg.MODEL.DENSE_HEAD.DIMS_PRIORS_FUNC = "exp"       # ['exp', 'sigmoid']
    cfg.MODEL.DENSE_HEAD.CHAMFER_POSE = False           # Use Chamfer L1 distance for pose & joint
    cfg.MODEL.DENSE_HEAD.DISENTANGLED_LOSS = True       # Compute loss per disentangled group
    cfg.MODEL.DENSE_HEAD.INVERSE_Z_WEIGHT = True        # Scale 3D loss by inverse depth
    cfg.MODEL.DENSE_HEAD.ALLOCENTRIC_POSE = True        # Predict allocentric rotation
    cfg.MODEL.DENSE_HEAD.VIRTUAL_DEPTH = True           # Use virtual camera focal length
    cfg.MODEL.DENSE_HEAD.VIRTUAL_FOCAL = 512.0

    # Dense Head Inference & NMS settings
    cfg.MODEL.DENSE_HEAD.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DENSE_HEAD.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.DENSE_HEAD.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DENSE_HEAD.MAX_DETECTIONS_PER_IMAGE = 100

    # -------------------------------------------------------------------------
    # Solver & Testing Options
    # -------------------------------------------------------------------------
    cfg.SOLVER.TYPE = 'sgd'
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.TEST.VISIBILITY_THRES = 0.5
    cfg.TEST.TRUNCATION_THRES = 0.5
    cfg.INPUT.RANDOM_FLIP = "horizontal"
