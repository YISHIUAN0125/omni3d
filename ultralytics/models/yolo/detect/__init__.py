# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

from .train_3d import Detection3DTrainer
from .val_3d import Detection3DValidator
from .predict_3d import Detection3DPredictor

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "Detection3DTrainer", "Detection3DValidator", "Detection3DPredictor"
