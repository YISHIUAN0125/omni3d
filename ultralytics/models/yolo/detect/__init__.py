# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

from .train_3d import Detection3DTrainer
from .val_3d import Detection3DValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
