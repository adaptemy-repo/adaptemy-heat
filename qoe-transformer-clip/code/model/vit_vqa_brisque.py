import numpy as np
import torch
import torch.nn as nn
from munch import Munch

from . import full_model


@full_model
class BRISQUE(nn.Module):
    def __init__(self, model_config, cache_path):
        super().__init__()
        if not isinstance(model_config, Munch):
            model_config = Munch(model_config)
        self.model_path = getattr(model_config, "model_path", None)
        self.range_path = getattr(model_config, "range_path", None)
        if self.model_path is None or self.range_path is None:
            raise ValueError("BRISQUE requires model_config.model_path and model_config.range_path.")
        self.brisque = self._load_brisque(self.model_path, self.range_path)

    def _load_brisque(self, model_path, range_path):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV (cv2) is required for BRISQUE scoring.") from exc
        return cv2.quality.QualityBRISQUE_create(str(model_path), str(range_path))

    def forward(self, batch, label=None):
        views = batch.get("views", None)
        if views is None:
            raise ValueError("BRISQUE expects batch['views'] with shape (N, C, H, W).")
        scores = self.score(views)
        return {"output": scores, "qoe": scores}

    def score(self, views):
        if views.dim() == 5:
            views = views.reshape(-1, *views.shape[-3:])
        scores = []
        for view in views:
            scores.append(self._score_single(view))
        return torch.tensor(scores, dtype=torch.float32)

    def _score_single(self, view):
        if view.dim() != 3:
            raise ValueError("BRISQUE expects view with shape (C, H, W).")
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV (cv2) is required for BRISQUE scoring.") from exc

        if view.size(0) == 3:
            img = view.permute(1, 2, 0).contiguous().cpu().numpy()
            if img.dtype != "uint8":
                img = img.astype("float32")
                img = img.clip(0.0, 255.0).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = view[0].contiguous().cpu().numpy()
            if img.dtype != "uint8":
                img = img.astype("float32")
                img = img.clip(0.0, 255.0).astype("uint8")

        score = self.brisque.compute(img)[0]
        if isinstance(score, (list, tuple, np.ndarray)):
            score = float(score[0])
        return score
