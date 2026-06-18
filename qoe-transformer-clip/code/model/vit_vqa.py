import torch
import torch.nn as nn
from . import full_model


@full_model
class SaliVQA(nn.Module):
    def __init__(self, model_config, cache_path):
        super().__init__()

        self.vqa_in_channels = model_config.vqa_in_channels
        self.vqa_feat_channels = self.vqa_in_channels - 1
        self.video_in_channels = model_config.video_in_channels
        self.video_proj = nn.Conv2d(self.video_in_channels, self.vqa_feat_channels, kernel_size=3, padding=1, bias=False)

        self.vqa_head = nn.Sequential(
            nn.Conv2d(self.vqa_feat_channels, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, bias=True),
        )

    def forward(self, batch, label=None):
        frame = batch["frame"]
        heatmap = batch["salimap"]
        if heatmap is None:
            raise ValueError("heatmap is required in batch for VQA scoring.")

        if frame is None:
            raise ValueError("frame is required in batch for VQA scoring.")

        if frame.dim() != 4 or heatmap.dim() != 4:
            raise ValueError("Expected frame and heatmap to be 4D tensors (B, C, H, W).")
        if frame.shape[-2:] != heatmap.shape[-2:]:
            raise ValueError("Expected frame and heatmap to share spatial dimensions.")

        b, c, h, w = frame.shape
        frame = self.video_proj(frame)
        heatmap = heatmap.reshape(b, 1, h, w)
        x = frame * heatmap
        x = self.vqa_head(x).flatten(1)
        x = self.fc(x).reshape(b, 1)

        vqa_score = x
        result = {"output": vqa_score, "vqa_score": vqa_score}

        if label is not None and "dmos" in label:
            dmos = label["dmos"].to(vqa_score.dtype)
            if dmos.dim() == 1:
                dmos = dmos[:, None].expand_as(vqa_score)
            else:
                dmos = dmos.reshape_as(vqa_score)
            loss_rmse = torch.sqrt(torch.mean((vqa_score - dmos) ** 2))
            result["loss_total"] = loss_rmse
            result["loss_rmse"] = loss_rmse.detach().item()

        return result
