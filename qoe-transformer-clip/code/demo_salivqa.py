import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from exp import ex
from ckpt import load_ckpt_adapt
from model import get_model
from data.vqa_dmos import VQA_DMOS
from matplotlib import pyplot as plt


def _rankdata(values):
    values = np.asarray(values)
    n = values.size
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[sorter] = np.arange(n, dtype=float)

    sorted_vals = values[sorter]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            ranks[sorter[i:j + 1]] = avg
        i = j + 1

    return ranks


def _spearmanr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx_std = rx.std()
    ry_std = ry.std()
    if rx_std == 0 or ry_std == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _reduce_weight(weight, reduce_mode):
    if torch.is_tensor(weight):
        if weight.numel() == 1:
            return float(weight.item())
        if reduce_mode == "sum":
            return float(weight.sum().item())
        if reduce_mode == "max":
            return float(weight.max().item())
        return float(weight.mean().item())
    return float(weight)

def _heatmap_to_rgb(heatmap, size):
    hmap = heatmap.numpy()
    hmap = hmap - np.min(hmap)
    hmap = hmap / (np.max(hmap) - np.min(hmap) + 1e-8)
    hmap = plt.get_cmap("jet")(hmap, bytes=True)[..., :3]
    img = Image.fromarray(hmap, mode="RGB").resize(size, resample=Image.BILINEAR)
    return img


@ex.capture()
def demo_salivqa(device, list_path, data_path, cache_path, rebuild_cache,
                 num_workers, clip_length, extractor_name, feature_norm, frame_stride,
                 model_config_pos, model_config_neg, sta_config, fixed_width=448, fixed_height=224,
                 ckpt_path=None, ckpt_name=None, mode="test", weight_reduce="mean",
                 vis_dir="/home/ivan/projects/heat/ViT-ODV/data/vis", vis_every=10, vis_max=10, vis_overlay=True):
    dataset = VQA_DMOS(
        data_path=data_path,
        cache_path=cache_path,
        rebuild_cache=rebuild_cache,
        num_workers=num_workers,
        clip_length=clip_length,
        extractor_name=extractor_name,
        feature_norm=feature_norm,
        frame_stride=frame_stride,
        model_config_pos=model_config_pos,
        model_config_neg=model_config_neg,
        sta_config=sta_config,
        list_path=list_path,
        mode=mode,
        device=device,
        fixed_width=fixed_width,
        fixed_height=fixed_height,
    )

    model = get_model().to(device)
    model = load_ckpt_adapt(model=model, ckpt_name=ckpt_name, ckpt_path=ckpt_path)
    model.eval()

    pred_sum = defaultdict(float)
    weight_sum = defaultdict(float)
    gt_dmos = {}
    gt_sum = defaultdict(float)
    vis_written = 0
    vis_every = int(vis_every)
    vis_max = int(vis_max)
    if vis_every > 0:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset.items, desc="SaliVQA Eval")):
            frame = item["frame"].unsqueeze(0).to(device, non_blocking=True)
            salimap = item["salimap"].unsqueeze(0).to(device, non_blocking=True)
            result = model({"frame": frame, "salimap": salimap})
            score = float(result["vqa_score"].item())

            weight = _reduce_weight(item.get("weight", 1.0), weight_reduce)
            video_id = item["video_id"]
            segment_id = item.get("segment_id", idx)

            if vis_every > 0 and (idx % vis_every == 0) and (vis_max <= 0 or vis_written < vis_max):
                frame_np = item["frame"].clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                frame_img = Image.fromarray(frame_np)
                frame_img.save(vis_dir / f"{video_id}_{segment_id:06d}_frame.png")

                heatmap = item["salimap"].squeeze(0).cpu()
                frame_h, frame_w = item["frame"].shape[-2], item["frame"].shape[-1]
                heatmap_img = _heatmap_to_rgb(heatmap, (frame_w, frame_h))
                heatmap_img.save(vis_dir / f"{video_id}_{segment_id:06d}_heatmap.png")

                if vis_overlay:
                    overlay_frame = item["frame"].clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    overlay_frame = Image.fromarray(overlay_frame)
                    overlay_img = Image.blend(overlay_frame, heatmap_img, 0.5)
                    overlay_img.save(vis_dir / f"{video_id}_{segment_id:06d}_overlay.png")

                vis_written += 1

            pred_sum[video_id] += score * weight
            weight_sum[video_id] += weight
            if video_id not in gt_dmos:
                gt_dmos[video_id] = float(item["dmos"].item())
            gt_sum[video_id] += gt_dmos[video_id] * weight

    y_true = []
    y_pred = []
    for video_id, gt in gt_dmos.items():
        denom = weight_sum[video_id]
        pred = pred_sum[video_id] / denom if denom != 0 else math.nan
        gt_weighted = gt_sum[video_id] / denom if denom != 0 else math.nan
        if not math.isnan(pred) and not math.isnan(gt_weighted):
            y_true.append(gt_weighted)
            y_pred.append(pred)

    srocc = _spearmanr(y_true, y_pred)
    plcc = _pearsonr(y_true, y_pred)
    rmse = math.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)) if y_true else float("nan")
    print(f"[demo_salivqa] videos={len(y_true)} srocc={srocc:.6f} plcc={plcc:.6f} rmse={rmse:.6f}")
    print(f"[demo_salivqa] y_true={y_true}")
    print(f"[demo_salivqa] y_pred={y_pred}")
    return {"srocc": srocc, "plcc": plcc, "rmse": rmse, "y_true": y_true, "y_pred": y_pred}
