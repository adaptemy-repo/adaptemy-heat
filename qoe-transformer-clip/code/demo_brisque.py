import math
from collections import defaultdict

import numpy as np
import scipy.stats
import torch
from tqdm import tqdm

from exp import ex
from data.vqa_dmos_views import VQA_DMOS_VIEWS
from data.vqa_dmos_views_cluster import VQA_DMOS_VIEWS_CLUSTER
from model.vit_vqa_brisque import BRISQUE


@ex.capture()
def demo_brisque_tmp0(device, list_path, data_path, cache_path, rebuild_cache,
                 num_workers, clip_length, extractor_name, feature_norm, frame_stride,
                 model_config_pos, model_config_neg, sta_config, brisque_model_path, brisque_range_path,
                 saliency_width=448, saliency_height=224, frame_width=2048, frame_height=1024, mode="test"):
    dataset = VQA_DMOS_VIEWS(
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
        saliency_width=saliency_width,
        saliency_height=saliency_height,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if brisque_model_path is None or brisque_range_path is None:
        raise ValueError("demo_brisque requires brisque_model_path and brisque_range_path.")

    brisque = BRISQUE(
        {"model_path": brisque_model_path, "range_path": brisque_range_path},
        cache_path=cache_path,
    )
    brisque.eval()

    per_video_sum = defaultdict(float)
    per_video_count = defaultdict(int)
    gt_dmos = {}

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="BRISQUE Eval"):
            data, label, meta = dataset[idx]
            views = data["views"]
            weights = label.get("view_weights", None)
            mask = label.get("view_mask", None)
            scores = brisque.score(views)

            if weights is None:
                frame_score = float(scores.mean().item())
            else:
                if mask is not None:
                    weights = weights * mask
                denom = float(weights.sum().item())
                if denom <= 0:
                    frame_score = float(scores.mean().item())
                else:
                    frame_score = float((scores * weights).sum().item() / denom)

            video_id = meta["video_id"]
            per_video_sum[video_id] += frame_score
            per_video_count[video_id] += 1
            if video_id not in gt_dmos:
                gt_dmos[video_id] = float(label["dmos"].item())

    video_scores = {}
    for video_id, total in per_video_sum.items():
        count = per_video_count[video_id]
        video_scores[video_id] = total / count if count > 0 else math.nan

    for video_id, score in sorted(video_scores.items()):
        gt = gt_dmos.get(video_id, float("nan"))
        print(f"{video_id}\t{score:.6f}\tgt_dmos={gt:.6f}")

    pred_vals = []
    gt_vals = []
    for video_id, score in sorted(video_scores.items()):
        gt = gt_dmos.get(video_id, float("nan"))
        if math.isnan(gt) or math.isnan(score):
            continue
        pred_vals.append(score)
        gt_vals.append(gt)

    if len(pred_vals) >= 2:
        pred_arr = np.array(pred_vals, dtype=np.float32)
        gt_arr = np.array(gt_vals, dtype=np.float32)
        pred_mean = pred_arr.mean()
        gt_mean = gt_arr.mean()
        pred_std = pred_arr.std()
        gt_std = gt_arr.std()
        if pred_std > 0 and gt_std > 0:
            plcc = float(((pred_arr - pred_mean) * (gt_arr - gt_mean)).mean() / (pred_std * gt_std))
            print(f"PLCC\t{plcc:.6f}")
        else:
            print("PLCC\tNaN (zero variance)")
    else:
        print("PLCC\tNaN (insufficient data)")

    if len(pred_vals) >= 2:
        srcc, _ = scipy.stats.spearmanr(pred_arr, gt_arr)
        print(f"SRCC\t{float(srcc):.6f}")
    else:
        print("SRCC\tNaN (insufficient data)")

    # 4-parameter logistic mapping before PLCC (standard in VQA)
    if len(pred_vals) >= 4:
        try:
            from scipy.optimize import curve_fit

            def logistic(x, beta1, beta2, beta3, beta4):
                return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / (np.abs(beta4) + 1e-8)))

            x = pred_arr.astype(np.float64)
            y = gt_arr.astype(np.float64)
            p0 = [y.max(), y.min(), x.mean(), x.std() if x.std() > 0 else 1.0]
            params, _ = curve_fit(logistic, x, y, p0=p0, maxfev=20000)
            y_hat = logistic(x, *params)
            y_mean = y.mean()
            y_hat_mean = y_hat.mean()
            y_std = y.std()
            y_hat_std = y_hat.std()
            if y_std > 0 and y_hat_std > 0:
                plcc_mapped = float(((y - y_mean) * (y_hat - y_hat_mean)).mean() / (y_std * y_hat_std))
                print(f"PLCC_MAPPED4\t{plcc_mapped:.6f}")
            else:
                print("PLCC_MAPPED4\tNaN (zero variance)")
        except Exception as exc:
            print(f"PLCC_MAPPED4\tNaN (logistic fit failed: {exc})")
    else:
        print("PLCC_MAPPED4\tNaN (insufficient data)")

    # 5-parameter logistic mapping before PLCC
    if len(pred_vals) >= 5:
        try:
            from scipy.optimize import curve_fit

            def logistic5(x, beta1, beta2, beta3, beta4, beta5):
                return beta2 + (beta1 - beta2) / (1.0 + np.exp(-((x - beta3) / (np.abs(beta4) + 1e-8)))) ** beta5

            x = pred_arr.astype(np.float64)
            y = gt_arr.astype(np.float64)
            p0 = [y.max(), y.min(), x.mean(), x.std() if x.std() > 0 else 1.0, 1.0]
            params, _ = curve_fit(logistic5, x, y, p0=p0, maxfev=30000)
            y_hat = logistic5(x, *params)
            y_mean = y.mean()
            y_hat_mean = y_hat.mean()
            y_std = y.std()
            y_hat_std = y_hat.std()
            if y_std > 0 and y_hat_std > 0:
                plcc_mapped5 = float(((y - y_mean) * (y_hat - y_hat_mean)).mean() / (y_std * y_hat_std))
                print(f"PLCC_MAPPED5\t{plcc_mapped5:.6f}")
            else:
                print("PLCC_MAPPED5\tNaN (zero variance)")
        except Exception as exc:
            print(f"PLCC_MAPPED5\tNaN (logistic fit failed: {exc})")
    else:
        print("PLCC_MAPPED5\tNaN (insufficient data)")

    return video_scores


@ex.capture()
def demo_brisque(device, list_path, data_path, cache_path, rebuild_cache,
                 num_workers, clip_length, extractor_name, feature_norm, frame_stride,
                 model_config_pos, model_config_neg, sta_config, brisque_model_path, brisque_range_path,
                 saliency_width=448, saliency_height=224, frame_width=2048, frame_height=1024,
                 cluster_iters=10, cluster_min_weight=0.5, mode="test",
                 save_views_dir="/home/ivan/projects/heat/ViT-ODV/data/views",
                 save_views_per_video=1):
    dataset = VQA_DMOS_VIEWS_CLUSTER(
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
        saliency_width=saliency_width,
        saliency_height=saliency_height,
        frame_width=frame_width,
        frame_height=frame_height,
        cluster_iters=cluster_iters,
        cluster_min_weight=cluster_min_weight,
    )

    if brisque_model_path is None or brisque_range_path is None:
        raise ValueError("demo_brisque_cluster requires brisque_model_path and brisque_range_path.")

    brisque = BRISQUE(
        {"model_path": brisque_model_path, "range_path": brisque_range_path},
        cache_path=cache_path,
    )
    brisque.eval()

    per_video_sum = defaultdict(float)
    per_video_count = defaultdict(int)
    gt_dmos = {}

    all_view_centers = defaultdict(list)
    all_view_weights = defaultdict(list)
    saved_per_video = defaultdict(int)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="BRISQUE Eval (Cluster)"):
            data, label, meta = dataset[idx]
            views = data["views"]
            weights = label.get("view_weights", None)
            mask = label.get("view_mask", None)
            all_view_centers[meta["video_id"]].append(meta.get("view_centers"))
            all_view_weights[meta["video_id"]].append(weights)
            scores = brisque.score(views)

            if weights is None:
                frame_score = float(scores.mean().item())
            else:
                if mask is not None:
                    weights = weights * mask
                denom = float(weights.sum().item())
                if denom <= 0:
                    frame_score = float(scores.mean().item())
                else:
                    frame_score = float((scores * weights).sum().item() / denom)

            video_id = meta["video_id"]
            per_video_sum[video_id] += frame_score
            per_video_count[video_id] += 1
            if video_id not in gt_dmos:
                gt_dmos[video_id] = float(label["dmos"].item())

            if save_views_per_video > 0 and saved_per_video[video_id] < save_views_per_video:
                import os
                import cv2
                os.makedirs(save_views_dir, exist_ok=True)
                # Save original frame
                frame = data.get("frame", None)
                if frame is not None:
                    img = frame.permute(1, 2, 0).contiguous().cpu().numpy()
                    if img.dtype != np.uint8:
                        img = img.clip(0.0, 255.0).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_name = f"{video_id}_seg{meta['segment_id']}_frame.png"
                    cv2.imwrite(os.path.join(save_views_dir, out_name), img)
                # Save heatmap
                salimap = data.get("salimap", None)
                if salimap is not None:
                    heat = salimap.squeeze().contiguous().cpu().numpy()
                    heat = heat - heat.min()
                    denom = heat.max() - heat.min()
                    if denom > 0:
                        heat = heat / denom
                    heat_img = (heat * 255.0).clip(0, 255).astype(np.uint8)
                    heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
                    out_name = f"{video_id}_seg{meta['segment_id']}_heatmap.png"
                    cv2.imwrite(os.path.join(save_views_dir, out_name), heat_img)
                # views: (V, C, H, W)
                for v in range(views.shape[0]):
                    view = views[v]
                    img = view.permute(1, 2, 0).contiguous().cpu().numpy()
                    if img.dtype != np.uint8:
                        img = img.clip(0.0, 255.0).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out_name = f"{video_id}_seg{meta['segment_id']}_v{v}.png"
                    cv2.imwrite(os.path.join(save_views_dir, out_name), img)
                saved_per_video[video_id] += 1

    video_scores = {}
    for video_id, total in per_video_sum.items():
        count = per_video_count[video_id]
        video_scores[video_id] = total / count if count > 0 else math.nan

    for video_id, score in sorted(video_scores.items()):
        gt = gt_dmos.get(video_id, float("nan"))
        print(f"{video_id}\t{score:.6f}\tgt_dmos={gt:.6f}")

    print("VIEWPORTS_AND_WEIGHTS_BEGIN")
    for video_id in sorted(all_view_centers.keys()):
        print(f"{video_id}")
        centers_list = all_view_centers[video_id]
        weights_list = all_view_weights[video_id]
        for i in range(len(centers_list)):
            print(f"  segment_{i}")
            print(f"    centers={centers_list[i]}")
            print(f"    weights={weights_list[i]}")
    print("VIEWPORTS_AND_WEIGHTS_END")

    pred_vals = []
    gt_vals = []
    for video_id, score in sorted(video_scores.items()):
        gt = gt_dmos.get(video_id, float("nan"))
        if math.isnan(gt) or math.isnan(score):
            continue
        pred_vals.append(score)
        gt_vals.append(gt)

    if len(pred_vals) >= 2:
        pred_arr = np.array(pred_vals, dtype=np.float32)
        gt_arr = np.array(gt_vals, dtype=np.float32)
        pred_mean = pred_arr.mean()
        gt_mean = gt_arr.mean()
        pred_std = pred_arr.std()
        gt_std = gt_arr.std()
        if pred_std > 0 and gt_std > 0:
            plcc = float(((pred_arr - pred_mean) * (gt_arr - gt_mean)).mean() / (pred_std * gt_std))
            print(f"PLCC\t{plcc:.6f}")
        else:
            print("PLCC\tNaN (zero variance)")
    else:
        print("PLCC\tNaN (insufficient data)")

    if len(pred_vals) >= 2:
        srcc, _ = scipy.stats.spearmanr(pred_arr, gt_arr)
        print(f"SRCC\t{float(srcc):.6f}")
    else:
        print("SRCC\tNaN (insufficient data)")

    # 4-parameter logistic mapping before PLCC (standard in VQA)
    if len(pred_vals) >= 4:
        try:
            from scipy.optimize import curve_fit

            def logistic(x, beta1, beta2, beta3, beta4):
                return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / (np.abs(beta4) + 1e-8)))

            x = pred_arr.astype(np.float64)
            y = gt_arr.astype(np.float64)
            p0 = [y.max(), y.min(), x.mean(), x.std() if x.std() > 0 else 1.0]
            params, _ = curve_fit(logistic, x, y, p0=p0, maxfev=20000)
            y_hat = logistic(x, *params)
            y_mean = y.mean()
            y_hat_mean = y_hat.mean()
            y_std = y.std()
            y_hat_std = y_hat.std()
            if y_std > 0 and y_hat_std > 0:
                plcc_mapped = float(((y - y_mean) * (y_hat - y_hat_mean)).mean() / (y_std * y_hat_std))
                print(f"PLCC_MAPPED4\t{plcc_mapped:.6f}")
            else:
                print("PLCC_MAPPED4\tNaN (zero variance)")
        except Exception as exc:
            print(f"PLCC_MAPPED4\tNaN (logistic fit failed: {exc})")
    else:
        print("PLCC_MAPPED4\tNaN (insufficient data)")

    # 5-parameter logistic mapping before PLCC
    if len(pred_vals) >= 5:
        try:
            from scipy.optimize import curve_fit

            def logistic5(x, beta1, beta2, beta3, beta4, beta5):
                return beta2 + (beta1 - beta2) / (1.0 + np.exp(-((x - beta3) / (np.abs(beta4) + 1e-8)))) ** beta5

            x = pred_arr.astype(np.float64)
            y = gt_arr.astype(np.float64)
            p0 = [y.max(), y.min(), x.mean(), x.std() if x.std() > 0 else 1.0, 1.0]
            params, _ = curve_fit(logistic5, x, y, p0=p0, maxfev=30000)
            y_hat = logistic5(x, *params)
            y_mean = y.mean()
            y_hat_mean = y_hat.mean()
            y_std = y.std()
            y_hat_std = y_hat.std()
            if y_std > 0 and y_hat_std > 0:
                plcc_mapped5 = float(((y - y_mean) * (y_hat - y_hat_mean)).mean() / (y_std * y_hat_std))
                print(f"PLCC_MAPPED5\t{plcc_mapped5:.6f}")
            else:
                print("PLCC_MAPPED5\tNaN (zero variance)")
        except Exception as exc:
            print(f"PLCC_MAPPED5\tNaN (logistic fit failed: {exc})")
    else:
        print("PLCC_MAPPED5\tNaN (insufficient data)")

    return video_scores


@ex.capture()
def demo_brisque_tmp(device, list_path, data_path, cache_path, rebuild_cache,
                   num_workers, clip_length, extractor_name, feature_norm, frame_stride,
                   model_config_pos, model_config_neg, sta_config, brisque_model_path, brisque_range_path,
                   saliency_width=448, saliency_height=224, frame_width=2048, frame_height=1024,
                   top_k=1, mode="test"):
    dataset = VQA_DMOS_VIEWS(
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
        saliency_width=saliency_width,
        saliency_height=saliency_height,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if brisque_model_path is None or brisque_range_path is None:
        raise ValueError("demo_brisque_k requires brisque_model_path and brisque_range_path.")

    brisque = BRISQUE(
        {"model_path": brisque_model_path, "range_path": brisque_range_path},
        cache_path=cache_path,
    )
    brisque.eval()

    per_video_sum = defaultdict(float)
    per_video_count = defaultdict(int)
    gt_dmos = {}

    top_k = int(top_k)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="BRISQUE Eval (Top-K)"):
            data, label, meta = dataset[idx]
            views = data["views"]
            weights = label.get("view_weights", None)
            mask = label.get("view_mask", None)
            scores = brisque.score(views)

            if weights is None:
                frame_score = float(scores.mean().item())
            else:
                if mask is not None:
                    weights = weights * mask
                valid = weights > 0
                if idx == 0:
                    print(f"weights={weights.detach().cpu().numpy()}")
                    print(f"valid_count={int(valid.sum().item())}")
                if valid.any():
                    if top_k > 0:
                        k = min(top_k, int(valid.sum().item()))
                        top_idx = torch.topk(weights, k).indices
                        if idx == 0:
                            print(f"top_k={k} top_idx={top_idx.detach().cpu().numpy()}")
                        scores_k = scores[top_idx]
                        weights_k = weights[top_idx]
                    else:
                        scores_k = scores
                        weights_k = weights
                    denom = float(weights_k.sum().item())
                    if denom > 0:
                        frame_score = float((scores_k * weights_k).sum().item() / denom)
                    else:
                        frame_score = float(scores.mean().item())
                else:
                    frame_score = float(scores.mean().item())

            video_id = meta["video_id"]
            per_video_sum[video_id] += frame_score
            per_video_count[video_id] += 1
            if video_id not in gt_dmos:
                gt_dmos[video_id] = float(label["dmos"].item())

    video_scores = {}
    for video_id, total in per_video_sum.items():
        count = per_video_count[video_id]
        video_scores[video_id] = total / count if count > 0 else math.nan

    for video_id, score in sorted(video_scores.items()):
        gt = gt_dmos.get(video_id, float("nan"))
        print(f"{video_id}\t{score:.6f}\tgt_dmos={gt:.6f}")

    pred_vals = []
    gt_vals = []
    for video_id, score in sorted(video_scores.items()):
        gt = gt_dmos.get(video_id, float("nan"))
        if math.isnan(gt) or math.isnan(score):
            continue
        pred_vals.append(score)
        gt_vals.append(gt)

    if len(pred_vals) >= 2:
        pred_arr = np.array(pred_vals, dtype=np.float32)
        gt_arr = np.array(gt_vals, dtype=np.float32)
        pred_mean = pred_arr.mean()
        gt_mean = gt_arr.mean()
        pred_std = pred_arr.std()
        gt_std = gt_arr.std()
        if pred_std > 0 and gt_std > 0:
            plcc = float(((pred_arr - pred_mean) * (gt_arr - gt_mean)).mean() / (pred_std * gt_std))
            print(f"PLCC\t{plcc:.6f}")
        else:
            print("PLCC\tNaN (zero variance)")
    else:
        print("PLCC\tNaN (insufficient data)")

    if len(pred_vals) >= 2:
        srcc, _ = scipy.stats.spearmanr(pred_arr, gt_arr)
        print(f"SRCC\t{float(srcc):.6f}")
    else:
        print("SRCC\tNaN (insufficient data)")

    # 4-parameter logistic mapping before PLCC (standard in VQA)
    if len(pred_vals) >= 4:
        try:
            from scipy.optimize import curve_fit

            def logistic(x, beta1, beta2, beta3, beta4):
                return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / (np.abs(beta4) + 1e-8)))

            x = pred_arr.astype(np.float64)
            y = gt_arr.astype(np.float64)
            p0 = [y.max(), y.min(), x.mean(), x.std() if x.std() > 0 else 1.0]
            params, _ = curve_fit(logistic, x, y, p0=p0, maxfev=20000)
            y_hat = logistic(x, *params)
            y_mean = y.mean()
            y_hat_mean = y_hat.mean()
            y_std = y.std()
            y_hat_std = y_hat.std()
            if y_std > 0 and y_hat_std > 0:
                plcc_mapped = float(((y - y_mean) * (y_hat - y_hat_mean)).mean() / (y_std * y_hat_std))
                print(f"PLCC_MAPPED4\t{plcc_mapped:.6f}")
            else:
                print("PLCC_MAPPED4\tNaN (zero variance)")
        except Exception as exc:
            print(f"PLCC_MAPPED4\tNaN (logistic fit failed: {exc})")
    else:
        print("PLCC_MAPPED4\tNaN (insufficient data)")

    # 5-parameter logistic mapping before PLCC
    if len(pred_vals) >= 5:
        try:
            from scipy.optimize import curve_fit

            def logistic5(x, beta1, beta2, beta3, beta4, beta5):
                return beta2 + (beta1 - beta2) / (1.0 + np.exp(-((x - beta3) / (np.abs(beta4) + 1e-8)))) ** beta5

            x = pred_arr.astype(np.float64)
            y = gt_arr.astype(np.float64)
            p0 = [y.max(), y.min(), x.mean(), x.std() if x.std() > 0 else 1.0, 1.0]
            params, _ = curve_fit(logistic5, x, y, p0=p0, maxfev=30000)
            y_hat = logistic5(x, *params)
            y_mean = y.mean()
            y_hat_mean = y_hat.mean()
            y_std = y.std()
            y_hat_std = y_hat.std()
            if y_std > 0 and y_hat_std > 0:
                plcc_mapped5 = float(((y - y_mean) * (y_hat - y_hat_mean)).mean() / (y_std * y_hat_std))
                print(f"PLCC_MAPPED5\t{plcc_mapped5:.6f}")
            else:
                print("PLCC_MAPPED5\tNaN (zero variance)")
        except Exception as exc:
            print(f"PLCC_MAPPED5\tNaN (logistic fit failed: {exc})")
    else:
        print("PLCC_MAPPED5\tNaN (insufficient data)")

    return video_scores
