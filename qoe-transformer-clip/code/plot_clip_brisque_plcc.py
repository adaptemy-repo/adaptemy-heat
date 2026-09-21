import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from data.vqa_dmos_views_cluster import VQA_DMOS_VIEWS_CLUSTER
from model.vit_vqa_brisque import BRISQUE


def evaluate_all_videos(args, project_root):
    with open(project_root / "code/configs/extractor/adaptvit.json", "r", encoding="utf-8") as handle:
        extractor_config = json.load(handle)
    with open(project_root / "code/configs/model/stadecoder_single_clip.json", "r", encoding="utf-8") as handle:
        sta_config = json.load(handle)

    sta_config["model_config"]["feature_branch"] = "pos"
    sta_config["ckpt_path"] = args.clip_checkpoint
    sta_config["ckpt_name"] = args.clip_checkpoint_name

    dataset = VQA_DMOS_VIEWS_CLUSTER(
        data_path=project_root / "data",
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        num_workers=args.num_workers,
        clip_length=args.clip_length,
        extractor_name=extractor_config["extractor_name"],
        feature_norm=extractor_config["feature_norm"],
        frame_stride=args.frame_stride,
        model_config_pos=extractor_config["model_config_pos"],
        model_config_neg=extractor_config["model_config_neg"],
        sta_config=sta_config,
        list_path=args.dmos_list,
        mode="all",
        device=args.device,
        saliency_width=args.saliency_width,
        saliency_height=args.saliency_height,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        cluster_iters=args.cluster_iters,
        cluster_min_weight=args.cluster_min_weight,
        cache_tag="clip_pos_all180",
    )

    brisque = BRISQUE(
        {"model_path": args.brisque_model, "range_path": args.brisque_range},
        cache_path=args.cache_path,
    )
    brisque.eval()

    score_sum = defaultdict(float)
    frame_count = defaultdict(int)
    dmos = {}

    with torch.no_grad():
        for index in tqdm(range(len(dataset)), desc="CLIP BRISQUE (all impaired videos)"):
            data, label, meta = dataset[index]
            valid = label["view_mask"].bool()
            if not valid.any():
                continue
            viewport_scores = brisque.score(data["views"][valid])
            weights = label["view_weights"][valid]

            denominator = float(weights.sum().item())
            if denominator > 0:
                frame_score = float((viewport_scores * weights).sum().item() / denominator)
            else:
                frame_score = float(viewport_scores.mean().item())

            video_id = meta["video_id"]
            score_sum[video_id] += frame_score
            frame_count[video_id] += 1
            dmos[video_id] = float(label["dmos"].item())

    rows = []
    for video_id in sorted(score_sum):
        score = score_sum[video_id] / frame_count[video_id]
        if math.isfinite(score) and math.isfinite(dmos[video_id]):
            rows.append({
                "video_id": video_id,
                "brisque": score,
                "dmos": dmos[video_id],
                "frame_count": frame_count[video_id],
            })
    return rows


def logistic4(x, beta1, beta2, beta3, beta4):
    scale = np.abs(beta4) + 1e-8
    return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / scale))


def logistic5(x, beta1, beta2, beta3, beta4, beta5):
    scale = np.abs(beta4) + 1e-8
    return beta2 + (beta1 - beta2) / (1.0 + np.exp(-(x - beta3) / scale)) ** beta5


def fit_logistic_predictions(raw_scores, dmos_scores, parameter_count):
    scale = raw_scores.std() if raw_scores.std() > 0 else 1.0
    if parameter_count == 4:
        function = logistic4
        initial = [dmos_scores.max(), dmos_scores.min(), raw_scores.mean(), scale]
        maxfev = 20000
    else:
        function = logistic5
        initial = [dmos_scores.max(), dmos_scores.min(), raw_scores.mean(), scale, 1.0]
        maxfev = 30000
    parameters, _ = curve_fit(function, raw_scores, dmos_scores, p0=initial, maxfev=maxfev)
    return function(raw_scores, *parameters)


def save_csv(rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video_id", "brisque_raw", "brisque_mapped4", "brisque_mapped5",
                        "dmos", "frame_count"),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(predictions, dmos_scores, plcc, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
    ax.scatter(predictions, dmos_scores, s=34, color="#087f8c", edgecolor="white",
               linewidth=0.6, alpha=0.85)
    if np.std(predictions) > 0:
        slope, intercept = np.polyfit(predictions, dmos_scores, 1)
        x_line = np.linspace(predictions.min(), predictions.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="#d1495b", linewidth=2.0)
    ax.set_title(f"CLIP Viewports: Mapped BRISQUE vs DMOS (PLCC = {plcc:.4f})")
    ax.set_xlabel("4-parameter logistic-mapped BRISQUE prediction")
    ax.set_ylabel("DMOS")
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP-selected viewports for all 180 impaired ERP videos and plot PLCC."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=45)
    parser.add_argument("--saliency-width", type=int, default=448)
    parser.add_argument("--saliency-height", type=int, default=224)
    parser.add_argument("--frame-width", type=int, default=2048)
    parser.add_argument("--frame-height", type=int, default=1024)
    parser.add_argument("--cluster-iters", type=int, default=10)
    parser.add_argument("--cluster-min-weight", type=float, default=0.5)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--dmos-list", type=Path, default=project_root / "data/VQA_ODV/train_dmos.txt")
    parser.add_argument("--cache-path", type=Path, default=project_root / "data/cache")
    parser.add_argument("--clip-checkpoint", type=Path,
                        default=project_root / "data/log/2026_01_23_05_46_31_yodaczsbvw_")
    parser.add_argument("--clip-checkpoint-name", default="04_ckpt")
    parser.add_argument("--brisque-model", type=Path,
                        default=project_root / "data/brisque/brisque_model_live.yml")
    parser.add_argument("--brisque-range", type=Path,
                        default=project_root / "data/brisque/brisque_range_live.yml")
    parser.add_argument("--output", type=Path,
                        default=project_root / "data/clip_brisque_dmos_plcc_all180.png")
    parser.add_argument("--csv-output", type=Path,
                        default=project_root / "data/clip_brisque_dmos_scores_all180.csv")
    args = parser.parse_args()

    rows = evaluate_all_videos(args, project_root)
    if len(rows) < 2:
        raise RuntimeError("At least two evaluated videos are required to calculate PLCC.")

    brisque_scores = np.asarray([row["brisque"] for row in rows], dtype=np.float64)
    dmos_scores = np.asarray([row["dmos"] for row in rows], dtype=np.float64)
    raw_plcc = float(pearsonr(brisque_scores, dmos_scores).statistic)
    mapped4 = fit_logistic_predictions(brisque_scores, dmos_scores, 4)
    mapped5 = fit_logistic_predictions(brisque_scores, dmos_scores, 5)
    mapped4_plcc = float(pearsonr(mapped4, dmos_scores).statistic)
    mapped5_plcc = float(pearsonr(mapped5, dmos_scores).statistic)

    for row, raw, prediction4, prediction5 in zip(rows, brisque_scores, mapped4, mapped5):
        row["brisque_raw"] = float(raw)
        row["brisque_mapped4"] = float(prediction4)
        row["brisque_mapped5"] = float(prediction5)
        del row["brisque"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(rows, args.csv_output)
    save_plot(mapped4, dmos_scores, mapped4_plcc, args.output)

    print(f"videos={len(rows)} frames={sum(row['frame_count'] for row in rows)}")
    print(f"PLCC_RAW={raw_plcc:.6f}")
    print(f"PLCC_MAPPED4={mapped4_plcc:.6f}")
    print(f"PLCC_MAPPED5={mapped5_plcc:.6f}")
    print(f"plot={args.output}")
    print(f"scores={args.csv_output}")


if __name__ == "__main__":
    main()
