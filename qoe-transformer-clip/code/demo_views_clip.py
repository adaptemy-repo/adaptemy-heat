import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from exp import ex
from data.vqa_dmos_views_cluster import VQA_DMOS_VIEWS_CLUSTER


def _save_rgb_tensor(image, path):
    image = image.permute(1, 2, 0).contiguous().cpu().numpy()
    image = image.clip(0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@ex.capture()
def demo_views_clip(device, list_path, data_path, cache_path, rebuild_cache,
                    num_workers, clip_length, extractor_name, feature_norm, frame_stride,
                    model_config_pos, model_config_neg, sta_config,
                    saliency_width=448, saliency_height=224,
                    frame_width=2048, frame_height=1024,
                    cluster_iters=10, cluster_min_weight=0.5, mode="test",
                    clip_ckpt_path=None, clip_ckpt_name="04_ckpt",
                    save_views_dir=None, reference_views_dir=None,
                    clean_output=True):
    project_root = Path(__file__).resolve().parents[1]
    clip_config_path = project_root / "code" / "configs" / "model" / "stadecoder_single_clip.json"
    with open(clip_config_path, "r", encoding="utf-8") as handle:
        clip_config = json.load(handle)

    if clip_ckpt_path is None:
        clip_ckpt_path = project_root / "data" / "log" / "2026_01_23_05_46_31_yodaczsbvw_"
    if save_views_dir is None:
        save_views_dir = project_root / "data" / "views_clip"
    if reference_views_dir is None:
        reference_views_dir = project_root / "data" / "views"
    if not Path(list_path).exists():
        relocated_list = project_root / "data" / "VQA_ODV" / "train_dmos.txt"
        if not relocated_list.exists():
            raise FileNotFoundError(f"Dataset list not found: {list_path}")
        list_path = relocated_list

    clip_sta_config = dict(sta_config)
    clip_sta_config.update(clip_config)
    clip_sta_config["model_config"]["feature_branch"] = "pos"
    clip_sta_config["ckpt_path"] = Path(clip_ckpt_path)
    clip_sta_config["ckpt_name"] = clip_ckpt_name

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
        sta_config=clip_sta_config,
        list_path=list_path,
        mode=mode,
        device=device,
        saliency_width=saliency_width,
        saliency_height=saliency_height,
        frame_width=frame_width,
        frame_height=frame_height,
        cluster_iters=cluster_iters,
        cluster_min_weight=cluster_min_weight,
        cache_tag="clip_pos",
    )

    output_dir = Path(save_views_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for old_image in output_dir.glob("*.png"):
            old_image.unlink()

    reference_dir = Path(reference_views_dir)
    reference_items = set()
    for frame_path in reference_dir.glob("*_frame.png"):
        stem = frame_path.stem
        video_id, separator, segment_text = stem.rpartition("_seg")
        if not separator or not segment_text.endswith("_frame"):
            continue
        reference_items.add((video_id, int(segment_text[:-6])))
    if not reference_items:
        raise FileNotFoundError(f"No reference frame images found in {reference_dir}")

    saved_frames = 0

    with torch.no_grad():
        for index in tqdm(range(len(dataset)), desc="Generate CLIP viewports"):
            data, label, meta = dataset[index]
            video_id = meta["video_id"]
            segment_id = int(meta["segment_id"])
            if (video_id, segment_id) not in reference_items:
                continue

            prefix = f"{video_id}_seg{segment_id}"
            _save_rgb_tensor(data["frame"], output_dir / f"{prefix}_frame.png")

            heatmap = data["salimap"].squeeze().cpu().numpy()
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / max(float(heatmap.max()), 1e-8)
            heatmap = cv2.applyColorMap((heatmap * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / f"{prefix}_heatmap.png"), heatmap)

            view_mask = label["view_mask"]
            for view_index, view in enumerate(data["views"]):
                if not bool(view_mask[view_index]):
                    continue
                _save_rgb_tensor(view, output_dir / f"{prefix}_v{view_index}.png")

            saved_frames += 1

    missing = len(reference_items) - saved_frames
    print(
        f"[demo_views_clip] saved {saved_frames}/{len(reference_items)} reference frames "
        f"to {output_dir}; missing={missing}"
    )
    return {"output_dir": str(output_dir), "saved_frames": saved_frames}
