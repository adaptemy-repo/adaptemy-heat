import json
import sys
import time
from pathlib import Path

import torch


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    start_time = time.perf_counter()
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from data.vqa_dmos import VQA_DMOS

    data_root = code_root.parent / "data"
    list_path = data_root / "VQA_ODV" / "train_dmos.txt"
    cache_path = data_root / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    extractor_cfg = _load_json(code_root / "configs" / "extractor" / "adaptvit.json")
    model_cfg = _load_json(code_root / "configs" / "model" / "stadecoder_single_clip.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = VQA_DMOS(
        data_path=data_root,
        cache_path=cache_path,
        rebuild_cache=False,
        num_workers=8,
        clip_length=5,
        extractor_name=extractor_cfg["extractor_name"],
        feature_norm=extractor_cfg["feature_norm"],
        frame_stride=extractor_cfg.get("frame_stride", 45),
        model_config_pos=extractor_cfg["model_config_pos"],
        model_config_neg=extractor_cfg["model_config_neg"],
        model_name=model_cfg["model_name"],
        model_config=model_cfg["model_config"],
        ckpt_name=None,
        ckpt_path=None,
        list_path=list_path,
        mode="train",
        device=device,
    )

    print(f"[VQA_DMOS] Dataset length: {len(ds)}")
    data, label, meta = ds[0]
    print(f"[VQA_DMOS] video shape: {data['video'].shape}")
    print(f"[VQA_DMOS] heatmap shape: {data['heatmap'].shape}")
    print(f"[VQA_DMOS] dmos: {label['dmos']}")
    print(f"[VQA_DMOS] segment_ids: {meta['segment_ids']}")
    elapsed_s = time.perf_counter() - start_time
    print(f"[VQA_DMOS] elapsed: {elapsed_s:.3f}s")


if __name__ == "__main__":
    main()
