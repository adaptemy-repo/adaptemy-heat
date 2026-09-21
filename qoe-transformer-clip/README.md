# Saliency-guided viewport quality assessment

This repository evaluates video quality using saliency-selected viewports and BRISQUE. Start in the project root, then change to `code/` as shown below.

## Setup

- Put the VQA-ODV videos and `train_dmos.txt` under `data/VQA_ODV/`. Paths in the list must point to readable videos; the loader also searches the local `train/` and `test/` video directories.
- Put the pretrained decoder checkpoints under `data/log/` and the OpenCV BRISQUE model/range files under `data/brisque/`.
- Install the project dependencies and `opencv-contrib-python` (for `cv2.quality`), and make sure `ffmpeg` and a CUDA-capable PyTorch installation are available. `environment.yml` is the original environment specification.

## Run experiments

From `code/`, Sacred commands can be run with `python cli.py <command> with key=value`. For example:

```bash
cd code
python cli.py demo_views_clip with num_workers=1
python cli.py demo_brisque with num_workers=1 brisque_model_path=data/brisque/brisque_model_live.yml brisque_range_path=data/brisque/brisque_range_live.yml
```

`demo_views_clip` writes CLIP-selected viewport images to `data/views_clip/`. It uses the frame/segment manifest in `data/views/`, so that directory must already contain reference `*_frame.png` images. `demo_brisque` runs the saliency-weighted BRISQUE evaluation.

For training, first adjust `code/config.py` and the JSON files in `code/configs/`, then run the relevant command from `code/`:

```bash
python cli.py train_adapt_single
```

To evaluate CLIP-selected viewports on all available impaired videos and create the video-level PLCC plot:

```bash
python plot_clip_brisque_plcc.py --num-workers 1
```


To compare the three QP versions within each source video using the saved CSV (no video, GPU, or saliency recomputation):

```bash
python plot_brisque_within_source.py
```

This writes `data/clip_brisque_within_source.svg`, `data/clip_brisque_within_source.csv`, and `data/clip_brisque_source_correlations.csv`. The within-source correlation is a diagnostic, not a replacement for all-video PLCC.

Saliency and decoded frames are cached under `data/cache/`. Subsequent runs reuse matching cache parts unless `--rebuild-cache` is passed to the standalone PLCC script or `rebuild_cache=True` is supplied to a Sacred command. Large video datasets can exhaust shared memory with multiple workers; `num_workers=1` is the safer starting point.
