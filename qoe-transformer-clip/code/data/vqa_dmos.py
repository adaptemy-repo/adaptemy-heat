import logging
import multiprocessing as mp
from pathlib import Path

import ffmpeg
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from munch import Munch
from torch.utils.data import Dataset

from exp import ex
from ckpt import load_ckpt_adapt
from model import get_pos_neg_extractors, get_model
from .utils import save_by_segment, load_by_segment, serial_save_by_segment, serial_load_by_segment


class VQA_DMOS(Dataset):
    @ex.capture()
    def __init__(self, data_path, cache_path, rebuild_cache, num_workers, clip_length, extractor_name, feature_norm,
                 frame_stride, model_config_pos, model_config_neg, sta_config, list_path, mode, device,
                 fixed_width=448, fixed_height=224):
        super().__init__()

        self.name = "VQA_DMOS"
        self.split = mode
        self.logger = logging.getLogger(__name__)

        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.rebuild_cache = rebuild_cache
        self.num_workers = num_workers
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.feature_norm = feature_norm
        self.extractor_name = extractor_name
        self.fixed_width = int(fixed_width)
        self.fixed_height = int(fixed_height)

        self.model_config_pos = Munch(model_config_pos)
        self.model_config_neg = Munch(model_config_neg)
        self.sta_config = Munch(sta_config)
        self.input_resolution = self.model_config_pos.input_resolution
        self.input_type = self.model_config_pos.input_type
        self.device = torch.device(device)

        self.feature_extractor_pos = None
        self.feature_extractor_neg = None
        self.sta_model = None

        list_path = Path(list_path)
        if not list_path.is_absolute():
            list_path = self.data_path / list_path

        self.video_list = self._load_list(list_path, mode)
        self.items = self.get_video()

    def _init_extractors_and_model(self):
        if self.feature_extractor_pos is not None:
            return

        extractors = get_pos_neg_extractors(
            extractor_name=self.extractor_name,
            model_config_pos=self.model_config_pos,
            model_config_neg=self.model_config_neg,
            cache_path=self.cache_path,
        )
        self.feature_extractor_pos = extractors["pos_extractor"].to(self.device)
        self.feature_extractor_neg = extractors["neg_extractor"].to(self.device)
        self.feature_extractor_pos.eval()
        self.feature_extractor_neg.eval()

        self.sta_model = get_model(
            model_name=self.sta_config.model_name,
            model_config=self.sta_config.model_config,
            cache_path=self.cache_path,
        )
        self.sta_model = load_ckpt_adapt(
            model=self.sta_model,
            ckpt_name=self.sta_config.ckpt_name,
            ckpt_path=Path(self.sta_config.ckpt_path),
            model_config=self.sta_config.model_config,
            cache_path=self.cache_path,
        ).to(self.device)
        self.sta_model.eval()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        data = {
            "frame": item["frame"],
            "salimap": item["salimap"],
        }

        label = {
            "dmos": item["dmos"],
        }

        if item["frame"].dim() == 4:
            mask = torch.where(torch.norm(item["frame"], dim=(-2, -1)).sum(-1) > 1e-5, 1.0, 0.0)
            label["mask"] = mask
        elif item["frame"].dim() == 3:
            mask = torch.where(torch.norm(item["frame"], dim=(-2, -1)).sum(-1) > 1e-5, 1.0, 0.0)
            label["mask"] = mask

        meta = {
            "width": item["width"],
            "height": item["height"],
            "video_id": item["video_id"],
            "segment_id": item["segment_id"],
        }

        return data, label, meta

    def _load_list(self, list_path, mode):
        entries = []
        if not list_path.exists():
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Expected 'video_path dmos split' per line, got: {line}")
                video_path, dmos_str, split_str = parts[0], parts[1], parts[2]
                split_flag = int(split_str)
                if mode == "train" and split_flag != 1:
                    continue
                if mode == "test" and split_flag != 0:
                    continue
                video_path = Path(video_path)
                if not video_path.exists():
                    self.logger.warning("Skipping missing video path: %s", video_path)
                    continue
                entries.append({
                    "path": video_path,
                    "dmos": float(dmos_str),
                })
        return entries

    def get_video(self):

        cache_file = (
            f"{self.name}_n{len(self.video_list)}_{self.split}_f{self.clip_length}_s{self.frame_stride}_"
            f"fx{self.fixed_width}x{self.fixed_height}_"
            f"r{self.input_resolution}_{self.input_type}_"
            f"{self.extractor_name}:{self.model_config_pos.train_type}|{self.model_config_neg.train_type}_"
            f"{self.sta_config.model_name}.pkl"
        )
        cache_file = self.cache_path / cache_file

        if self.rebuild_cache and len(list(cache_file.parent.glob(f"{cache_file.stem}*"))) > 0:
            for part_file in cache_file.parent.glob(f"{cache_file.stem}*"):
                part_file.unlink()

        if len(list(cache_file.parent.glob(f"{cache_file.stem}*"))) > 0:
            print(f"[VQA_DMOS] Loading cached data from {cache_file}, skipping extraction.")
            if self.input_resolution >= 224:
                items = serial_load_by_segment(load_dir=cache_file)
            else:
                items = load_by_segment(load_dir=cache_file)
            return items

        items = []
        if self.num_workers and self.num_workers > 1:
            pool = mp.Pool(self.num_workers)
            loader = pool.imap_unordered(self._load_video_clips, self.video_list)
            results = []
            for result in tqdm(loader, total=len(self.video_list), desc="VQA-DMOS VID"):
                if result is None:
                    continue
                results.append(result)
            pool.close()
            pool.join()

            # Initialize CUDA models after pool work to avoid pickling CUDA tensors.
            self._init_extractors_and_model()
            for result in results:
                items.extend(self._build_items_for_video(result))
        else:
            self._init_extractors_and_model()
            for entry in tqdm(self.video_list, desc="VQA-DMOS VID"):
                result = self._load_video_clips(entry)
                if result is None:
                    continue
                items.extend(self._build_items_for_video(result))

        if self.input_resolution >= 224:
            serial_save_by_segment(data=items, save_dir=cache_file)
        else:
            save_by_segment(data=items, save_dir=cache_file)

        return items

    def _load_video_clips(self, entry):
        video_id, width, height, fps, total_frames = self._get_video_info(entry["path"])
        if total_frames <= 0:
            return None

        clips = []
        for center_idx in range(22, total_frames, self.frame_stride):
            clip = self._get_centered_clip(
                entry["path"], width, height, fps, total_frames, center_idx
            )
            if clip is None:
                continue
            clips.append((center_idx, clip))

        if not clips:
            return None

        return {
            "video_id": video_id,
            "dmos": entry["dmos"],
            "clips": clips,
        }

    def _build_items_for_video(self, result):
        dmos_tensor = torch.tensor(result["dmos"], dtype=torch.float32)
        video_frames = []
        salimap_sum = None

        for center_idx, clip in result["clips"]:
            heatmap = self._compute_heatmap(clip, self.fixed_width, self.fixed_height)
            frame_idx = clip.size(0) // 2
            frame = clip[frame_idx]
            heatmap = heatmap.squeeze(0)

            video_frames.append({
                "video_id": result["video_id"],
                "segment_id": center_idx,
                "frame": frame,
                "salimap": heatmap,
                "width": self.fixed_width,
                "height": self.fixed_height,
                "dmos": dmos_tensor,
            })
            if salimap_sum is None:
                salimap_sum = heatmap.clone()
            else:
                salimap_sum = salimap_sum + heatmap

        for entry_item in video_frames:
            denom = (salimap_sum - entry_item["salimap"]).clamp_min(1e-6)
            entry_item["weight"] = entry_item["salimap"] / denom

        return video_frames

    def _prepare_extractor_input(self, clip):
        if self.input_type == "pano":
            target_h = self.input_resolution
            target_w = self.input_resolution * 2
            return F.interpolate(clip, size=(target_h, target_w), mode="bicubic", align_corners=False)

        _, _, h, w = clip.shape
        width = w
        height = h
        width = self.input_resolution if width < height else self.input_resolution * width // height
        height = self.input_resolution if width >= height else self.input_resolution * height // width
        resized = F.interpolate(clip, size=(height, width), mode="bicubic", align_corners=False)
        x = (width - self.input_resolution) // 2
        y = (height - self.input_resolution) // 2
        return resized[:, :, y:y + self.input_resolution, x:x + self.input_resolution]

    def _compute_heatmap(self, clip, width, height):
        clip_for_extractor = self._prepare_extractor_input(clip)
        if self.feature_norm:
            clip_for_extractor = ((clip_for_extractor / 255.0) - 0.5) / 0.5

        with torch.no_grad():
            video_cuda = clip_for_extractor.to(self.device, non_blocking=True)
            frame_pos = self.feature_extractor_pos(video_cuda).detach().cpu()
            frame_neg = self.feature_extractor_neg(video_cuda).detach().cpu()

        if len(frame_pos.size()) == 3:
            cls_pos = frame_pos[:, 0]
            cls_neg = frame_neg[:, 0]
            frame_pos = frame_pos[:, 1:]
            frame_neg = frame_neg[:, 1:]
        else:
            cls_pos = frame_pos[:, 0]
            cls_neg = frame_neg[:, 0]

        mask = torch.where(torch.norm(frame_pos, dim=(-2, -1)) > 1e-6, 1.0, 0.0)
        if len(frame_pos.size()) > 3:
            mask = torch.where(torch.norm(frame_pos, dim=(-2, -1)).sum(-1) > 1e-5, 1.0, 0.0)

        data = {
            "frame_pos": frame_pos.unsqueeze(0).to(self.device),
            "frame_neg": frame_neg.unsqueeze(0).to(self.device),
            "cls_pos": cls_pos.unsqueeze(0).to(self.device),
            "cls_neg": cls_neg.unsqueeze(0).to(self.device),
        }
        label = {"mask": mask.unsqueeze(0).to(self.device)}

        with torch.no_grad():
            result = self.sta_model(data, label)
            heatmap = self.sta_model.compute_heatmap(result["output"].contiguous())

        center_idx = clip_for_extractor.size(0) // 2
        heatmap = heatmap.squeeze(0)[center_idx:center_idx + 1].unsqueeze(1)
        heatmap = F.interpolate(heatmap, size=(height, width), mode="bicubic", align_corners=False)
        return heatmap

    def _get_video_info(self, vid):
        vid = Path(vid)
        video_id = vid.stem

        probe = ffmpeg.probe(vid)
        video_stream = next((stream for stream in probe["streams"]
                             if stream["codec_type"] == "video"), None)
        orig_width = int(video_stream["width"])
        orig_height = int(video_stream["height"])

        fps_str = video_stream.get("avg_frame_rate", "0/0")
        if fps_str == "0/0":
            fps_str = video_stream.get("r_frame_rate", "0/0")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0

        if "nb_frames" in video_stream:
            total_frames = int(video_stream["nb_frames"])
        else:
            duration = float(video_stream.get("duration", 0.0))
            total_frames = int(round(duration * fps)) if fps > 0 else 0

        return video_id, orig_width, orig_height, fps, total_frames

    def _get_centered_clip(self, vid, width, height, fps, total_frames, center_idx):
        if fps <= 0:
            return None

        half = self.clip_length // 2
        start_frame = center_idx - half
        end_frame = start_frame + self.clip_length
        read_start = max(start_frame, 0)
        read_end = min(end_frame, total_frames)
        if read_end <= read_start:
            return None

        start_time = read_start / fps
        duration = (read_end - read_start) / fps
        stream = ffmpeg.input(str(vid), ss=start_time, t=duration)
        if self.fixed_width > 0 and self.fixed_height > 0:
            stream = stream.filter("scale", self.fixed_width, self.fixed_height)
            out_width = self.fixed_width
            out_height = self.fixed_height
        else:
            out_width = width
            out_height = height
        out, _ = (
            stream.output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8)
        if video.size == 0:
            return None
        video = video.reshape([-1, out_height, out_width, 3])
        video = torch.from_numpy(video.astype("float32")).permute(0, 3, 1, 2)

        pad_front = max(0, -start_frame)
        pad_back = max(0, end_frame - total_frames)
        if pad_front > 0 or pad_back > 0:
            pad_frame = torch.zeros_like(video[0])
            if pad_front > 0:
                front = pad_frame.unsqueeze(0).repeat(pad_front, 1, 1, 1)
                video = torch.cat((front, video), 0)
            if pad_back > 0:
                back = pad_frame.unsqueeze(0).repeat(pad_back, 1, 1, 1)
                video = torch.cat((video, back), 0)

        if video.size(0) != self.clip_length:
            return None
        return video
