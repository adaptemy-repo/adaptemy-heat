import logging
import math
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from munch import Munch
from torch.utils.data import Dataset
from tqdm import tqdm

from exp import ex
from ckpt import load_ckpt_adapt
from model import get_pos_neg_extractors, get_model
from .utils import save_by_segment, load_by_segment, serial_save_by_segment, serial_load_by_segment


@lru_cache(maxsize=4)
def _viewport2sph_coord(port_w, port_h, fov_x, fov_y):
    u_mesh, v_mesh = np.meshgrid(range(port_w), range(port_h))
    u_mesh, v_mesh = u_mesh.flatten(), v_mesh.flatten()

    u_mesh = u_mesh.astype(np.float64) + 0.5
    v_mesh = v_mesh.astype(np.float64) + 0.5

    fov_x_rad = math.pi * fov_x / 180.0
    fov_y_rad = math.pi * fov_y / 180.0
    fx = port_w / (2.0 * math.tan(fov_x_rad / 2.0))
    fy = port_h / (2.0 * math.tan(fov_y_rad / 2.0))

    k_mat = np.asmatrix([[fx, 0.0, port_w / 2.0], [0.0, -fy, port_h / 2.0], [0.0, 0.0, 1.0]])
    e = np.asmatrix([u_mesh, v_mesh, np.ones_like(u_mesh)])
    q = k_mat.I * e
    q_normed = q / np.linalg.norm(q, axis=0, keepdims=True)
    p = np.diag([1.0, 1.0, -1.0]) * q_normed
    return np.asarray(p, dtype=np.float32)


def _cal_alignment_grid(viewport_resolution, lat, lon, p_sph):
    viewport_num = lat.shape[0]
    phi = lat * math.pi / 180.0
    tht = -lon * math.pi / 180.0

    rot = torch.stack(
        (
            torch.stack((torch.cos(tht), torch.sin(tht) * torch.sin(phi), torch.sin(tht) * torch.cos(phi))),
            torch.stack((torch.zeros_like(phi), torch.cos(phi), -torch.sin(phi))),
            torch.stack((-torch.sin(tht), torch.cos(tht) * torch.sin(phi), torch.cos(tht) * torch.cos(phi))),
        )
    )

    p_sph = p_sph.to(rot)
    e = torch.matmul(rot.permute(0, 2, 1), p_sph)

    lat = 90.0 - torch.acos(e[1, :]) * 180.0 / math.pi
    lon = torch.atan2(e[0, :], -e[2, :]) * 180.0 / math.pi
    lat = lat.view((viewport_num, *viewport_resolution))
    lon = lon.view((viewport_num, *viewport_resolution))

    pix_height = -lat / 90.0
    pix_width = lon / 180.0
    grid = torch.stack((pix_width, pix_height))
    grid = grid.permute(1, 2, 3, 0).to(torch.float)
    return grid


def _viewport_alignment(img, p_lat, t_lon, viewport_resolution, fov_x, fov_y):
    viewport_num = p_lat.shape[0]
    port_h, port_w = viewport_resolution
    p_sph = torch.tensor(_viewport2sph_coord(port_w, port_h, fov_x, fov_y))
    grid = _cal_alignment_grid(viewport_resolution, p_lat, t_lon, p_sph)
    viewport = F.grid_sample(img.expand(viewport_num, -1, -1, -1), grid)
    return viewport


class VQA_DMOS_VIEWS(Dataset):
    @ex.capture()
    def __init__(
        self,
        data_path,
        cache_path,
        rebuild_cache,
        num_workers,
        clip_length,
        extractor_name,
        feature_norm,
        frame_stride,
        model_config_pos,
        model_config_neg,
        sta_config,
        list_path,
        mode,
        device,
        saliency_width,
        saliency_height,
        frame_width,
        frame_height,
        viewport_count=8,
        viewport_fov_x=90.0,
        viewport_fov_y=90.0,
        viewport_height=600,
        viewport_width=540,
        preselect_k=512,
        nms_threshold_deg=15
    ):
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
        self.saliency_width = int(saliency_width)
        self.saliency_height = int(saliency_height)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)

        self.viewport_count = int(viewport_count)
        self.viewport_fov_x = float(viewport_fov_x)
        self.viewport_fov_y = float(viewport_fov_y)
        self.viewport_resolution = (int(viewport_height), int(viewport_width))
        self.preselect_k = int(preselect_k)
        self.nms_threshold_deg = float(nms_threshold_deg)

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
        viewports, view_centers, view_scores, view_mask, sal_viewports, view_weights = self._extract_viewports(
            item["frame"], item["salimap"]
        )

        data = {
            "views": viewports,
            "sal_views": sal_viewports,
            "frame": item["frame"],
            "salimap": item["salimap"],
        }

        label = {
            "dmos": item["dmos"],
            "view_mask": view_mask,
            "view_weights": view_weights,
        }

        meta = {
            "width": item["width"],
            "height": item["height"],
            "video_id": item["video_id"],
            "segment_id": item["segment_id"],
            "view_centers": view_centers,
            "view_scores": view_scores,
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
            f"sx{self.saliency_width}x{self.saliency_height}_"
            f"frx{self.frame_width}x{self.frame_height}_"
            f"r{self.input_resolution}_{self.input_type}_"
            f"v{self.viewport_count}_fovx{self.viewport_fov_x}_fovy{self.viewport_fov_y}_"
            f"vh{self.viewport_resolution[0]}x{self.viewport_resolution[1]}_"
            f"{self.extractor_name}:{self.model_config_pos.train_type}|{self.model_config_neg.train_type}_"
            f"{self.sta_config.model_name}.pkl"
        )
        cache_file = self.cache_path / cache_file

        if self.rebuild_cache and len(list(cache_file.parent.glob(f"{cache_file.stem}*"))) > 0:
            for part_file in cache_file.parent.glob(f"{cache_file.stem}*"):
                part_file.unlink()

        if len(list(cache_file.parent.glob(f"{cache_file.stem}*"))) > 0:
            print(f"[{self.name}] Loading cached data from {cache_file}, skipping extraction.")
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
            for result in tqdm(loader, total=len(self.video_list), desc=f"{self.name} VID"):
                if result is None:
                    continue
                results.append(result)
            pool.close()
            pool.join()

            self._init_extractors_and_model()
            for result in results:
                items.extend(self._build_items_for_video(result))
        else:
            self._init_extractors_and_model()
            for entry in tqdm(self.video_list, desc=f"{self.name} VID"):
                result = self._load_video_clips(entry)
                if result is None:
                    continue
                items.extend(self._build_items_for_video(result))

        if self.input_resolution >= 224:
            serial_save_by_segment(data=items, save_dir=cache_file)
        else:
            save_by_segment(data=items, save_dir=cache_file)

        return items

    def _resolve_saliency_size(self):
        if self.saliency_width > 0 and self.saliency_height > 0:
            return self.saliency_width, self.saliency_height
        return 0, 0

    def _resolve_frame_size(self):
        if self.frame_width > 0 and self.frame_height > 0:
            return self.frame_width, self.frame_height
        return 0, 0

    def _load_video_clips(self, entry):
        video_id, width, height, fps, total_frames = self._get_video_info(entry["path"])
        if total_frames <= 0:
            return None

        clip_width, clip_height = self._resolve_saliency_size()
        frame_width, frame_height = self._resolve_frame_size()

        clips = []
        for center_idx in range(22, total_frames, self.frame_stride):
            clip = self._get_centered_clip(
                entry["path"], width, height, fps, total_frames, center_idx,
                target_width=clip_width, target_height=clip_height,
            )
            if clip is None:
                continue
            frame = self._get_center_frame(
                entry["path"], width, height, fps, total_frames, center_idx,
                target_width=frame_width, target_height=frame_height,
            )
            if frame is None:
                continue
            clips.append({
                "center_idx": center_idx,
                "clip": clip,
                "frame": frame,
            })

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

        for clip_entry in result["clips"]:
            center_idx = clip_entry["center_idx"]
            clip = clip_entry["clip"]
            frame = clip_entry["frame"]
            height = int(frame.shape[1])
            width = int(frame.shape[2])
            heatmap = self._compute_heatmap(clip, width, height)
            heatmap = heatmap.squeeze(0)

            video_frames.append({
                "video_id": result["video_id"],
                "segment_id": center_idx,
                "frame": frame,
                "salimap": heatmap,
                "width": width,
                "height": height,
                "dmos": dmos_tensor,
            })

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

    def _get_centered_clip(self, vid, width, height, fps, total_frames, center_idx, target_width=0, target_height=0):
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
        if target_width > 0 and target_height > 0:
            stream = stream.filter("scale", target_width, target_height)
            out_width = target_width
            out_height = target_height
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

    def _get_center_frame(self, vid, width, height, fps, total_frames, center_idx, target_width=0, target_height=0):
        if fps <= 0:
            return None
        if center_idx < 0 or center_idx >= total_frames:
            return None

        center_time = center_idx / fps
        stream = ffmpeg.input(str(vid), ss=center_time)
        if target_width > 0 and target_height > 0:
            stream = stream.filter("scale", target_width, target_height)
            out_width = target_width
            out_height = target_height
        else:
            out_width = width
            out_height = height

        out, _ = (
            stream.output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1)
            .run(capture_stdout=True, quiet=True)
        )
        frame = np.frombuffer(out, np.uint8)
        if frame.size == 0:
            return None
        frame = frame.reshape([out_height, out_width, 3])
        frame = torch.from_numpy(frame.astype("float32")).permute(2, 0, 1)
        return frame

    def _select_view_centers(self, salimap):
        if salimap.dim() == 3:
            salimap = salimap.squeeze(0)
        if salimap.numel() == 0:
            return None, None

        h, w = salimap.shape
        flat = salimap.reshape(-1)
        k = min(self.preselect_k, flat.numel())
        values, idx = torch.topk(flat, k)
        ys = (idx // w).float()
        xs = (idx % w).float()
        lat = 90.0 - (ys + 0.5) * 180.0 / float(h)
        lon = (xs + 0.5) * 360.0 / float(w) - 180.0
        points = torch.stack([lat, lon], dim=1)

        if self.nms_threshold_deg <= 0:
            used = min(self.viewport_count, points.size(0))
            return points[:used], values[:used]

        return self._nms_on_sphere(points, values, self.viewport_count, self.nms_threshold_deg)

    def _nms_on_sphere(self, points, weights, proposal_count, threshold_deg):
        if points.numel() == 0:
            return points, weights

        idx = torch.argsort(weights, descending=True)
        selected_points = []
        selected_weights = []
        for i in idx:
            if len(selected_points) >= proposal_count:
                break
            candidate = points[i]
            if not selected_points:
                selected_points.append(candidate)
                selected_weights.append(weights[i])
                continue

            stack = torch.stack(selected_points, dim=0)
            distances = self._spherical_distance_deg(stack, candidate)
            if float(distances.min()) < threshold_deg:
                continue
            selected_points.append(candidate)
            selected_weights.append(weights[i])

        if not selected_points:
            return points[:0], weights[:0]
        return torch.stack(selected_points, dim=0), torch.stack(selected_weights, dim=0)

    def _spherical_distance_deg(self, points, ref):
        lat1 = points[:, 0] * math.pi / 180.0
        lon1 = points[:, 1] * math.pi / 180.0
        lat2 = ref[0] * math.pi / 180.0
        lon2 = ref[1] * math.pi / 180.0
        cos_val = torch.sin(lat1) * math.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lon1 - lon2)
        cos_val = torch.clamp(cos_val, -1.0, 1.0)
        return torch.acos(cos_val) * 180.0 / math.pi

    def _extract_viewports(self, frame, salimap):
        points, scores = self._select_view_centers(salimap)
        port_h, port_w = self.viewport_resolution
        view_count = self.viewport_count
        views = torch.zeros((view_count, frame.shape[0], port_h, port_w), dtype=frame.dtype)
        view_centers = torch.zeros((view_count, 2), dtype=salimap.dtype)
        view_scores = torch.zeros((view_count,), dtype=salimap.dtype)
        view_mask = torch.zeros((view_count,), dtype=salimap.dtype)
        sal_views = torch.zeros((view_count, 1, port_h, port_w), dtype=salimap.dtype)
        view_weights = torch.zeros((view_count,), dtype=salimap.dtype)

        if points is None or points.numel() == 0:
            return views, view_centers, view_scores, view_mask, sal_views, view_weights

        used = min(view_count, points.size(0))
        view_centers[:used] = points[:used]
        view_scores[:used] = scores[:used]
        view_mask[:used] = 1.0

        img = frame.unsqueeze(0)
        fov_x = self.viewport_fov_x
        fov_y = self.viewport_fov_y
        viewports = _viewport_alignment(
            img,
            points[:used, 0],
            points[:used, 1],
            viewport_resolution=self.viewport_resolution,
            fov_x=fov_x,
            fov_y=fov_y,
        )
        views[:used] = viewports
        if salimap.dim() == 3:
            salimap = salimap.squeeze(0)
        sal_img = salimap.unsqueeze(0).unsqueeze(0)
        sal_viewports = _viewport_alignment(
            sal_img,
            points[:used, 0],
            points[:used, 1],
            viewport_resolution=self.viewport_resolution,
            fov_x=fov_x,
            fov_y=fov_y,
        )
        sal_views[:used] = sal_viewports
        sal_sums = sal_viewports.view(used, -1).sum(dim=1)
        denom = sal_sums.sum().clamp_min(1e-6)
        view_weights[:used] = sal_sums / denom
        return views, view_centers, view_scores, view_mask, sal_views, view_weights
