import math

import torch

from .vqa_dmos_views import VQA_DMOS_VIEWS


def _latlon_to_xyz(lat, lon):
    lat_rad = lat * math.pi / 180.0
    lon_rad = lon * math.pi / 180.0
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.sin(lat_rad)
    z = -torch.cos(lat_rad) * torch.sin(lon_rad)
    return torch.stack([x, y, z], dim=-1)


def _xyz_to_latlon(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lat = torch.asin(torch.clamp(y, -1.0, 1.0)) * 180.0 / math.pi
    lon = torch.atan2(-z, x) * 180.0 / math.pi
    return torch.stack([lat, lon], dim=1)


def _weighted_kmeans_sphere(points_xyz, weights, k, iters=10):
    if points_xyz.numel() == 0 or k <= 0:
        return points_xyz[:0], weights[:0]

    k = min(k, points_xyz.shape[0])
    weights = weights.clamp_min(0.0)

    # Initialize with top-weighted points
    init_idx = torch.topk(weights, k).indices
    centers = points_xyz[init_idx].clone()

    for _ in range(iters):
        # Assign by maximum cosine similarity (dot on unit sphere)
        sim = torch.matmul(points_xyz, centers.t())
        assign = torch.argmax(sim, dim=1)

        new_centers = []
        for ci in range(k):
            mask = assign == ci
            if not mask.any():
                # Re-init empty cluster to highest-weight point
                new_centers.append(points_xyz[torch.argmax(weights)])
                continue
            w = weights[mask].unsqueeze(1)
            pts = points_xyz[mask]
            mean = (pts * w).sum(dim=0)
            norm = mean.norm().clamp_min(1e-6)
            new_centers.append(mean / norm)
        centers = torch.stack(new_centers, dim=0)

    # Final cluster weights: sum of assigned point weights
    sim = torch.matmul(points_xyz, centers.t())
    assign = torch.argmax(sim, dim=1)
    cluster_weights = []
    for ci in range(k):
        mask = assign == ci
        if not mask.any():
            cluster_weights.append(weights.new_tensor(0.0))
        else:
            cluster_weights.append(weights[mask].sum())
    cluster_weights = torch.stack(cluster_weights, dim=0)

    return centers, cluster_weights


def _merge_close_centers(points_latlon, weights, threshold_deg):
    if points_latlon.numel() == 0:
        return points_latlon, weights
    if threshold_deg <= 0:
        return points_latlon, weights

    idx = torch.argsort(weights, descending=True)
    used = torch.zeros(points_latlon.shape[0], dtype=torch.bool, device=points_latlon.device)
    merged_centers = []
    merged_weights = []

    for i in idx:
        if used[i]:
            continue
        seed = points_latlon[i]
        used[i] = True

        # Find all points within threshold
        dists = VQA_DMOS_VIEWS._spherical_distance_deg(VQA_DMOS_VIEWS, points_latlon, seed)
        close = (dists < threshold_deg) & (~used)
        group_idx = torch.cat([i.view(1), torch.nonzero(close, as_tuple=False).squeeze(1)])
        used[close] = True

        group_latlon = points_latlon[group_idx]
        group_weights = weights[group_idx].clamp_min(0.0)

        # Weighted average in XYZ then normalize
        group_xyz = _latlon_to_xyz(group_latlon[:, 0], group_latlon[:, 1])
        w = group_weights.unsqueeze(1)
        mean = (group_xyz * w).sum(dim=0)
        norm = mean.norm().clamp_min(1e-6)
        center_xyz = mean / norm
        merged_centers.append(_xyz_to_latlon(center_xyz.unsqueeze(0))[0])
        merged_weights.append(group_weights.sum())

    merged_centers = torch.stack(merged_centers, dim=0)
    merged_weights = torch.stack(merged_weights, dim=0)
    return merged_centers, merged_weights


class VQA_DMOS_VIEWS_CLUSTER(VQA_DMOS_VIEWS):
    def __init__(self, *args, cluster_iters=10, cluster_min_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_iters = int(cluster_iters)
        self.cluster_min_weight = float(cluster_min_weight)

    def _select_view_centers(self, salimap):
        if salimap.dim() == 3:
            salimap = salimap.squeeze(0)
        if salimap.numel() == 0:
            return None, None

        h, w = salimap.shape
        flat = salimap.reshape(-1)
        total = flat.numel()

        if self.cluster_min_weight > 0:
            mask = flat > self.cluster_min_weight
            if mask.any():
                values = flat[mask]
                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            else:
                idx = torch.arange(total, device=flat.device)
                values = flat
        else:
            idx = torch.arange(total, device=flat.device)
            values = flat


        ys = (idx // w).float()
        xs = (idx % w).float()
        lat = 90.0 - (ys + 0.5) * 180.0 / float(h)
        lon = (xs + 0.5) * 360.0 / float(w) - 180.0

        points_xyz = _latlon_to_xyz(lat, lon)
        centers_xyz, cluster_weights = _weighted_kmeans_sphere(
            points_xyz, values, self.viewport_count, iters=self.cluster_iters
        )
        if centers_xyz.numel() == 0:
            return points_xyz[:0], values[:0]

        centers = _xyz_to_latlon(centers_xyz)

        if self.nms_threshold_deg > 0:
            centers, cluster_weights = _merge_close_centers(
                centers, cluster_weights, self.nms_threshold_deg
            )
            if centers.shape[0] > self.viewport_count:
                centers, cluster_weights = self._nms_on_sphere(
                    centers, cluster_weights, self.viewport_count, self.nms_threshold_deg
                )
        return centers, cluster_weights
