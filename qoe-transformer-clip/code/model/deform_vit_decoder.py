import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp
from timm.models.vision_transformer import Block

from . import full_model
from .decoder import BaseDecoder, FusedSTBlock



@full_model
class CLSAdapter(nn.Module):
    """
        MLP adapter on top of a frozen vit image encoder.

        Input:  x  (B, dim)    vit image embedding
        Output: x_new   (B, dim)   adapted embedding (L2-normalized by default)
    """
    def __init__(self, model_config,cache_path):
        super().__init__()

        dim = model_config.num_features

        if model_config.cls_encoder_type == "mlp":
            self.cls_encoder = Mlp(in_features=2 * dim, out_features=dim)
        elif model_config.cls_encoder_type == "identity":
            self.cls_encoder = nn.Identity()
        else:
            raise NotImplementedError

        self.use_residual = model_config.use_residual
        self.normalize = model_config.normalize
    
    def forward(self, x1, x2):
        """
        x1, x2: (..., dim), same shape and same last-dim size.
        """
        # Concatenate along the feature dimension
        x_cat = torch.cat([x1, x2], dim=-1)  # (..., 2 * dim)

        x_delta = self.cls_encoder(x_cat)  # (..., dim)

        if self.use_residual:
            # Residual uses x1 as the base
            x_new = x1 + x_delta
        else:
            x_new = x_delta

        if self.normalize:
            x_new = F.normalize(x_new, dim=-1)

        return x_new


@full_model
class PatchAdapter(nn.Module):
    """
    Patch-level MLP adapter.

    Takes two patch embeddings with the same dimension (dim),
    concatenates them to size 2 * dim, and projects back to dim.

    Inputs:
        x1: (..., dim)  patch embedding
        x2: (..., dim)  patch embedding (same shape as x1)

    Output:
        x_new: (..., dim) adapted patch embedding (L2-normalized by default)
    """
    def __init__(self, model_config, cache_path):
        super().__init__()

        dim = model_config.num_features
        self.shortcut = nn.Linear(2 * dim, dim)  # W_s

        # MLP: 2 * dim  ->  dim
        self.patch_encoder = Mlp(
            in_features=2 * dim,
            out_features=dim
        )

        self.use_residual = model_config.use_residual
        self.normalize = model_config.normalize

    def forward(self, x1, x2):
        """
        x1, x2: (..., dim), same shape and same last-dim size.
        """
        # Concatenate along the feature dimension
        x_cat = torch.cat([x1, x2], dim=-1)  # (..., 2 * dim)

        x_delta = self.patch_encoder(x_cat)  # (..., dim)

        if self.use_residual:
            # Residual uses x_cat linear projection as the base
            res = self.shortcut(x_cat)
            x_new = res + x_delta
        else:
            x_new = x_delta

        if self.normalize:
            x_new = F.normalize(x_new, dim=-1)

        return x_new


class SingleAdapter(nn.Module):
    """
    Single-input adapter for CLS or patch embeddings.

    Input:  x (..., dim)
    Output: x_new (..., dim) adapted embedding (L2-normalized by default)
    """
    def __init__(self, model_config, encoder_type="mlp"):
        super().__init__()

        dim = model_config.num_features
        if encoder_type == "mlp":
            self.encoder = Mlp(in_features=dim, out_features=dim)
        elif encoder_type == "identity":
            self.encoder = nn.Identity()
        else:
            raise NotImplementedError

        self.use_residual = model_config.use_residual
        self.normalize = model_config.normalize

    def forward(self, x):
        x_delta = self.encoder(x)

        if self.use_residual:
            x_new = x + x_delta
        else:
            x_new = x_delta

        if self.normalize:
            x_new = F.normalize(x_new, dim=-1)

        return x_new


@full_model
class SingleAdaptDecoder(BaseDecoder):
    """
    AdaptDecoder variant that uses a single extractor.

    batch must contain either:
      - 'frame_pos' : BTNC or BNC (patch embeddings)
      - 'cls_pos'   : BTC  (CLS embeddings)
    or:
      - 'frame'     : BTNC or BNC
      - 'cls'       : BTC
    """
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)

        self.cls_adapter = SingleAdapter(model_config, encoder_type=model_config.cls_encoder_type)
        self.patch_adapter = SingleAdapter(model_config, encoder_type="mlp")

        self.attn_layer = Block(dim=model_config.num_features, num_heads=8)

        # Mfp related variables
        self.mask_prob = model_config.mask_prob

        self.lambda_cls = model_config.lambda_cls
        self.lambda_spatial_feat = model_config.lambda_spatial_feat
        self.lambda_temporal_feat = model_config.lambda_temporal_feat

        # Score related variables
        self.coeff_cls = model_config.coeff_cls
        self.coeff_time = model_config.coeff_time
        self.coeff_space = model_config.coeff_space

        # verbose mode for decoupling each score maps
        self.verbose_mode = model_config.verbose_mode

        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

        self.temp_score_weight = nn.Parameter(torch.ones(1))
        self.spat_score_weight = nn.Parameter(torch.ones(1))

    def forward(self, batch, label):
        frame = batch['frame_neg']
        cls_feat = batch['cls_neg']


        # Handle BNC -> B1NC (single time-step) case
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(1)  # BNC -> B1NC
            if 'mask' in label:
                label['mask'] = label['mask'].unsqueeze(1)
                if label['mask'] < 0.1:
                    return {'loss_total': 0.}



        # Main decoder
        predicted_features = self.attn_layer(frame)

        result = {}
        result['loss_total'] = 0.

        # Adapted CLS
        cls_adapt = self.cls_adapter.forward(cls_feat)  # B T C

        # cls / time / space scores (use adapted CLS)
        cls_score = self.mse_loss(cls_adapt.unsqueeze(-2), predicted_features).mean(-1)
        time_score = self.mse_loss(
            predicted_features.mean(-3).unsqueeze(-3), predicted_features
        ).mean(-1)
        space_score = self.mse_loss(
            predicted_features.mean(-2).unsqueeze(-2), predicted_features
        ).mean(-1)

        out = self.coeff_cls * cls_score + self.coeff_time * time_score + self.coeff_space * space_score

        result['cls_weight'] = self.mse_loss(cls_adapt, cls_feat).mean().item()

        if self.verbose_mode:
            result['output_cls'] = cls_score
            result['output_time'] = time_score
            result['output_space'] = space_score

        # CLS consistency loss
        if 'mask' in label.keys():
            mask = label['mask']  # B T
        else:
            mask = torch.ones(cls_adapt.shape[:2], device=cls_adapt.device)

        masked_cls = cls_adapt * mask.unsqueeze(-1)  # B T C
        cls_sum = masked_cls.sum(dim=1, keepdim=True)  # B 1 C
        denom = (mask.sum(dim=1, keepdim=True) + 1e-6).unsqueeze(-1)  # B 1 1
        cls_mean = cls_sum / denom  # B 1 C
        cls_mean = cls_mean.expand_as(cls_adapt)  # B T C

        loss_cls = self.mse_loss(cls_adapt, cls_mean).mean(-1)  # B T
        loss_cls = (loss_cls * mask).sum() / (mask.sum() + 1e-6)

        result['loss_total'] += loss_cls * self.lambda_cls
        result['loss_cls'] = loss_cls.detach().item()

        # spatial feature consistency
        neighbor_avg = torch.einsum(
            "BTNC,NN->BTNC", predicted_features, self.adjacency_matrix
        ).detach()
        loss_feat_s = self.mse_loss(
            predicted_features - neighbor_avg,
            torch.zeros_like(predicted_features)
        ).mean(-1)  # B T N

        if 'mask' in label.keys():
            loss_feat_s = loss_feat_s.mean(-1) * label['mask']
        loss_feat_s = loss_feat_s.mean()
        result['loss_total'] += loss_feat_s * self.lambda_spatial_feat
        result['loss_feat_spatial'] = loss_feat_s.detach().item()

        # temporal feature consistency (BTNC)
        temporal_avg = predicted_features.detach().clone()
        temporal_avg[:, 1:] = predicted_features.detach()[:, :-1]
        temporal_avg[:, 0] = 0.
        temporal_avg[:, :-1] += predicted_features.detach()[:, 1:]
        temporal_avg[:, 1:-1] /= 2.

        loss_feat_t = self.mse_loss(
            predicted_features, temporal_avg.detach()
        ).mean(-1)  # B T N
        if 'mask' in label.keys():
            loss_feat_t = loss_feat_t.mean(-1) * label['mask']
        loss_feat_t = loss_feat_t.mean()
        result['loss_total'] += loss_feat_t * self.lambda_temporal_feat
        result['loss_feat_temporal'] = loss_feat_t.detach().item()

        result['output'] = out.detach()

        return result


class AdaptDecoder(BaseDecoder):
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)

        # ---------------------------------------------------------------------
        # CLS adapter (for cls_pos / cls_neg)
        # ---------------------------------------------------------------------
        self.cls_adapter = CLSAdapter(model_config, cache_path)

        # ---------------------------------------------------------------------
        # Patch adapter (features fused)
        # ---------------------------------------------------------------------
        self.patch_adapter = PatchAdapter(model_config, cache_path)

        self.attn_layer = Block(dim=model_config.num_features, num_heads=8)

        # Mfp related variables
        self.mask_prob = model_config.mask_prob

        self.lambda_cls = model_config.lambda_cls
        self.lambda_spatial_feat = model_config.lambda_spatial_feat
        self.lambda_temporal_feat = model_config.lambda_temporal_feat

        # Score related variables
        self.coeff_cls = model_config.coeff_cls
        self.coeff_time = model_config.coeff_time
        self.coeff_space = model_config.coeff_space

        # verbose mode for decoupling each score maps
        self.verbose_mode = model_config.verbose_mode

        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

        self.temp_score_weight = nn.Parameter(torch.ones(1))
        self.spat_score_weight = nn.Parameter(torch.ones(1))

        # margin for triplet-style cls loss
        self.margin_cls = model_config.margin_cls

    def forward(self, batch, label):
        """
        batch must contain:
          - 'frame_pos' : BTNC or BNC (patch embeddings, positive)
          - 'frame_neg'  : BTNC or BNC (patch embeddings, negative)
          - 'cls_pos'   : BTC  (positive CLS, e.g. CLIP)
          - 'cls_neg'    : BTC  (negative CLS, e.g. ImageNet ViT)
        """

        # ---------------------------------------------------------------------
        # Patch features: combine frame_pos and frame_neg via PatchAdapter
        # ---------------------------------------------------------------------
        frame_pos = batch['frame_pos']  # shape B T N C or B N C
        frame_neg  = batch['frame_neg']   # same shape as frame_pos

        # Handle BNC -> B1NC (single time-step) case
        if len(frame_pos.shape) == 3:
            frame_pos = frame_pos.unsqueeze(1)  # BNC -> B1NC
            frame_neg  = frame_neg.unsqueeze(1)
            label['mask'] = label['mask'].unsqueeze(1)
            if label['mask'] < 0.1:
                return {'loss_total': 0.}

        # Adapted patch features (no triplet loss here, just combining)
        # PatchAdapter works on (..., C), so it’s agnostic to B/T/N layout.
        features = self.patch_adapter.forward(frame_pos, frame_neg)  # same shape as frame_pos

        # Main decoder
        predicted_features = self.attn_layer(features)

        result = {}
        result['loss_total'] = 0.

        # ---------------------------------------------------------------------
        # CLS embeddings: adapted via CLSAdapter
        # ---------------------------------------------------------------------
        cls_pos = batch['cls_pos']  # B T C
        cls_neg  = batch['cls_neg']   # B T C

        # adapted CLS (this is the anchor used in CLS loss)
        cls_adapt = self.cls_adapter.forward(cls_pos, cls_neg)  # B T C

        # cls / time / space scores (use adapted CLS)
        cls_score = self.mse_loss(cls_adapt.unsqueeze(-2), predicted_features).mean(-1)
        time_score = self.mse_loss(
            predicted_features.mean(-3).unsqueeze(-3), predicted_features
        ).mean(-1)
        space_score = self.mse_loss(
            predicted_features.mean(-2).unsqueeze(-2), predicted_features
        ).mean(-1)


        out = self.coeff_cls * (1/cls_score) + self.coeff_time * time_score + self.coeff_space * space_score


        # diagnostic: distance between adapted CLS and original cls_pos
        result['cls_weight_pos'] = self.mse_loss(cls_adapt, cls_pos).mean().item()
        result['cls_weight_neg'] = self.mse_loss(cls_adapt, cls_neg).mean().item()

        if self.verbose_mode:
            result['output_cls'] = cls_score
            result['output_time'] = time_score
            result['output_space'] = space_score

        # ---------------------------------------------------------------------
        # CLS loss: consistency + triplet (adapted pos to cls_pos, neg from cls_neg)
        # ---------------------------------------------------------------------
        if 'mask' in label.keys():
            mask = label['mask']  # B T
        else:
            mask = torch.ones(cls_adapt.shape[:2], device=cls_adapt.device)

        # per-video consistency using adapted CLS
        masked_cls = cls_adapt * mask.unsqueeze(-1)  # B T C
        cls_sum = masked_cls.sum(dim=1, keepdim=True)  # B 1 C
        denom = (mask.sum(dim=1, keepdim=True) + 1e-6).unsqueeze(-1)  # B 1 1
        cls_mean = cls_sum / denom  # B 1 C
        cls_mean = cls_mean.expand_as(cls_adapt)  # B T C

        loss_cls_cons = self.mse_loss(cls_adapt, cls_mean).mean(-1)  # B T
        loss_cls_cons = (loss_cls_cons * mask).sum() / (mask.sum() + 1e-6)

        # triplet-style term over frames: anchor=cls_adapt, pos=cls_pos, neg=cls_neg
        B, T, C = cls_adapt.shape
        anchor = cls_adapt.reshape(B * T, C)
        pos    = cls_pos.reshape(B * T, C)
        neg    = cls_neg.reshape(B * T, C)

        anchor = F.normalize(anchor, dim=-1)
        pos    = F.normalize(pos,    dim=-1)
        neg    = F.normalize(neg,    dim=-1)

        d_pos = (anchor - pos).pow(2).sum(dim=-1)  # (B*T,)
        d_neg = (anchor - neg).pow(2).sum(dim=-1)  # (B*T,)

        triplet = F.relu(d_pos - d_neg + self.margin_cls)  # (B*T,)


        mask_flat = mask.reshape(B * T)
        loss_cls_triplet = (triplet * mask_flat).sum() / (mask_flat.sum() + 1e-6)

        loss_cls = loss_cls_cons + loss_cls_triplet

        result['loss_total'] += loss_cls * self.lambda_cls
        result['loss_cls'] = loss_cls.detach().item()
        result['loss_cls_cons'] = loss_cls_cons.detach().item()
        result['loss_cls_triplet'] = loss_cls_triplet.detach().item()

        # ---------------------------------------------------------------------
        # spatial feature consistency
        # ---------------------------------------------------------------------
        neighbor_avg = torch.einsum(
            "BTNC,NN->BTNC", predicted_features, self.adjacency_matrix
        ).detach()
        loss_feat_s = self.mse_loss(
            predicted_features - neighbor_avg,
            torch.zeros_like(predicted_features)
        ).mean(-1)  # B T N

        if 'mask' in label.keys():
            loss_feat_s = loss_feat_s.mean(-1) * label['mask']
        loss_feat_s = loss_feat_s.mean()
        result['loss_total'] += loss_feat_s * self.lambda_spatial_feat
        result['loss_feat_spatial'] = loss_feat_s.detach().item()

        # ---------------------------------------------------------------------
        # temporal feature consistency (BTNC)
        # ---------------------------------------------------------------------
        temporal_avg = predicted_features.detach().clone()
        temporal_avg[:, 1:] = predicted_features.detach()[:, :-1]
        temporal_avg[:, 0] = 0.
        temporal_avg[:, :-1] += predicted_features.detach()[:, 1:]
        temporal_avg[:, 1:-1] /= 2.

        loss_feat_t = self.mse_loss(
            predicted_features, temporal_avg.detach()
        ).mean(-1)  # B T N
        if 'mask' in label.keys():
            loss_feat_t = loss_feat_t.mean(-1) * label['mask']
        loss_feat_t = loss_feat_t.mean()
        result['loss_total'] += loss_feat_t * self.lambda_temporal_feat
        result['loss_feat_temporal'] = loss_feat_t.detach().item()

        result['output'] = out.detach()

        return result



class ReconAdaptDecoder(BaseDecoder):
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)

        self.cls_adapter = CLSAdapter(model_config, cache_path)
        self.patch_adapter = PatchAdapter(model_config, cache_path)
        self.attn_layer = Block(dim=model_config.num_features, num_heads=8)

        self.mask_prob = model_config.mask_prob

        self.lambda_spatial_feat = model_config.lambda_spatial_feat
        self.lambda_temporal_feat = model_config.lambda_temporal_feat
        self.lambda_cls = model_config.lambda_cls

        self.coeff_cls = model_config.coeff_cls
        self.coeff_time = model_config.coeff_time
        self.coeff_space = model_config.coeff_space

        self.verbose_mode = model_config.verbose_mode

        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

        self.temp_score_weight = nn.Parameter(torch.ones(1))
        self.spat_score_weight = nn.Parameter(torch.ones(1))

        self.margin_cls = model_config.margin_cls

        # learned patch scorer for simplex weights
        hidden = max(1, model_config.num_features // 2)
        self.patch_score = nn.Sequential(
            nn.LayerNorm(model_config.num_features),
            nn.Linear(model_config.num_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch, label):
        frame_pos = batch['frame_pos']
        frame_neg = batch['frame_neg']

        if len(frame_pos.shape) == 3:
            frame_pos = frame_pos.unsqueeze(1)
            frame_neg = frame_neg.unsqueeze(1)
            label['mask'] = label['mask'].unsqueeze(1)
            if label['mask'] < 0.1:
                return {'loss_total': 0.}

        features = self.patch_adapter.forward(frame_pos, frame_neg)
        predicted_features = self.attn_layer(features)

        result = {}
        result['loss_total'] = 0.

        cls_pos = batch['cls_pos']
        cls_neg = batch['cls_neg']

        # Patch-only learned scoring for reconstruction
        scores = self.patch_score(predicted_features).squeeze(-1)  # B T N
        weights = scores.softmax(dim=-1)
        cls_recon = (weights.unsqueeze(-1) * predicted_features).sum(-2)  # B T C

        # Scores for heatmap
        cls_score = self.mse_loss(cls_recon.unsqueeze(-2), predicted_features).mean(-1)
        time_score = self.mse_loss(
            predicted_features.mean(-3).unsqueeze(-3), predicted_features
        ).mean(-1)
        space_score = self.mse_loss(
            predicted_features.mean(-2).unsqueeze(-2), predicted_features
        ).mean(-1)

        # out = (
        #     self.coeff_cls * cls_score
        #     + self.coeff_time * time_score
        #     + self.coeff_space * space_score
        # )

        out = weights

        if self.verbose_mode:
            result['output_cls'] = cls_score
            result['output_time'] = time_score
            result['output_space'] = space_score

        if 'mask' in label.keys():
            mask = label['mask']
        else:
            mask = torch.ones(cls_pos.shape[:2], device=cls_pos.device)

        # Triplet loss with reconstructed CLS as anchor
        B, T, C = cls_recon.shape
        anchor = cls_recon.reshape(B * T, C)
        pos = cls_pos.reshape(B * T, C)
        neg = cls_neg.reshape(B * T, C)

        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        d_pos = (anchor - pos).pow(2).sum(dim=-1)
        d_neg = (anchor - neg).pow(2).sum(dim=-1)
        triplet = F.relu(d_pos - d_neg + self.margin_cls)

        mask_flat = mask.reshape(B * T)
        loss_cls_recon_triplet = (triplet * mask_flat).sum() / (mask_flat.sum() + 1e-6)

        masked_cls = cls_recon * mask.unsqueeze(-1)
        cls_sum = masked_cls.sum(dim=1, keepdim=True)
        denom = (mask.sum(dim=1, keepdim=True) + 1e-6).unsqueeze(-1)
        cls_mean = cls_sum / denom
        cls_mean = cls_mean.expand_as(cls_recon)

        loss_cls_cons = self.mse_loss(cls_recon, cls_mean).mean(-1)
        loss_cls_cons = (loss_cls_cons * mask).sum() / (mask.sum() + 1e-6)

        loss_cls = loss_cls_cons + loss_cls_recon_triplet

        result['loss_cls_recon_triplet'] = loss_cls_recon_triplet.detach().item()
        result['loss_cls'] = loss_cls.detach().item()
        result['loss_cls_cons'] = loss_cls_cons.detach().item()


        result['loss_total'] += loss_cls * self.lambda_cls

        # spatial feature consistency
        neighbor_avg = torch.einsum(
            "BTNC,NN->BTNC", predicted_features, self.adjacency_matrix
        ).detach()
        loss_feat_s = self.mse_loss(
            predicted_features - neighbor_avg,
            torch.zeros_like(predicted_features)
        ).mean(-1)

        if 'mask' in label.keys():
            loss_feat_s = loss_feat_s.mean(-1) * label['mask']
        loss_feat_s = loss_feat_s.mean()
        result['loss_total'] += loss_feat_s * self.lambda_spatial_feat
        result['loss_feat_spatial'] = loss_feat_s.detach().item()

        # temporal feature consistency (BTNC)
        temporal_avg = predicted_features.detach().clone()
        temporal_avg[:, 1:] = predicted_features.detach()[:, :-1]
        temporal_avg[:, 0] = 0.
        temporal_avg[:, :-1] += predicted_features.detach()[:, 1:]
        temporal_avg[:, 1:-1] /= 2.

        loss_feat_t = self.mse_loss(
            predicted_features, temporal_avg.detach()
        ).mean(-1)
        if 'mask' in label.keys():
            loss_feat_t = loss_feat_t.mean(-1) * label['mask']
        loss_feat_t = loss_feat_t.mean()
        result['loss_total'] += loss_feat_t * self.lambda_temporal_feat
        result['loss_feat_temporal'] = loss_feat_t.detach().item()

        result['output'] = out.detach()
        return result


# @full_model
# class STADecoder(AdaptDecoder):
#     # Space-time decoupled AdaptDecoder
#     def __init__(self, model_config, cache_path):
#         super().__init__(model_config, cache_path)
#         self.attn_layer = FusedSTBlock(dim=model_config.num_features, num_heads=8)


# @full_model
# class STADecoder(ReconAdaptDecoder):
#     # Space-time decoupled SingleAdaptDecoder
#     def __init__(self, model_config, cache_path):
#         super().__init__(model_config, cache_path)
#         self.attn_layer = FusedSTBlock(dim=model_config.num_features, num_heads=8)

@full_model
class STADecoder(SingleAdaptDecoder):
    # Space-time decoupled SingleAdaptDecoder
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)
        self.attn_layer = FusedSTBlock(dim=model_config.num_features, num_heads=8)


@full_model
class SingleSTADecoder(SingleAdaptDecoder):
    # Space-time decoupled SingleAdaptDecoder
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)
        self.attn_layer = FusedSTBlock(dim=model_config.num_features, num_heads=8)
