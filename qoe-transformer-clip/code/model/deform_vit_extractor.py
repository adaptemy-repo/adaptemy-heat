import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

import timm
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed
from timm.layers import to_2tuple

from . import full_model
from geometry import compute_deform_offset

class ViTExtractor(nn.Module):
    """
    Wrapper around timm Vision Transformer (ViT) / CLIP-ViT to extract
    CLS and patch tokens.

    - model_config: dict like:
        {
            "extractor_name": "DeformViTExtractor",
            "clip_length": 25,
            "feature_norm": True,
            "model_config": {       # model_config_pos or model_config_neg
                "input_resolution": 224,
                "model_resolution": 224,
                "patch_size": 16,
                "num_heads": 12,
                "num_features": 768,
                "input_type": "pano",
                "num_classes": 1000,   
                "train_type": "imgn"   # or "clip"
            }
        }

    - cache_path: path to a checkpoint file (downloaded weights).
    """

    def __init__(self, model_config, cache_path):
        super().__init__()

        self.train_type = model_config.train_type
        self.input_resolution = model_config.input_resolution
        self.model_resolution = model_config.model_resolution
        self.patch_size = model_config.patch_size
        self.cache_path = cache_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_tokens = 1
        self.patch_num = self.input_resolution // self.patch_size

        # Build timm model name from config
        self.model_name = self._build_model_name_from_config(model_config)
        print(f"[ViTExtractor] Using backbone: {self.model_name}")

        # Create model skeleton (no pretrained here; we load manually)
        self.model: VisionTransformer = timm.create_model(
            self.model_name,
            pretrained=True,
            global_pool="",  # return patch tokens + cls token
        )


        if model_config.input_type == 'pano':
            input_shape = (self.input_resolution, self.input_resolution * 2)
        else:
            input_shape = (self.input_resolution, self.input_resolution)
        
        # only update some constants in patch_embed
        self.model.patch_embed.img_size = input_shape
        self.model.patch_embed.grid_size = (self.model.patch_embed.img_size[0] // self.model.patch_embed.patch_size[0],
                                      self.model.patch_embed.img_size[1] // self.model.patch_embed.patch_size[1])
        self.model.patch_embed.num_patches = self.model.patch_embed.grid_size[0] * self.model.patch_embed.grid_size[1]

        # interpolate pos_embed
        if model_config.input_type == 'pano':
            pe = self.model.pos_embed.data
            # Use official interpolation function, using bicubic interp.
            pe_new = resize_pos_embed(pe, 
                                      torch.zeros(1, 1 + self.patch_num * self.patch_num * 2, self.model.embed_dim),
                                      self.num_tokens,
                                      self.model.patch_embed.grid_size)
            self.model.pos_embed = nn.Parameter(pe_new)



    # Helper: pick a timm model name from config
    def _build_model_name_from_config(self, model_config):
        img_size = model_config.input_resolution
        patch_size = model_config.patch_size
        train_type = model_config.train_type.lower()

        # assume ViT-B (768 dim, 12 heads), patch16, 224
        assert patch_size == 16 and img_size == 224, "currently only patch16 and img_size 224 base ViT supported"
        if train_type == "clip":
            return f"vit_base_patch{patch_size}_clip_{img_size}"   # CLIP vit backbone
        elif train_type == "imgn":
            return f"vit_base_patch{patch_size}_{img_size}"         # ImageNet ViT backbone
        else:
            raise ValueError(f"Unknown train_type: {train_type}")

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x:
                - shape (B, C, H, W) for single frames

        Returns:
            cls_tokens:   (B, C_feat)          
            patch_tokens: (B, N_patches, C_feat)
        """

        x = self.model.patch_embed(x)   # BCHW -> BNC
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x[:, :] # self.pre_logits(x) BNC



@full_model
class DeformViTExtractor(ViTExtractor):
    def __init__(self, model_config, cache_path):
        super(DeformViTExtractor, self).__init__(model_config, cache_path)


        # Load pretrained weight from Normal conv2d module
        conv_weight = self.model.patch_embed.proj.weight
        conv_bias = self.model.patch_embed.proj.bias


        # DeformConv
        self.model.patch_embed = DeformPatchEmbed(img_size=self.model.patch_embed.img_size,
                                            patch_size=self.model.patch_embed.patch_size, 
                                            in_chans=3,
                                            embed_dim=self.model.embed_dim,
                                            is_discrete=True,
                                            model_config=model_config)

        self.model.patch_embed.proj.weight = conv_weight
        self.model.patch_embed.proj.bias = conv_bias
        

    def forward(self, x):
        return super(DeformViTExtractor, self).forward(x)


class DeformPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, is_discrete=True, model_config=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = DeformConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Compute offset (constant) and put it inside buffer
        offset = torch.from_numpy(compute_deform_offset(model_config=model_config,
                                                        is_discrete=is_discrete)).float()
        self.register_buffer('offset', offset)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x, self.offset.repeat(B, 1, 1, 1))
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x