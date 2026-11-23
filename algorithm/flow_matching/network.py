# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Visual conditional velocity network for offline Flow Matching."""
from copy import deepcopy
import einops
import torch
from torch import nn
from .mlp import MLP, ResidualMLP
from .embedding import SinusoidalPosEmb
from .modules import SpatialEmb, RandomShiftsAug
from .vit import VitEncoder


class VisionFlowMLP(nn.Module):
    """With ViT backbone"""
    def __init__(
        self,
        backbone: VitEncoder,
        action_dim,
        horizon_steps,
        cond_dim,                       # proprioception only
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,         # visual feature dim
        dropout=0,
        num_img=1,                      # currently only supports 1 or 2
        augment=False,
    ):
        super().__init__()

        # action chunk
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = action_dim * horizon_steps

        # historical proprioception and visual inputs
        self.prop_dim = cond_dim
        self.img_cond_steps = img_cond_steps

        # time
        self.time_dim = time_dim

        self.backbone = backbone
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style
        self.spatial_emb = spatial_emb

        self.dropout = dropout
        self.num_img = num_img
        self.augment = augment

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        if spatial_emb > 0:
            if num_img == 2:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            elif num_img == 1:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            else:
                raise NotImplementedError(f"num_img={num_img} Currently we only support 1 or 2 image inputs")
            visual_feature_dim = spatial_emb * num_img
        else: # spatial embedding not specified, use default value 128
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        self.cond_enc_dim = visual_feature_dim + self.prop_dim     # rgb and  proprioception

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Flow
        input_dim = (
            time_dim + \
                action_dim * horizon_steps + \
                        self.cond_enc_dim
        )

        # output action chunk
        output_dim = action_dim * horizon_steps

        # velocity head
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def forward(
        self,
        action,
        time,
        cond: dict,
        output_embedding=False,
        **kwargs,
    ):
        """
        inputs:
            action: (B, Ta, Da) action chunk
            time: (B,) or float within [0,1), flow time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        outputs:

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = action.shape
        _, T_rgb, C, H, W = cond["rgb"].shape
        # flatten chunk
        action = action.view(B, -1)

        # flatten history (proprioception, here we use the raw input without encoding)
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        # concatenate images in cond by channels
        if self.num_img >1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif self.num_img==1:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            raise ValueError(f"self.num_img={self.num_img} <1. ")
        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # visual and proprioceptive embeddings: get vit output - pass in two images separately
        if self.num_img ==2:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone.forward(rgb1)
            feat1 = self.compress1.forward(feat1, state)

            feat2 = self.backbone.forward(rgb2)
            feat2 = self.compress2.forward(feat2, state)

            feat = torch.cat([feat1, feat2], dim=-1)
        elif self.num_img ==1:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone.forward(rgb)
            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        else:
            raise NotImplementedError(f"num_img={self.num_img} Currently we only support 1 or 2 image inputs")
        cond_encoded = torch.cat([feat, state], dim=-1)   # visual and proprioception inputs.

        # time embedding
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)

        # all embeddings: time, visual-proprioceptive
        emb = torch.cat([action, time_emb, cond_encoded], dim=-1)

        # velocity head
        vel = self.mlp_mean(emb)
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_encoded
        return vel.view(B, Ta, Da)
