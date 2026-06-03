import math
import logging

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

logger = logging.getLogger(__name__)


class SubVoxelFeatureRecovery(nn.Module):
    def __init__(
        self,
        embed_dim,
        pos_embed_dim=64,
        subvoxel_grid_size=4,
        use_subvoxel_attention=True,
    ):
        super().__init__()
        if not (2 <= subvoxel_grid_size <= 5):
            raise ValueError(
                f"subvoxel_grid_size P={subvoxel_grid_size} out of range [2,5]"
            )
        self.embed_dim = embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.subvoxel_grid_size = subvoxel_grid_size
        self.use_subvoxel_attention = use_subvoxel_attention
        self.num_subvoxels = subvoxel_grid_size ** 3

        self.mlp_3dpe = nn.Sequential(
            nn.Linear(6, pos_embed_dim),
            nn.GELU(),
            nn.Linear(pos_embed_dim, pos_embed_dim),
        )
        self.mlp_proj = nn.Sequential(
            nn.Linear(embed_dim + pos_embed_dim + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if use_subvoxel_attention:
            self.subvoxel_attn = nn.MultiheadAttention(
                embed_dim, num_heads=1, batch_first=True
            )

        self._cached_pe = None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _compute_3d_position_encoding(self, device):
        P = self.subvoxel_grid_size
        gx = torch.arange(P, device=device, dtype=torch.float32)
        gy = torch.arange(P, device=device, dtype=torch.float32)
        gz = torch.arange(P, device=device, dtype=torch.float32)
        grid_3d = torch.stack(
            [
                x.reshape(-1)
                for x in torch.meshgrid(gx, gy, gz, indexing="ij")
            ],
            dim=-1,
        )
        theta = 100.0
        D_e = self.pos_embed_dim
        freq_split = D_e // 3
        freqs_x = 1.0 / theta ** torch.linspace(
            0, 1, freq_split, device=device
        )
        freqs_y = 1.0 / theta ** torch.linspace(
            0, 1, freq_split, device=device
        )
        freqs_z = 1.0 / theta ** torch.linspace(
            0, 1, max(1, freq_split - 1), device=device
        )

        sin_cos = []
        sin_cos.append(torch.sin(freqs_x * grid_3d[:, 0:1]))
        sin_cos.append(torch.cos(freqs_x * grid_3d[:, 0:1]))
        sin_cos.append(torch.sin(freqs_y * grid_3d[:, 1:2]))
        sin_cos.append(torch.cos(freqs_y * grid_3d[:, 1:2]))
        sin_cos.append(torch.sin(freqs_z * grid_3d[:, 2:3]))
        sin_cos.append(torch.cos(freqs_z * grid_3d[:, 2:3]))
        raw_pe = torch.cat(sin_cos, dim=-1)

        if raw_pe.shape[-1] < 6:
            raw_pe = torch.nn.functional.pad(raw_pe, (0, 6 - raw_pe.shape[-1]))
        pe = self.mlp_3dpe(raw_pe[:, :6])
        return pe

    def forward(self, block_features, occupancy):
        M = block_features.shape[0]
        P3 = self.num_subvoxels
        device = block_features.device

        if self._cached_pe is None or self._cached_pe.device != device:
            self._cached_pe = self._compute_3d_position_encoding(device)

        pe = self._cached_pe
        pe_exp = pe.unsqueeze(0).expand(M, -1, -1)
        f_exp = block_features.unsqueeze(1).expand(-1, P3, -1)

        if occupancy.shape[1] != P3:
            if occupancy.shape[1] < P3:
                pad_size = P3 - occupancy.shape[1]
                occupancy = torch.nn.functional.pad(
                    occupancy, (0, 0, 0, pad_size)
                )
            else:
                occupancy = occupancy[:, :P3]

        inp = torch.cat([f_exp, pe_exp, occupancy], dim=-1)
        H = self.mlp_proj(inp)

        if self.use_subvoxel_attention:
            attn_mask = (occupancy.squeeze(-1) == 0)
            H_sa, _ = self.subvoxel_attn(
                H, H, H, key_padding_mask=attn_mask
            )
            H = H + H_sa

        return H
