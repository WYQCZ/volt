import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter
from torch.nn.init import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch, batch2offset


@MODELS.register_module("VoltAdapter")
class VoltBackboneAdapter(nn.Module):
    def __init__(
        self,
        in_channels=6,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        return_mid_feature=False,
        mid_feature_layer=None,
        mask_token=False,
        subvoxel_grid_size=4,
        token_level_output=False,
    ):
        super().__init__()
        self.return_mid_feature = return_mid_feature
        self.mid_feature_layer = (
            mid_feature_layer if mid_feature_layer is not None else depth // 2
        )
        self.mask_token = mask_token
        self.subvoxel_grid_size = subvoxel_grid_size
        self.token_level_output = token_level_output
        if mask_token:
            self.mask_token_embed = nn.Parameter(torch.zeros(1, embed_dim))
            trunc_normal_(self.mask_token_embed, std=0.02)

        from pointcept.models.volt.volt_base import RoPE, RoPE_Attention, Block
        from pointcept.models.volt.decoder import Decoder
        from timm.layers import Mlp

        self.tokenizer = spconv.SparseConv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            indice_key="embedding",
        )

        if increase_drop_path:
            drop_path_list = torch.linspace(0, drop_path, depth).tolist()
        else:
            drop_path_list = [drop_path] * depth

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=drop_path_list[i],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    mlp_layer=Mlp,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        self.embed_dim = embed_dim
        self.pos_enc = RoPE()
        self.stride = stride
        self.kernel_size = kernel_size
        if not token_level_output:
            self.decoder = Decoder(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                indice_key="embedding",
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    @staticmethod
    def compute_seqlens(batch_indices):
        points_per_batch = torch.bincount(batch_indices + 1)
        cu_seqlens = torch.cumsum(points_per_batch, dim=0, dtype=torch.int32)
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = sequence_lengths.max().item()
        return cu_seqlens, max_seqlen

    def forward(self, data_dict):
        if isinstance(data_dict, dict):
            feat = data_dict["feat"]
            has_mask = "mask" in data_dict
            if has_mask:
                mask = data_dict["mask"]
            if "grid_coord" in data_dict:
                grid_coord = data_dict["grid_coord"]
            else:
                grid_coord = torch.div(
                    data_dict["coord"] - data_dict["coord"].min(0)[0],
                    data_dict["grid_size"],
                    rounding_mode="trunc",
                ).int()
            if "batch" in data_dict:
                batch = data_dict["batch"]
            else:
                batch = offset2batch(data_dict["offset"])
            if "origin_coord" in data_dict:
                origin_coord = data_dict["origin_coord"]
            else:
                origin_coord = None
            grid_size = data_dict.get("grid_size", None)
        else:
            feat = data_dict.feat
            has_mask = "mask" in data_dict.keys()
            if has_mask:
                mask = data_dict.mask
            if "grid_coord" in data_dict.keys():
                grid_coord = data_dict.grid_coord
            else:
                grid_coord = torch.div(
                    data_dict.coord - data_dict.coord.min(0)[0],
                    data_dict.grid_size,
                    rounding_mode="trunc",
                ).int()
            if "batch" in data_dict.keys():
                batch = data_dict.batch
            else:
                batch = offset2batch(data_dict.offset)
            if "origin_coord" in data_dict.keys():
                origin_coord = data_dict.origin_coord
            else:
                origin_coord = None
            grid_size = getattr(data_dict, "grid_size", None)

        N = grid_coord.shape[0]
        token_coords = (grid_coord // self.stride).int()
        token_grid = torch.cat(
            [batch.unsqueeze(-1), token_coords], dim=-1
        )
        unique_tokens, point_to_token = torch.unique(
            token_grid, dim=0, sorted=True, return_inverse=True
        )

        P = self.subvoxel_grid_size
        P3 = P ** 3
        M = unique_tokens.shape[0]
        local_sub_coords = grid_coord % self.stride
        subvoxel_coords = (local_sub_coords * P // self.stride).int()
        subvoxel_coords = torch.clamp(subvoxel_coords, 0, P - 1)
        subvoxel_idx = (
            subvoxel_coords[:, 0] * P * P
            + subvoxel_coords[:, 1] * P
            + subvoxel_coords[:, 2]
        )
        token_sub_idx = point_to_token * P3 + subvoxel_idx
        counts = torch.bincount(token_sub_idx, minlength=M * P3)
        occupancy_flat = counts[: M * P3].reshape(M, P3, 1)
        occupancy = (occupancy_flat > 0).float()

        sparse_shape = torch.add(
            torch.max(grid_coord, dim=0).values, 96
        ).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=batch[-1].item() + 1,
        )
        x = self.tokenizer(x)

        M_spconv = x.features.shape[0]
        spconv_key = (
            x.indices[:, 0].long() * 1000000000
            + x.indices[:, 1].long() * 1000000
            + x.indices[:, 2].long() * 1000
            + x.indices[:, 3].long()
        )
        logical_key = (
            unique_tokens[:, 0].long() * 1000000000
            + unique_tokens[:, 1].long() * 1000000
            + unique_tokens[:, 2].long() * 1000
            + unique_tokens[:, 3].long()
        )
        key_to_logical = {k: i for i, k in enumerate(logical_key.tolist())}
        spconv_to_logical = torch.zeros(M_spconv, dtype=torch.long, device=grid_coord.device)
        for i in range(M_spconv):
            spconv_to_logical[i] = key_to_logical.get(spconv_key[i].item(), 0)

        if self.mask_token and has_mask:
            if mask.shape[0] == M:
                spconv_token_mask = mask[spconv_to_logical]
            elif mask.shape[0] == N:
                logical_token_has_mask = torch_scatter.segment_coo(
                    mask.float(), point_to_token, reduce="max"
                )
                spconv_token_mask = logical_token_has_mask[spconv_to_logical] > 0
            else:
                raise ValueError(
                    f"mask shape {mask.shape} incompatible with N={N} or M={M}"
                )
            x.features[spconv_token_mask] = self.mask_token_embed.expand(
                spconv_token_mask.sum(), -1
            )

        cu_seqlens, max_seqlen = self.compute_seqlens(x.indices[:, 0])
        freqs_cis = self.pos_enc.compute_axial_cis_efficient(x.indices[:, 1:])

        mid_features = None
        features = x.features
        if self.return_mid_feature or self.token_level_output:
            sort_idx = spconv_to_logical.argsort()
            valid_mask = spconv_to_logical[sort_idx] < M

        for i, blk in enumerate(self.blocks):
            if self.return_mid_feature and i == self.mid_feature_layer:
                mid_features_raw = features.detach().clone()
                mid_features = mid_features_raw[sort_idx][valid_mask]
            features = blk(features, freqs_cis, cu_seqlens, max_seqlen)

        x = x.replace_feature(features)

        if self.token_level_output:
            token_feat = x.features[sort_idx][valid_mask]
            token_batch_ids = unique_tokens[:, 0]
            token_offset = batch2offset(token_batch_ids)

            if origin_coord is not None:
                token_origin_coord = torch_scatter.segment_coo(
                    origin_coord, point_to_token, reduce="mean"
                )
            else:
                token_grid_pos = unique_tokens[:, 1:].float() * self.stride
                if grid_size is not None:
                    gs = grid_size if isinstance(grid_size, (int, float)) else grid_size.item() if hasattr(grid_size, 'item') else 0.02
                else:
                    gs = 0.02
                token_origin_coord = token_grid_pos * gs

            out_point = Point(
                feat=token_feat,
                coord=token_origin_coord,
                origin_coord=token_origin_coord,
                batch=token_batch_ids,
                offset=token_offset,
                point_to_token=point_to_token,
                token_coord=unique_tokens[:, 1:].float(),
                num_tokens=M,
                subvoxel_occupancy=occupancy,
            )

            if self.return_mid_feature and mid_features is not None:
                return out_point, mid_features
            return out_point

        x = self.decoder(x)

        output_feat = x.features
        output_batch = x.indices[:, 0]
        output_offset = batch2offset(output_batch)
        output_coord = x.indices[:, 1:].float()
        token_coord = unique_tokens[:, 1:].float() * self.stride

        out_point = Point(
            feat=output_feat,
            coord=output_coord,
            origin_coord=origin_coord if origin_coord is not None else output_coord,
            batch=output_batch,
            offset=output_offset,
            point_to_token=point_to_token,
            token_coord=token_coord,
            num_tokens=M,
            subvoxel_occupancy=occupancy,
        )

        if self.return_mid_feature and mid_features is not None:
            return out_point, mid_features
        return out_point
