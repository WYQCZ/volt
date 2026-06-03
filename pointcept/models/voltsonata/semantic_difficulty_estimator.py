import logging

import torch
import torch.nn as nn

from pointcept.models.voltsonata.subvoxel_feature_recovery import (
    SubVoxelFeatureRecovery,
)
from pointcept.models.voltsonata.semantic_entropy_aggregator import (
    SemanticEntropyAggregator,
)

logger = logging.getLogger(__name__)


class SemanticDifficultyEstimator(nn.Module):
    def __init__(
        self,
        embed_dim,
        pos_embed_dim=64,
        subvoxel_grid_size=4,
        num_prototypes=256,
        prototype_temperature=0.5,
        topk_percentile=0.1,
        sk_iterations=3,
        use_subvoxel_attention=True,
    ):
        super().__init__()
        self.subvoxel_recovery = SubVoxelFeatureRecovery(
            embed_dim=embed_dim,
            pos_embed_dim=pos_embed_dim,
            subvoxel_grid_size=subvoxel_grid_size,
            use_subvoxel_attention=use_subvoxel_attention,
        )
        self.entropy_aggregator = SemanticEntropyAggregator(
            embed_dim=embed_dim,
            num_prototypes=num_prototypes,
            prototype_temperature=prototype_temperature,
            topk_percentile=topk_percentile,
            sk_iterations=sk_iterations,
        )

    def forward(self, teacher_mid_features, occupancy):
        subvoxel_features = self.subvoxel_recovery(
            teacher_mid_features, occupancy
        )
        block_difficulty, subvoxel_entropy = self.entropy_aggregator(
            subvoxel_features, occupancy
        )

        if torch.isnan(block_difficulty).any() or torch.isinf(block_difficulty).any():
            logger.warning(
                "NaN/Inf detected in block_difficulty, replacing with 0.5"
            )
            block_difficulty = torch.where(
                torch.isnan(block_difficulty) | torch.isinf(block_difficulty),
                torch.tensor(0.5, device=block_difficulty.device),
                block_difficulty,
            )

        return block_difficulty, subvoxel_entropy

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
