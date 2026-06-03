import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.init import trunc_normal_

from pointcept.utils.comm import get_world_size, all_gather

logger = logging.getLogger(__name__)


class SemanticEntropyAggregator(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_prototypes=256,
        prototype_temperature=0.5,
        topk_percentile=0.1,
        sk_iterations=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.prototype_temperature = prototype_temperature
        self.topk_percentile = topk_percentile
        self.sk_iterations = sk_iterations

        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, embed_dim))
        trunc_normal_(self.prototypes, std=0.02)

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        feat = feat.float()
        q = torch.exp(feat / temp).t()
        n = sum(all_gather(q.shape[1]))
        k = q.shape[0]

        sum_q = q.sum()
        if get_world_size() > 1:
            dist.all_reduce(sum_q)
        q = q / sum_q

        for _ in range(num_iter):
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum / k
            q = q / q.sum(dim=0, keepdim=True) / n

        q *= n
        return q.t()

    def forward(self, subvoxel_features, occupancy):
        M, P3, D = subvoxel_features.shape
        K = self.num_prototypes

        H_norm = F.normalize(subvoxel_features, dim=-1)
        C_norm = F.normalize(self.prototypes, dim=-1)

        sim = torch.matmul(H_norm, C_norm.t())

        sim_flat = sim.reshape(-1, K)
        probs_flat = self.sinkhorn_knopp(
            sim_flat / self.prototype_temperature,
            1.0,
            num_iter=self.sk_iterations,
        )
        probs = probs_flat.reshape(M, P3, K)

        eps = 1e-8
        subvoxel_entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

        if occupancy is not None:
            occupied_mask = (occupancy.squeeze(-1) > 0).float()
            subvoxel_entropy = subvoxel_entropy * occupied_mask

        k = max(1, int(P3 * self.topk_percentile))
        if occupancy is not None:
            occupied_count = occupied_mask.sum(dim=1).long()
            k_per_block = torch.clamp(
                torch.minimum(
                    occupied_count,
                    torch.full_like(occupied_count, k)
                ),
                min=1,
            )
            block_entropy = torch.zeros(M, device=subvoxel_features.device)
            occupied_mask_bool = occupied_mask.bool()
            for i in range(M):
                ki = k_per_block[i].item()
                row = subvoxel_entropy[i]
                occ_mask = occupied_mask_bool[i]
                occupied_vals = row[occ_mask]
                if len(occupied_vals) > 0:
                    topk_vals, _ = occupied_vals.topk(min(ki, len(occupied_vals)))
                    block_entropy[i] = topk_vals.mean()
                else:
                    block_entropy[i] = 0.0
        else:
            topk_vals, _ = subvoxel_entropy.topk(k, dim=1)
            block_entropy = topk_vals.mean(dim=1)

        H_max = math.log(K)
        block_difficulty = block_entropy / H_max
        block_difficulty = torch.clamp(block_difficulty, 0.0, 1.0)

        return block_difficulty, subvoxel_entropy
