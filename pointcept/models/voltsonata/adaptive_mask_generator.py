import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdaptiveMaskGenerator(nn.Module):
    def __init__(
        self,
        mask_temperature=0.5,
        min_mask_prob=0.05,
        max_mask_prob=0.95,
    ):
        super().__init__()
        if mask_temperature <= 0:
            raise ValueError(
                f"mask_temperature must be > 0, got {mask_temperature}"
            )
        self.mask_temperature = mask_temperature
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob

    def forward(self, block_difficulty, target_ratio, curriculum_alpha):
        target_ratio = max(self.min_mask_prob, min(self.max_mask_prob, target_ratio))

        median_diff = block_difficulty.median().detach()
        adaptive_probs = target_ratio * (1.0 + torch.sigmoid(
            (block_difficulty - median_diff) / self.mask_temperature
        ))
        adaptive_probs = torch.clamp(
            adaptive_probs, self.min_mask_prob, self.max_mask_prob
        )

        random_probs = torch.full_like(adaptive_probs, target_ratio)
        final_probs = (1 - curriculum_alpha) * random_probs + curriculum_alpha * adaptive_probs
        final_probs = torch.clamp(final_probs, self.min_mask_prob, self.max_mask_prob)

        mask = torch.bernoulli(final_probs).bool()

        if mask.all() or (~mask).all():
            logger.warning(
                "All blocks masked or all unmasked, resampling..."
            )
            for _ in range(10):
                adjusted_probs = final_probs.clone()
                if mask.all():
                    adjusted_probs *= 0.5
                else:
                    adjusted_probs = 1.0 - (1.0 - adjusted_probs) * 0.5
                adjusted_probs = torch.clamp(
                    adjusted_probs, self.min_mask_prob, self.max_mask_prob
                )
                mask = torch.bernoulli(adjusted_probs).bool()
                if not mask.all() and not (~mask).all():
                    break

        return mask, adaptive_probs
