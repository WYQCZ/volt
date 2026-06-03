"""
实验1：语义熵分布分析 + 分位数掩码分布

目标：
  1. 验证3D场景中语义难度分布确实不均匀
  2. 证明随机掩码在各难度区间的掩码比例相同（与语义难度无关）
  3. 证明自适应掩码聚焦高难度区域

核心逻辑：
  - 随机掩码：每个token以固定概率r被掩码，与难度无关 → 各分位数区间掩码率≈r
  - 自适应掩码：掩码概率与难度正相关 → 高难度区间掩码率 >> 低难度区间
  - 效率分数 = 掩码区域平均难度 / 掩码比例，自适应 > 随机
"""
import os
import argparse
import logging

import numpy as np
import torch

from shared_utils import (
    build_model_from_config,
    build_dataloader_from_config,
    DifficultyExtractor,
    StatisticsHelper,
    VisualizationHelper,
    setup_logging,
)

logger = logging.getLogger(__name__)


def run_experiment(
    config_path: str,
    checkpoint_path: str = None,
    output_dir: str = "motivation_experiments/results/exp1",
    num_scenes: int = 50,
    num_random_samples: int = 100,
    mask_ratio: float = 0.5,
    batch_size: int = 1,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    viz = VisualizationHelper(output_dir)

    logger.info("=" * 60)
    logger.info("Experiment 1: Entropy Distribution & Quantile Mask Analysis")
    logger.info("=" * 60)

    model, cfg = build_model_from_config(config_path, checkpoint_path)
    extractor = DifficultyExtractor(model, device)

    loader = build_dataloader_from_config(
        cfg, split="train", batch_size=batch_size, dataset_type="ScanNetDataset"
    )

    all_difficulty = []
    all_adaptive_masks = []
    all_adaptive_probs = []
    all_random_coverage = []
    scene_count = 0

    for batch_idx, data_dict in enumerate(loader):
        if scene_count >= num_scenes:
            break

        try:
            result = extractor.extract_teacher_features_and_difficulty(
                data_dict, curriculum_alpha=1.0, mask_ratio=mask_ratio
            )

            if "block_difficulty" not in result:
                logger.warning(f"Scene {batch_idx}: no block_difficulty, skipping")
                continue

            block_difficulty = result["block_difficulty"].cpu().numpy()
            num_tokens = result["num_tokens"]
            adaptive_mask = result["adaptive_mask"].cpu().numpy()
            adaptive_probs = result["adaptive_probs"].cpu().numpy()

            random_samples = []
            for _ in range(min(num_random_samples, 20)):
                rm = extractor.generate_random_mask(num_tokens, mask_ratio).cpu().numpy()
                random_samples.append(rm.astype(float))
            random_coverage_mc = np.mean(random_samples, axis=0)

            all_difficulty.append(block_difficulty)
            all_adaptive_masks.append(adaptive_mask)
            all_adaptive_probs.append(adaptive_probs)
            all_random_coverage.append(random_coverage_mc)

            scene_count += 1
            if scene_count % 10 == 0:
                logger.info(f"Processed {scene_count}/{num_scenes} scenes")

        except Exception as e:
            logger.warning(f"Scene {batch_idx} failed: {e}")
            continue

    if len(all_difficulty) == 0:
        logger.error("No valid scenes processed!")
        return {}

    difficulty_all = np.concatenate(all_difficulty)
    adaptive_mask_all = np.concatenate(all_adaptive_masks)
    adaptive_probs_all = np.concatenate(all_adaptive_probs)
    random_coverage_all = np.concatenate(all_random_coverage)

    logger.info(f"\nTotal blocks collected: {len(difficulty_all)}")
    logger.info(f"Difficulty stats: mean={difficulty_all.mean():.4f}, "
                f"std={difficulty_all.std():.4f}, "
                f"median={np.median(difficulty_all):.4f}")

    # ===== 1. 难度分布可视化 =====
    viz.plot_entropy_distribution(
        difficulty_all,
        title="Semantic Difficulty Distribution Across All Scenes",
        filename="difficulty_distribution.png",
    )

    # ===== 2. 分位数掩码分布分析 =====
    logger.info("\n" + "=" * 50)
    logger.info("Quantile Distribution Analysis")
    logger.info("=" * 50)

    random_single_mask = np.random.rand(len(difficulty_all)) < mask_ratio

    random_quantile_stats = StatisticsHelper.compute_quantile_mask_distribution(
        difficulty_all, random_single_mask
    )
    adaptive_quantile_stats = StatisticsHelper.compute_quantile_mask_distribution(
        difficulty_all, adaptive_mask_all
    )

    logger.info("\nRandom Mask - Quantile Distribution:")
    for label, ratio, count in zip(
        random_quantile_stats["quantile_labels"],
        random_quantile_stats["mask_ratios"],
        random_quantile_stats["block_counts"],
    ):
        logger.info(f"  {label}: mask_ratio={ratio:.4f} ({count} blocks)")
    logger.info(f"  Pearson r={random_quantile_stats['pearson_r']:.4f}, "
                f"p={random_quantile_stats['pearson_p']:.2e}")

    logger.info("\nAdaptive Mask - Quantile Distribution:")
    for label, ratio, count in zip(
        adaptive_quantile_stats["quantile_labels"],
        adaptive_quantile_stats["mask_ratios"],
        adaptive_quantile_stats["block_counts"],
    ):
        logger.info(f"  {label}: mask_ratio={ratio:.4f} ({count} blocks)")
    logger.info(f"  Pearson r={adaptive_quantile_stats['pearson_r']:.4f}, "
                f"p={adaptive_quantile_stats['pearson_p']:.2e}")

    viz.plot_quantile_distribution(
        random_quantile_stats,
        title="Random Mask: Distribution Across Difficulty Quantiles",
        filename="quantile_random_mask.png",
    )
    viz.plot_quantile_distribution(
        adaptive_quantile_stats,
        title="Adaptive Mask: Distribution Across Difficulty Quantiles",
        filename="quantile_adaptive_mask.png",
    )

    # ===== 3. 散点图：难度 vs 掩码概率 =====
    viz.plot_scatter_entropy_vs_mask(
        difficulty_all, random_coverage_all,
        strategy_name="Random (Monte Carlo coverage)",
        filename="scatter_random_coverage.png",
    )
    viz.plot_scatter_entropy_vs_mask(
        difficulty_all, adaptive_probs_all,
        strategy_name="Adaptive",
        filename="scatter_adaptive_coverage.png",
    )

    # ===== 4. 效率分数 =====
    random_efficiency = StatisticsHelper.compute_efficiency_score(
        difficulty_all, random_single_mask
    )
    adaptive_efficiency = StatisticsHelper.compute_efficiency_score(
        difficulty_all, adaptive_mask_all
    )

    logger.info("\n" + "=" * 50)
    logger.info("Efficiency Score Comparison")
    logger.info("=" * 50)
    logger.info(f"Random Mask:   {random_efficiency}")
    logger.info(f"Adaptive Mask: {adaptive_efficiency}")

    viz.plot_control_comparison(
        {"Random": random_efficiency, "Adaptive": adaptive_efficiency},
        metric_key="efficiency_score",
        title="Efficiency Score: Random vs Adaptive",
        filename="efficiency_comparison.png",
    )

    # ===== 5. 综合报告 =====
    summary = {
        "Quantile Distribution": _format_quantile_table(
            random_quantile_stats, adaptive_quantile_stats
        ),
        "Efficiency Comparison": _format_efficiency(random_efficiency, adaptive_efficiency),
        "Correlation Analysis": _format_correlation(
            random_quantile_stats, adaptive_quantile_stats
        ),
    }
    viz.save_summary_report(summary, filename="exp1_report.md")

    return {
        "difficulty_all": difficulty_all,
        "random_quantile_stats": random_quantile_stats,
        "adaptive_quantile_stats": adaptive_quantile_stats,
        "random_efficiency": random_efficiency,
        "adaptive_efficiency": adaptive_efficiency,
    }


def _format_quantile_table(random_stats, adaptive_stats):
    lines = ["| Quantile | Random Mask Ratio | Adaptive Mask Ratio | Block Count |"]
    lines.append("|----------|-------------------|-------------------|-------------|")
    for i, label in enumerate(random_stats["quantile_labels"]):
        lines.append(
            f"| {label} | {random_stats['mask_ratios'][i]:.4f} | "
            f"{adaptive_stats['mask_ratios'][i]:.4f} | "
            f"{random_stats['block_counts'][i]} |"
        )
    return "\n".join(lines)


def _format_efficiency(random_eff, adaptive_eff):
    lines = []
    for k in random_eff:
        lines.append(f"- {k}: Random={random_eff[k]:.4f}, Adaptive={adaptive_eff[k]:.4f}")
    return "\n".join(lines)


def _format_correlation(random_stats, adaptive_stats):
    lines = [
        f"- Random: Pearson r={random_stats['pearson_r']:.4f} (p={random_stats['pearson_p']:.2e}), "
        f"Spearman r={random_stats['spearman_r']:.4f}",
        f"- Adaptive: Pearson r={adaptive_stats['pearson_r']:.4f} (p={adaptive_stats['pearson_p']:.2e}), "
        f"Spearman r={adaptive_stats['spearman_r']:.4f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Entropy & Quantile Analysis")
    parser.add_argument("--config", type=str,
                        default="configs/volt_sonata/pretrain-voltsonata-sam-0-base.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="motivation_experiments/results/exp1")
    parser.add_argument("--num_scenes", type=int, default=50)
    parser.add_argument("--num_random_samples", type=int, default=100)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        num_random_samples=args.num_random_samples,
        mask_ratio=args.mask_ratio,
        device=args.device,
    )
