"""
实验2：重构误差 vs 语义熵相关性分析（训练条件下多掩码版本）

改进思路：
  原: 单token掩码 → 脱离训练条件，高低难度差异被压缩
  新: 50%随机掩码(与训练一致) × 100次重复 → 在真实推理条件下测量重构误差

方法：
  对每个场景重复 num_repeats 次：
    1. 生成随机 mask（~50% token 被掩码，和训练一致）
    2. student forward → 获取被掩码 token 的 student_feat
    3. 对照 teacher_feat 计算 MSE
    4. 每个 token 累计"被掩码时的误差"和"被掩码次数"
  → 最终每个 token 的平均重构误差 = 累计误差 / 被掩码次数
  → 对比不同难度的 token 的平均重构误差是否不同

多checkpoint支持：
  传入 checkpoint_dir 或 checkpoint_paths，对每个checkpoint独立运行exp2，
  绘制 epoch-r 曲线，观察高难度组的重建误差是否始终高于低难度组

判断标准：
  - Pearson r > 0.5 且 p < 0.001 → 因果链成立，动机充分
  - 0.3 < r < 0.5 → 部分成立，需补充其他证据
  - r < 0.3 → 因果链不成立，需重新审视动机
"""
import os
import glob as glob_mod
import argparse
import logging
import re

import numpy as np
import torch
import torch.nn.functional as F

from shared_utils import (
    build_model_from_config,
    build_dataloader_from_config,
    DifficultyExtractor,
    StatisticsHelper,
    VisualizationHelper,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _extract_epoch_from_path(path: str) -> int:
    basename = os.path.basename(path)
    m = re.search(r"epoch[_]?(\d+)", basename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"iter[_]?(\d+)", basename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return -1


def _sort_checkpoints(paths):
    pairs = [(p, _extract_epoch_from_path(p)) for p in paths]
    pairs.sort(key=lambda x: x[1])
    return pairs


def run_experiment(
    config_path: str,
    checkpoint_path: str = None,
    output_dir: str = "motivation_experiments/results/exp2",
    num_scenes: int = 20,
    num_repeats: int = 100,
    mask_ratio: float = 0.5,
    batch_size: int = 1,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    viz = VisualizationHelper(output_dir)

    logger.info("=" * 60)
    logger.info("Experiment 2: Reconstruction Error vs Semantic Entropy")
    logger.info("(Multi-mask training-condition version)")
    logger.info("=" * 60)
    logger.info(f"num_repeats={num_repeats}, mask_ratio={mask_ratio}, num_scenes={num_scenes}")

    model, cfg = build_model_from_config(config_path, checkpoint_path)
    extractor = DifficultyExtractor(model, device)

    loader = build_dataloader_from_config(
        cfg, split="train", batch_size=batch_size, dataset_type="ScanNetDataset"
    )

    all_difficulty = []
    all_avg_errors = []
    all_teacher_feat_norm = []
    scene_count = 0

    for batch_idx, data_dict in enumerate(loader):
        if scene_count >= num_scenes:
            break

        try:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(device)

            if model.use_semantic_adaptive_masking:
                model.teacher.backbone.return_mid_feature = True
                model.teacher.backbone.mid_feature_layer = model.sde_teacher_feature_layer

            global_point = extractor._build_global_point(data_dict)

            with torch.no_grad():
                teacher_result = model.teacher.backbone(global_point)
                if isinstance(teacher_result, tuple):
                    teacher_point, mid_features = teacher_result[0], teacher_result[1]
                else:
                    teacher_point = teacher_result
                    mid_features = None
                teacher_feat = teacher_point.feat

            num_tokens = (
                teacher_point.num_tokens
                if hasattr(teacher_point, "num_tokens")
                else teacher_feat.shape[0]
            )

            has_token_info = hasattr(teacher_point, "num_tokens")
            if not has_token_info:
                logger.warning(f"Scene {batch_idx}: no token info, skipping")
                continue

            subvoxel_occupancy = teacher_point.subvoxel_occupancy

            if mid_features is not None and model.sde is not None:
                block_difficulty, _ = model.sde(mid_features, subvoxel_occupancy)
                difficulty_np = block_difficulty.cpu().numpy()
            else:
                logger.warning(f"Scene {batch_idx}: no SDE/mid_features, skipping")
                continue

            cumulative_error = torch.zeros(num_tokens, device=device)
            mask_count = torch.zeros(num_tokens, dtype=torch.long, device=device)

            for repeat_idx in range(num_repeats):
                random_mask = extractor.generate_random_mask(num_tokens, mask_ratio)
                masked_indices = random_mask.nonzero(as_tuple=False).squeeze(-1)

                if len(masked_indices) == 0:
                    continue

                mask_global_point = extractor._build_global_point(data_dict)
                mask_global_point["mask"] = random_mask

                with torch.no_grad():
                    student_result = model.student.backbone(mask_global_point)
                    if isinstance(student_result, tuple):
                        student_point = student_result[0]
                    else:
                        student_point = student_result
                    student_feat = student_point.feat

                per_token_error = (
                    (student_feat[masked_indices] - teacher_feat[masked_indices]) ** 2
                ).sum(dim=-1)

                cumulative_error.scatter_add_(
                    0, masked_indices, per_token_error
                )
                mask_count.scatter_add_(
                    0, masked_indices, torch.ones_like(per_token_error, dtype=torch.long)
                )

            valid_mask = mask_count > 0
            avg_error = torch.zeros(num_tokens, device=device)
            avg_error[valid_mask] = cumulative_error[valid_mask] / mask_count[valid_mask].float()
            avg_error_np = avg_error.cpu().numpy()
            teacher_norms = torch.norm(teacher_feat, dim=-1).cpu().numpy()

            valid_np = valid_mask.cpu().numpy()
            scene_avg_errors = avg_error_np[valid_np]
            scene_difficulty = difficulty_np[valid_np]
            scene_norms = teacher_norms[valid_np]

            all_difficulty.append(scene_difficulty)
            all_avg_errors.append(scene_avg_errors)
            all_teacher_feat_norm.append(scene_norms)

            scene_count += 1
            coverage = valid_np.mean()
            logger.info(
                f"Scene {scene_count}/{num_scenes}: "
                f"{num_tokens} tokens, {num_repeats} repeats, "
                f"coverage={coverage:.2%}, "
                f"mean_error={scene_avg_errors.mean():.6f}, "
                f"mean_difficulty={scene_difficulty.mean():.4f}"
            )

        except Exception as e:
            logger.warning(f"Scene {batch_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_difficulty) == 0:
        logger.error("No valid scenes processed!")
        return {}

    difficulty_all = np.concatenate(all_difficulty)
    errors_all = np.concatenate(all_avg_errors)
    norms_all = np.concatenate(all_teacher_feat_norm)

    normalized_errors = errors_all / (norms_all ** 2 + 1e-8)

    correlation = StatisticsHelper.compute_reconstruction_correlation(
        difficulty_all, errors_all
    )
    normalized_correlation = StatisticsHelper.compute_reconstruction_correlation(
        difficulty_all, normalized_errors
    )

    logger.info("\n" + "=" * 50)
    logger.info("Correlation Analysis: Raw Reconstruction Error")
    logger.info("=" * 50)
    logger.info(f"Pearson r  = {correlation['pearson_r']:.4f}  (p = {correlation['pearson_p']:.2e})")
    logger.info(f"Spearman r = {correlation['spearman_r']:.4f}  (p = {correlation['spearman_p']:.2e})")

    logger.info("\nCorrelation Analysis: Normalized Reconstruction Error")
    logger.info(f"Pearson r  = {normalized_correlation['pearson_r']:.4f}  "
                f"(p = {normalized_correlation['pearson_p']:.2e})")
    logger.info(f"Spearman r = {normalized_correlation['spearman_r']:.4f}  "
                f"(p = {normalized_correlation['spearman_p']:.2e})")

    verdict = _judge_correlation(correlation["pearson_r"])
    logger.info(f"\n>>> VERDICT: {verdict}")

    quantile_recon = _compute_quantile_reconstruction(difficulty_all, errors_all)

    quantile_labels = [qr[0] for qr in quantile_recon]
    quantile_means = [qr[1] for qr in quantile_recon]
    quantile_stds = [qr[2] for qr in quantile_recon]

    viz.plot_reconstruction_vs_entropy(
        difficulty_all, errors_all,
        pearson_r=correlation["pearson_r"],
        pearson_p=correlation["pearson_p"],
        filename="reconstruction_vs_entropy_raw.png",
    )
    viz.plot_reconstruction_vs_entropy(
        difficulty_all, normalized_errors,
        pearson_r=normalized_correlation["pearson_r"],
        pearson_p=normalized_correlation["pearson_p"],
        filename="reconstruction_vs_entropy_normalized.png",
    )

    _plot_per_scene_analysis(all_difficulty, all_avg_errors, output_dir)

    _plot_quantile_comparison(
        quantile_labels, quantile_means, quantile_stds, output_dir
    )

    logger.info("\nPer-Quantile Reconstruction Error:")
    for label, mean_err, std_err in quantile_recon:
        logger.info(f"  {label}: mean={mean_err:.6f}, std={std_err:.6f}")

    summary = {
        "Correlation Analysis": _format_correlation(correlation, normalized_correlation),
        "Verdict": verdict,
        "Per-Quantile Reconstruction Error": "\n".join(
            f"- {label}: mean={m:.6f}, std={s:.6f}"
            for label, m, s in quantile_recon
        ),
        "Statistics": (
            f"- Total tokens analyzed: {len(difficulty_all)}\n"
            f"- Scenes processed: {scene_count}\n"
            f"- Repeats per scene: {num_repeats}\n"
            f"- Mask ratio: {mask_ratio}\n"
            f"- Mean difficulty: {difficulty_all.mean():.4f}\n"
            f"- Mean avg recon error: {errors_all.mean():.6f}\n"
            f"- Mean normalized error: {normalized_errors.mean():.6f}"
        ),
    }
    viz.save_summary_report(summary, filename="exp2_report.md")

    return {
        "difficulty_all": difficulty_all,
        "errors_all": errors_all,
        "correlation": correlation,
        "verdict": verdict,
        "pearson_r": correlation["pearson_r"],
    }


def run_multi_checkpoint(
    config_path: str,
    checkpoint_paths: list = None,
    checkpoint_dir: str = None,
    checkpoint_pattern: str = "*.pth",
    output_dir: str = "motivation_experiments/results/exp2",
    num_scenes: int = 20,
    num_repeats: int = 100,
    mask_ratio: float = 0.5,
    batch_size: int = 1,
    device: str = "cuda",
):
    if checkpoint_paths is None and checkpoint_dir is not None:
        checkpoint_paths = sorted(
            glob_mod.glob(os.path.join(checkpoint_dir, checkpoint_pattern))
        )

    if not checkpoint_paths:
        logger.error("No checkpoints provided! Use --checkpoint_paths or --checkpoint_dir")
        return {}

    ckpt_pairs = _sort_checkpoints(checkpoint_paths)
    logger.info(f"Found {len(ckpt_pairs)} checkpoints")

    epoch_results = []
    for ckpt_path, epoch in ckpt_pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing checkpoint: {ckpt_path} (epoch={epoch})")
        logger.info(f"{'='*60}")

        ckpt_output = os.path.join(output_dir, f"epoch_{epoch:04d}")
        result = run_experiment(
            config_path=config_path,
            checkpoint_path=ckpt_path,
            output_dir=ckpt_output,
            num_scenes=num_scenes,
            num_repeats=num_repeats,
            mask_ratio=mask_ratio,
            batch_size=batch_size,
            device=device,
        )

        if result and "pearson_r" in result:
            epoch_results.append({
                "epoch": epoch,
                "checkpoint": ckpt_path,
                "pearson_r": result["pearson_r"],
                "pearson_p": result["correlation"]["pearson_p"],
                "spearman_r": result["correlation"]["spearman_r"],
                "mean_error": result["errors_all"].mean(),
                "mean_difficulty": result["difficulty_all"].mean(),
            })

    if len(epoch_results) > 1:
        _plot_epoch_r_curve(epoch_results, output_dir)

    return {"epoch_results": epoch_results}


def _judge_correlation(pearson_r: float) -> str:
    if pearson_r > 0.5:
        return (
            "CAUSAL CHAIN ESTABLISHED: High entropy regions are indeed harder to predict "
            "(r > 0.5). The motivation for adaptive masking is well-founded."
        )
    elif pearson_r > 0.3:
        return (
            "PARTIAL EVIDENCE: Moderate positive correlation (0.3 < r < 0.5). "
            "The causal chain is partially supported but needs additional evidence."
        )
    elif pearson_r > 0:
        return (
            "WEAK EVIDENCE: Weak positive correlation (r < 0.3). "
            "The causal chain is not well supported. Consider alternative difficulty measures."
        )
    else:
        return (
            "CONTRADICTION: Non-positive correlation! High entropy regions are NOT harder to predict. "
            "The motivation for adaptive masking needs fundamental rethinking."
        )


def _compute_quantile_reconstruction(difficulty, errors, num_quantiles=4):
    edges = np.quantile(difficulty, np.linspace(0, 1, num_quantiles + 1))
    results = []
    for i in range(num_quantiles):
        if i < num_quantiles - 1:
            mask = (difficulty >= edges[i]) & (difficulty < edges[i + 1])
        else:
            mask = (difficulty >= edges[i]) & (difficulty <= edges[i + 1])
        label = f"Q{i+1}({edges[i]:.3f}-{edges[i+1]:.3f})"
        if mask.sum() > 0:
            results.append((label, errors[mask].mean(), errors[mask].std()))
        else:
            results.append((label, 0.0, 0.0))
    return results


def _plot_per_scene_analysis(all_difficulty, all_errors, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scene_mean_diff = [d.mean() for d in all_difficulty]
    scene_mean_err = [e.mean() for e in all_errors]
    axes[0].scatter(scene_mean_diff, scene_mean_err, c="#4e79a7", s=50, alpha=0.7)
    axes[0].set_xlabel("Mean Difficulty per Scene", fontsize=12)
    axes[0].set_ylabel("Mean Recon Error per Scene", fontsize=12)
    axes[0].set_title("Per-Scene Mean", fontsize=13)

    scene_corr = []
    for d, e in zip(all_difficulty, all_errors):
        if len(d) > 2:
            from scipy.stats import pearsonr
            r, _ = pearsonr(d, e)
            scene_corr.append(r)
    if scene_corr:
        axes[1].hist(scene_corr, bins=20, color="#e15759", edgecolor="white", alpha=0.8)
        axes[1].axvline(x=np.mean(scene_corr), color="black", linestyle="--",
                        label=f"Mean r={np.mean(scene_corr):.3f}")
        axes[1].set_xlabel("Pearson r per Scene", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title("Distribution of Per-Scene Correlation", fontsize=13)
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_scene_analysis.png"), dpi=150)
    plt.close(fig)


def _plot_quantile_comparison(labels, means, stds, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"][:len(labels)],
                  edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Reconstruction Error", fontsize=12)
    ax.set_xlabel("Difficulty Quantile", fontsize=12)
    ax.set_title("Reconstruction Error by Difficulty Quantile\n(50% random mask, multi-repeat averaging)", fontsize=13)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quantile_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_epoch_r_curve(epoch_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in epoch_results]
    pearson_rs = [r["pearson_r"] for r in epoch_results]
    mean_errors = [r["mean_error"] for r in epoch_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, pearson_rs, "o-", color="#4e79a7", markersize=8, linewidth=2)
    axes[0].axhline(y=0.5, color="green", linestyle="--", alpha=0.7, label="Strong evidence (r=0.5)")
    axes[0].axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Weak evidence (r=0.3)")
    axes[0].axhline(y=0.0, color="red", linestyle=":", alpha=0.5, label="No correlation (r=0)")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Pearson r", fontsize=12)
    axes[0].set_title("Pearson r across Training", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.1, 1.0)

    axes[1].plot(epochs, mean_errors, "s-", color="#e15759", markersize=8, linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Mean Reconstruction Error", fontsize=12)
    axes[1].set_title("Mean Reconstruction Error across Training", fontsize=13)

    fig.suptitle("Exp2: Epoch-wise Analysis (Multi-Checkpoint)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "epoch_r_curve.png"), dpi=150)
    plt.close(fig)

    csv_path = os.path.join(output_dir, "epoch_r_curve.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,checkpoint,pearson_r,pearson_p,spearman_r,mean_error,mean_difficulty\n")
        for r in epoch_results:
            f.write(
                f"{r['epoch']},{r['checkpoint']},{r['pearson_r']:.6f},"
                f"{r['pearson_p']:.2e},{r['spearman_r']:.6f},"
                f"{r['mean_error']:.6f},{r['mean_difficulty']:.4f}\n"
            )
    logger.info(f"Epoch-r curve data saved to: {csv_path}")


def _format_correlation(raw, normalized):
    lines = [
        "### Raw Reconstruction Error",
        f"- Pearson r = {raw['pearson_r']:.4f} (p = {raw['pearson_p']:.2e})",
        f"- Spearman r = {raw['spearman_r']:.4f} (p = {raw['spearman_p']:.2e})",
        "",
        "### Normalized Reconstruction Error (by teacher feature norm)",
        f"- Pearson r = {normalized['pearson_r']:.4f} (p = {normalized['pearson_p']:.2e})",
        f"- Spearman r = {normalized['spearman_r']:.4f} (p = {normalized['spearman_p']:.2e})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Reconstruction Error Correlation (Multi-Mask)")
    parser.add_argument("--config", type=str,
                        default="configs/volt_sonata/pretrain-voltsonata-sam-0-base.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing multiple checkpoints for epoch-r curve")
    parser.add_argument("--checkpoint_pattern", type=str, default="*.pth",
                        help="Glob pattern for checkpoint files")
    parser.add_argument("--output_dir", type=str,
                        default="motivation_experiments/results/exp2")
    parser.add_argument("--num_scenes", type=int, default=20)
    parser.add_argument("--num_repeats", type=int, default=100,
                        help="Number of random mask repeats per scene")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Mask ratio (consistent with training)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.checkpoint_dir:
        run_multi_checkpoint(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_pattern=args.checkpoint_pattern,
            output_dir=args.output_dir,
            num_scenes=args.num_scenes,
            num_repeats=args.num_repeats,
            mask_ratio=args.mask_ratio,
            device=args.device,
        )
    else:
        run_experiment(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_scenes=args.num_scenes,
            num_repeats=args.num_repeats,
            mask_ratio=args.mask_ratio,
            device=args.device,
        )
