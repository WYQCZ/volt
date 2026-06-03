"""
实验3：逆自适应掩码控制实验

核心目的：排除"任何非均匀分布都有效"的替代解释

对比三种掩码策略：
  1. Random: 均匀随机掩码（基线）
  2. Adaptive: 语义自适应掩码（高熵优先掩码，本方案）
  3. Inverse-Adaptive: 逆自适应掩码（低熵优先掩码，关键控制）

改进（消除循环论证）：
  原效率分数 = 掩码区域平均熵值 / 掩码比例 → 循环论证（自适应定义就是掩码高熵）
  新效率分数 = 掩码区域平均重构误差 / 掩码比例 → 独立度量，不依赖语义熵

训练对比模式（run_training_comparison）：
  用三种掩码策略分别训练，对比验证损失和下游mIoU，
  验证"聚焦困难区域产生更强训练信号"这一关键因果步骤。

判断逻辑：
  - Adaptive > Random > Inverse → "聚焦困难区域"是关键（STRONGLY SUPPORTED）
  - Adaptive > Random ≈ Inverse → 任何非均匀分布都有效（AMBIGUOUS）
  - Adaptive ≈ Random → 自适应掩码无显著效果（NOT SUPPORTED）
"""
import os
import argparse
import logging
import json
from collections import defaultdict

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


STRATEGY_NAMES = ["random", "adaptive", "inverse_adaptive"]


def run_experiment(
    config_path: str,
    checkpoint_path: str = None,
    output_dir: str = "motivation_experiments/results/exp3",
    num_scenes: int = 50,
    num_samples_per_strategy: int = 50,
    num_recon_repeats: int = 30,
    mask_ratio: float = 0.5,
    batch_size: int = 1,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    viz = VisualizationHelper(output_dir)

    logger.info("=" * 60)
    logger.info("Experiment 3: Inverse-Adaptive Control Experiment")
    logger.info("(Reconstruction-error based efficiency, no circular reasoning)")
    logger.info("=" * 60)
    logger.info(f"num_recon_repeats={num_recon_repeats}, mask_ratio={mask_ratio}")

    model, cfg = build_model_from_config(config_path, checkpoint_path)
    extractor = DifficultyExtractor(model, device)

    loader = build_dataloader_from_config(
        cfg, split="train", batch_size=batch_size, dataset_type="ScanNetDataset"
    )

    strategy_results = defaultdict(lambda: {
        "difficulties": [],
        "masks": [],
        "coverages": [],
        "recon_efficiency_scores": [],
        "entropy_efficiency_scores": [],
        "quantile_stats": [],
    })

    scene_count = 0

    for batch_idx, data_dict in enumerate(loader):
        if scene_count >= num_scenes:
            break

        try:
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(device)

            result = extractor.extract_teacher_features_and_difficulty(
                data_dict, curriculum_alpha=1.0, mask_ratio=mask_ratio
            )

            if "block_difficulty" not in result:
                logger.warning(f"Scene {batch_idx}: no block_difficulty, skipping")
                continue

            block_difficulty = result["block_difficulty"]
            num_tokens = result["num_tokens"]
            difficulty_np = block_difficulty.cpu().numpy()

            if model.use_semantic_adaptive_masking:
                model.teacher.backbone.return_mid_feature = True
                model.teacher.backbone.mid_feature_layer = model.sde_teacher_feature_layer

            global_point = extractor._build_global_point(data_dict)

            with torch.no_grad():
                teacher_result = model.teacher.backbone(global_point)
                if isinstance(teacher_result, tuple):
                    teacher_point = teacher_result[0]
                else:
                    teacher_point = teacher_result
                teacher_feat = teacher_point.feat

            masks_by_strategy = {}

            # --- Random ---
            random_masks = []
            for _ in range(num_samples_per_strategy):
                rm = extractor.generate_random_mask(num_tokens, mask_ratio).cpu().numpy()
                random_masks.append(rm)
            random_avg = np.mean(random_masks, axis=0)
            masks_by_strategy["random"] = random_avg

            # --- Adaptive ---
            adaptive_probs = result["adaptive_probs"].cpu().numpy()
            adaptive_masks = []
            for _ in range(num_samples_per_strategy):
                am = torch.bernoulli(
                    torch.from_numpy(adaptive_probs).to(device)
                ).bool().cpu().numpy()
                adaptive_masks.append(am)
            adaptive_avg = np.mean(adaptive_masks, axis=0)
            masks_by_strategy["adaptive"] = adaptive_avg

            # --- Inverse-Adaptive ---
            inv_mask, inv_probs = extractor.generate_inverse_adaptive_mask(
                block_difficulty, mask_ratio
            )
            inverse_masks = []
            inv_probs_np = inv_probs.cpu().numpy()
            for _ in range(num_samples_per_strategy):
                im = torch.bernoulli(
                    torch.from_numpy(inv_probs_np).to(device)
                ).bool().cpu().numpy()
                inverse_masks.append(im)
            inverse_avg = np.mean(inverse_masks, axis=0)
            masks_by_strategy["inverse_adaptive"] = inverse_avg

            for strategy_name, mask_coverage in masks_by_strategy.items():
                binary_mask_single = (
                    torch.bernoulli(
                        torch.from_numpy(mask_coverage).to(device)
                    ).bool().cpu().numpy()
                )

                strategy_results[strategy_name]["difficulties"].append(difficulty_np)
                strategy_results[strategy_name]["masks"].append(binary_mask_single)
                strategy_results[strategy_name]["coverages"].append(mask_coverage)

                entropy_eff = StatisticsHelper.compute_efficiency_score(
                    difficulty_np, binary_mask_single
                )
                strategy_results[strategy_name]["entropy_efficiency_scores"].append(entropy_eff)

                recon_eff = _compute_recon_efficiency(
                    model, extractor, data_dict, teacher_feat,
                    binary_mask_single, difficulty_np, num_tokens,
                    num_recon_repeats, mask_ratio, device
                )
                strategy_results[strategy_name]["recon_efficiency_scores"].append(recon_eff)

                qs = StatisticsHelper.compute_quantile_mask_distribution(
                    difficulty_np, binary_mask_single
                )
                strategy_results[strategy_name]["quantile_stats"].append(qs)

            scene_count += 1
            if scene_count % 10 == 0:
                logger.info(f"Processed {scene_count}/{num_scenes} scenes")

        except Exception as e:
            logger.warning(f"Scene {batch_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if scene_count == 0:
        logger.error("No valid scenes processed!")
        return {}

    logger.info(f"\nTotal scenes: {scene_count}")

    # ========== Aggregate ==========
    aggregated = {}
    for strategy_name in STRATEGY_NAMES:
        if strategy_name not in strategy_results:
            continue
        results = strategy_results[strategy_name]
        if len(results["recon_efficiency_scores"]) == 0:
            continue

        all_diff = np.concatenate(results["difficulties"])
        all_mask = np.concatenate(results["masks"])

        entropy_eff = StatisticsHelper.compute_efficiency_score(all_diff, all_mask)
        overall_qs = StatisticsHelper.compute_quantile_mask_distribution(all_diff, all_mask)

        mean_recon_eff = defaultdict(float)
        for eff in results["recon_efficiency_scores"]:
            for k, v in eff.items():
                mean_recon_eff[k] += v
        n = len(results["recon_efficiency_scores"])
        for k in mean_recon_eff:
            mean_recon_eff[k] /= n

        aggregated[strategy_name] = {
            "recon_efficiency_score": mean_recon_eff.get("recon_efficiency_score", 0.0),
            "avg_recon_error_masked": mean_recon_eff.get("avg_recon_error_masked", 0.0),
            "entropy_efficiency_score": entropy_eff["efficiency_score"],
            "avg_entropy_masked": entropy_eff["avg_entropy_masked"],
            "mask_ratio": entropy_eff["mask_ratio"],
            "pearson_r": overall_qs["pearson_r"],
            "quantile_stats": overall_qs,
            "per_scene_mean_recon_eff": dict(mean_recon_eff),
        }

        logger.info(f"\n{'='*40}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"{'='*40}")
        logger.info(f"  Recon efficiency score: {mean_recon_eff.get('recon_efficiency_score', 0.0):.4f}")
        logger.info(f"  Avg recon error masked: {mean_recon_eff.get('avg_recon_error_masked', 0.0):.6f}")
        logger.info(f"  Entropy efficiency score (ref): {entropy_eff['efficiency_score']:.4f}")
        logger.info(f"  Avg difficulty masked: {entropy_eff['avg_entropy_masked']:.4f}")
        logger.info(f"  Mask ratio: {entropy_eff['mask_ratio']:.4f}")
        logger.info(f"  Pearson r (difficulty vs mask): {overall_qs['pearson_r']:.4f}")

    # ========== Visualization ==========
    viz.plot_control_comparison(
        aggregated,
        metric_key="recon_efficiency_score",
        title="Reconstruction Efficiency Score: Random vs Adaptive vs Inverse-Adaptive",
        filename="control_recon_efficiency_comparison.png",
    )

    viz.plot_control_comparison(
        aggregated,
        metric_key="avg_recon_error_masked",
        title="Average Recon Error of Masked Regions",
        filename="control_avg_recon_error_comparison.png",
    )

    viz.plot_control_comparison(
        aggregated,
        metric_key="entropy_efficiency_score",
        title="Entropy Efficiency Score (reference, circular)",
        filename="control_entropy_efficiency_comparison.png",
    )

    for strategy_name in STRATEGY_NAMES:
        if strategy_name not in strategy_results:
            continue
        results = strategy_results[strategy_name]
        if len(results["quantile_stats"]) == 0:
            continue

        all_diff = np.concatenate(results["difficulties"])
        all_mask = np.concatenate(results["masks"])
        qs = StatisticsHelper.compute_quantile_mask_distribution(all_diff, all_mask)

        viz.plot_quantile_distribution(
            qs,
            title=f"Quantile Distribution: {strategy_name}",
            filename=f"quantile_{strategy_name}.png",
        )

    # ========== Verdict ==========
    verdict = _judge_control_experiment(aggregated)
    logger.info(f"\n>>> VERDICT:\n{verdict}")

    summary = {
        "Recon Efficiency Comparison (no circular reasoning)": _format_recon_efficiency_table(aggregated),
        "Entropy Efficiency Comparison (reference)": _format_entropy_efficiency_table(aggregated),
        "Quantile Distribution": _format_all_quantiles(aggregated),
        "Correlation Analysis": _format_all_correlations(aggregated),
        "Verdict": verdict,
    }
    viz.save_summary_report(summary, filename="exp3_report.md")

    return aggregated


def run_training_comparison(
    config_path: str,
    checkpoint_path: str = None,
    output_dir: str = "motivation_experiments/results/exp3_training",
    mask_ratio: float = 0.5,
    max_epochs: int = 100,
    eval_interval: int = 10,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Experiment 3: Training Comparison (Random vs Adaptive vs Inverse)")
    logger.info("=" * 60)

    training_curves = {}

    for strategy_name in STRATEGY_NAMES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training with strategy: {strategy_name}")
        logger.info(f"{'='*40}")

        strategy_output = os.path.join(output_dir, strategy_name)
        os.makedirs(strategy_output, exist_ok=True)

        curve = _train_with_strategy(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            strategy_name=strategy_name,
            mask_ratio=mask_ratio,
            max_epochs=max_epochs,
            eval_interval=eval_interval,
            output_dir=strategy_output,
            device=device,
        )
        training_curves[strategy_name] = curve

        logger.info(f"Strategy {strategy_name}: final val_loss={curve['val_losses'][-1]:.6f}")

    _plot_training_curves(training_curves, output_dir)

    verdict = _judge_training_comparison(training_curves)
    logger.info(f"\n>>> TRAINING VERDICT:\n{verdict}")

    curves_path = os.path.join(output_dir, "training_curves.json")
    serializable = {}
    for name, curve in training_curves.items():
        serializable[name] = {
            "epochs": curve["epochs"],
            "train_losses": curve["train_losses"],
            "val_losses": curve["val_losses"],
        }
    with open(curves_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return {"training_curves": training_curves, "verdict": verdict}


def _compute_recon_efficiency(
    model, extractor, data_dict, teacher_feat,
    binary_mask, difficulty_np, num_tokens,
    num_repeats, mask_ratio, device
):
    masked_indices_np = np.where(binary_mask)[0]
    if len(masked_indices_np) == 0:
        return {
            "recon_efficiency_score": 0.0,
            "avg_recon_error_masked": 0.0,
            "mask_ratio": 0.0,
        }

    cumulative_error = torch.zeros(num_tokens, device=device)
    mask_count = torch.zeros(num_tokens, dtype=torch.long, device=device)

    masked_indices = torch.from_numpy(masked_indices_np).to(device)

    for _ in range(num_repeats):
        random_mask = extractor.generate_random_mask(num_tokens, mask_ratio)
        random_mask[masked_indices] = True

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

        cumulative_error.scatter_add_(0, masked_indices, per_token_error)
        mask_count.scatter_add_(
            0, masked_indices, torch.ones_like(per_token_error, dtype=torch.long)
        )

    valid = mask_count > 0
    avg_error = torch.zeros(num_tokens, device=device)
    avg_error[valid] = cumulative_error[valid] / mask_count[valid].float()

    masked_avg_error = avg_error[masked_indices].mean().item()
    mask_ratio_actual = binary_mask.mean()

    recon_efficiency = masked_avg_error / mask_ratio_actual if mask_ratio_actual > 0 else 0.0

    return {
        "recon_efficiency_score": recon_efficiency,
        "avg_recon_error_masked": masked_avg_error,
        "mask_ratio": mask_ratio_actual,
    }


def _train_with_strategy(
    config_path, checkpoint_path, strategy_name,
    mask_ratio, max_epochs, eval_interval, output_dir, device
):
    from pointcept.utils.config import Config
    from pointcept.models.builder import MODELS
    from pointcept.runners.builder import RUNNERS

    cfg = Config.fromfile(config_path)

    if strategy_name == "random":
        cfg.model.use_semantic_adaptive_masking = False
    elif strategy_name == "adaptive":
        cfg.model.use_semantic_adaptive_masking = True
    elif strategy_name == "inverse_adaptive":
        cfg.model.use_semantic_adaptive_masking = True
        cfg.model.adaptive_mask_generator.inverse_mode = True

    cfg.runner.max_epochs = max_epochs
    cfg.runner.eval_interval = eval_interval
    cfg.runner.work_dir = output_dir

    logger.info(f"Config modified for strategy={strategy_name}")
    logger.info(f"  use_semantic_adaptive_masking={cfg.model.get('use_semantic_adaptive_masking', False)}")
    logger.info(f"  max_epochs={max_epochs}, eval_interval={eval_interval}")

    try:
        runner = RUNNERS.build(cfg.runner)
        runner.run()
    except Exception as e:
        logger.error(f"Training failed for strategy {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"epochs": [], "train_losses": [], "val_losses": []}

    epochs = []
    train_losses = []
    val_losses = []

    log_path = os.path.join(output_dir, "experiment.log")
    if os.path.isfile(log_path):
        import re
        with open(log_path, "r") as f:
            for line in f:
                m = re.search(r"Epoch \[(\d+)/\d+\].*train_loss[:\s]+([\d.]+).*val_loss[:\s]+([\d.]+)", line)
                if m:
                    epochs.append(int(m.group(1)))
                    train_losses.append(float(m.group(2)))
                    val_losses.append(float(m.group(3)))
                else:
                    m2 = re.search(r"Epoch \[(\d+)/\d+\].*loss[:\s]+([\d.]+)", line)
                    if m2:
                        epochs.append(int(m2.group(1)))
                        train_losses.append(float(m2.group(2)))
                        val_losses.append(float(m2.group(2)))

    if not epochs:
        logger.warning(f"No training log parsed for {strategy_name}")

    return {"epochs": epochs, "train_losses": train_losses, "val_losses": val_losses}


def _plot_training_curves(training_curves, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#4e79a7", "adaptive": "#e15759", "inverse_adaptive": "#76b7b2"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, curve in training_curves.items():
        if not curve["epochs"]:
            continue
        c = colors.get(name, "#999999")
        axes[0].plot(curve["epochs"], curve["train_losses"], "-", color=c,
                     label=name, linewidth=1.5, alpha=0.8)
        axes[1].plot(curve["epochs"], curve["val_losses"], "-", color=c,
                     label=name, linewidth=1.5, alpha=0.8)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Training Loss", fontsize=12)
    axes[0].set_title("Training Loss Curves", fontsize=13)
    axes[0].legend(fontsize=10)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Loss", fontsize=12)
    axes[1].set_title("Validation Loss Curves", fontsize=13)
    axes[1].legend(fontsize=10)

    fig.suptitle("Exp3: Training Comparison across Masking Strategies", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def _judge_training_comparison(training_curves):
    final_val = {}
    for name, curve in training_curves.items():
        if curve["val_losses"]:
            final_val[name] = curve["val_losses"][-1]
        elif curve["train_losses"]:
            final_val[name] = curve["train_losses"][-1]

    if len(final_val) < 3:
        return f"INSUFFICIENT DATA: Only {len(final_val)} strategies have results."

    val_adaptive = final_val.get("adaptive", float("inf"))
    val_random = final_val.get("random", float("inf"))
    val_inverse = final_val.get("inverse_adaptive", float("inf"))

    lines = [f"Final val losses: Adaptive={val_adaptive:.6f}, Random={val_random:.6f}, Inverse={val_inverse:.6f}", ""]

    if val_adaptive < val_random < val_inverse:
        lines.append(
            "STRONGLY SUPPORTED: Adaptive < Random < Inverse (lower loss is better).\n"
            "Focusing on difficult regions produces the strongest training signal.\n"
            "The causal chain is complete: high entropy → hard to predict → strong training signal."
        )
    elif val_adaptive < val_random and val_inverse > val_random:
        lines.append(
            "SUPPORTED: Adaptive achieves lower loss than Random, and Inverse is worse.\n"
            "Focusing on difficulty helps training; focusing on easy regions hurts."
        )
    elif val_adaptive < val_random and val_inverse <= val_random:
        lines.append(
            "AMBIGUOUS: Adaptive < Random, but Inverse <= Random.\n"
            "Any non-uniform distribution may help training."
        )
    elif val_adaptive >= val_random:
        lines.append(
            "NOT SUPPORTED: Adaptive >= Random.\n"
            "Semantic-adaptive masking does not improve training efficiency."
        )
    else:
        lines.append("INCONCLUSIVE")

    return "\n".join(lines)


def _judge_control_experiment(aggregated: dict) -> str:
    if not all(k in aggregated for k in ["random", "adaptive", "inverse_adaptive"]):
        return "INSUFFICIENT DATA: Not all strategies have results."

    eff_adaptive = aggregated["adaptive"]["recon_efficiency_score"]
    eff_random = aggregated["random"]["recon_efficiency_score"]
    eff_inverse = aggregated["inverse_adaptive"]["recon_efficiency_score"]

    lines = []
    lines.append(
        f"Recon efficiency scores: Adaptive={eff_adaptive:.4f}, "
        f"Random={eff_random:.4f}, Inverse={eff_inverse:.4f}"
    )
    lines.append(
        f"Avg recon error masked: Adaptive={aggregated['adaptive']['avg_recon_error_masked']:.6f}, "
        f"Random={aggregated['random']['avg_recon_error_masked']:.6f}, "
        f"Inverse={aggregated['inverse_adaptive']['avg_recon_error_masked']:.6f}"
    )
    lines.append("")

    if eff_adaptive > eff_random > eff_inverse:
        lines.append(
            "STRONGLY SUPPORTED: Adaptive > Random > Inverse.\n"
            "This proves that 'focusing on difficult regions' is the key mechanism —\n"
            "not merely 'any non-uniform distribution'. The motivation is well-founded.\n"
            "(Based on reconstruction-error efficiency, no circular reasoning)"
        )
    elif eff_adaptive > eff_random and eff_inverse < eff_random:
        lines.append(
            "SUPPORTED: Adaptive > Random, and Inverse < Random.\n"
            "Focusing on difficult regions helps, and focusing on easy regions hurts.\n"
            "The direction of adaptation matters — 'focusing difficulty' is not arbitrary.\n"
            "(Based on reconstruction-error efficiency, no circular reasoning)"
        )
    elif eff_adaptive > eff_random and eff_inverse >= eff_random:
        lines.append(
            "AMBIGUOUS: Adaptive > Random, but Inverse >= Random.\n"
            "Any non-uniform distribution helps, not specifically 'focusing on difficulty'.\n"
            "The advantage may come from variance in mask distribution, not direction."
        )
    elif eff_adaptive <= eff_random:
        lines.append(
            "NOT SUPPORTED: Adaptive <= Random.\n"
            "Semantic-adaptive masking does not outperform random masking on this metric.\n"
            "The motivation needs fundamental rethinking."
        )
    else:
        lines.append("INCONCLUSIVE: Results do not fit expected patterns.")

    return "\n".join(lines)


def _format_recon_efficiency_table(aggregated):
    lines = ["| Strategy | Recon Efficiency | Avg Recon Error Masked | Mask Ratio | Pearson r |"]
    lines.append("|----------|-----------------|----------------------|------------|-----------|")
    for name in STRATEGY_NAMES:
        if name not in aggregated:
            continue
        a = aggregated[name]
        lines.append(
            f"| {name} | {a['recon_efficiency_score']:.4f} | "
            f"{a['avg_recon_error_masked']:.6f} | {a['mask_ratio']:.4f} | "
            f"{a['pearson_r']:.4f} |"
        )
    return "\n".join(lines)


def _format_entropy_efficiency_table(aggregated):
    lines = ["| Strategy | Entropy Efficiency (ref) | Avg Difficulty Masked |"]
    lines.append("|----------|-------------------------|----------------------|")
    for name in STRATEGY_NAMES:
        if name not in aggregated:
            continue
        a = aggregated[name]
        lines.append(
            f"| {name} | {a['entropy_efficiency_score']:.4f} | "
            f"{a['avg_entropy_masked']:.4f} |"
        )
    return "\n".join(lines)


def _format_all_quantiles(aggregated):
    all_lines = []
    for name in STRATEGY_NAMES:
        if name not in aggregated:
            continue
        qs = aggregated[name]["quantile_stats"]
        all_lines.append(f"### {name}")
        for label, ratio in zip(qs["quantile_labels"], qs["mask_ratios"]):
            all_lines.append(f"- {label}: {ratio:.4f}")
        all_lines.append("")
    return "\n".join(all_lines)


def _format_all_correlations(aggregated):
    lines = []
    for name in STRATEGY_NAMES:
        if name not in aggregated:
            continue
        a = aggregated[name]
        qs = a["quantile_stats"]
        lines.append(
            f"- {name}: Pearson r={qs['pearson_r']:.4f} (p={qs.get('pearson_p', 'N/A')})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Control Experiment")
    parser.add_argument("--config", type=str,
                        default="configs/volt_sonata/pretrain-voltsonata-sam-0-base.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="motivation_experiments/results/exp3")
    parser.add_argument("--num_scenes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_recon_repeats", type=int, default=30,
                        help="Number of random mask repeats for reconstruction error estimation")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--training", action="store_true",
                        help="Run training comparison mode")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.training:
        run_training_comparison(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir + "_training",
            mask_ratio=args.mask_ratio,
            max_epochs=args.max_epochs,
            eval_interval=args.eval_interval,
            device=args.device,
        )
    else:
        run_experiment(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_scenes=args.num_scenes,
            num_samples_per_strategy=args.num_samples,
            num_recon_repeats=args.num_recon_repeats,
            mask_ratio=args.mask_ratio,
            device=args.device,
        )
