"""
Motivation验证实验 - 主入口脚本

一键运行全部3个实验，或选择运行单个实验。

流程：
  1. quick_pretrain: 若无预训练模型，先训练5-10个epoch获取初始特征
  2. 运行Motivation实验（exp1/exp2/exp3）
  3. 生成综合报告

实验概览：
  exp1: 语义熵分布分析 + 分位数掩码分布（证明问题存在）
  exp2: 重构误差 vs 语义熵相关性（50%随机掩码×多次重复，消除循环论证，建立因果链）
  exp3: 逆自适应掩码控制实验（排除替代解释）

使用方式：
  # 自动quick_pretrain + 运行全部实验
  python run_all.py --config configs/volt_sonata/pretrain-voltsonata-sam-0-base.py

  # 指定已有checkpoint（跳过quick_pretrain）
  python run_all.py --config ... --checkpoint exp/volt_sonata/pretrain/model/model_last.pth

  # 仅运行指定实验
  python run_all.py --config ... --exps 1 2 3

  # 快速测试
  python run_all.py --config ... --quick

  # 仅quick_pretrain
  python run_all.py --config ... --pretrain_only
"""
import os
import sys
import argparse
import logging
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

EXPERIMENTS = {
    1: ("exp1_entropy_quantile", "Entropy Distribution & Quantile Mask Analysis"),
    2: ("exp2_reconstruction_correlation", "Reconstruction Error vs Semantic Entropy"),
    3: ("exp3_control_experiment", "Inverse-Adaptive Control Experiment"),
}

PRETRAIN_SAVE_BASE = "exp/volt_sonata/motivation_pretrain"


def run_quick_pretrain(
    config_path: str,
    pretrain_epochs: int = 10,
    save_path: str = None,
    num_gpus: int = 1,
):
    save_path = save_path or PRETRAIN_SAVE_BASE
    model_dir = os.path.join(save_path, "model")
    last_ckpt = os.path.join(model_dir, "model_last.pth")

    if os.path.isfile(last_ckpt):
        logger.info(f"Found existing checkpoint: {last_ckpt}")
        return last_ckpt

    logger.info("=" * 60)
    logger.info(f"Quick Pre-training: {pretrain_epochs} epochs")
    logger.info(f"Save path: {save_path}")
    logger.info("=" * 60)

    train_script = os.path.join(str(PROJECT_ROOT), "tools", "train.py")
    if not os.path.isfile(train_script):
        logger.error(f"Training script not found: {train_script}")
        return None

    cmd = [
        sys.executable, train_script,
        "--config-file", config_path,
        "--num-gpus", str(num_gpus),
        "--options",
        f"save_path={save_path}",
        f"epoch={pretrain_epochs}",
        f"eval_epoch={pretrain_epochs}",
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        process = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
        )

        if process.returncode != 0:
            logger.error(f"Training failed with return code {process.returncode}")
            return None

    except Exception as e:
        logger.error(f"Training process error: {e}")
        return None

    if os.path.isfile(last_ckpt):
        logger.info(f"Checkpoint saved: {last_ckpt}")
        return last_ckpt

    epoch_ckpts = sorted(Path(model_dir).glob("epoch_*.pth")) if os.path.isdir(model_dir) else []
    if epoch_ckpts:
        latest = str(epoch_ckpts[-1])
        logger.info(f"Using latest epoch checkpoint: {latest}")
        return latest

    logger.error("No checkpoint found after training!")
    return None


def find_existing_checkpoint(save_path: str = None):
    save_path = save_path or PRETRAIN_SAVE_BASE
    model_dir = os.path.join(save_path, "model")

    last_ckpt = os.path.join(model_dir, "model_last.pth")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    epoch_ckpts = sorted(Path(model_dir).glob("epoch_*.pth")) if os.path.isdir(model_dir) else []
    if epoch_ckpts:
        return str(epoch_ckpts[-1])

    return None


def run_single_experiment(
    exp_id: int,
    config_path: str,
    checkpoint_path: str = None,
    output_base: str = "motivation_experiments/results",
    num_scenes: int = 50,
    device: str = "cuda",
    **kwargs,
):
    module_name, description = EXPERIMENTS[exp_id]
    output_dir = os.path.join(output_base, f"exp{exp_id}")

    logger.info(f"\n{'#'*70}")
    logger.info(f"# Experiment {exp_id}: {description}")
    logger.info(f"# Output: {output_dir}")
    logger.info(f"{'#'*70}\n")

    start_time = time.time()

    if exp_id == 1:
        from exp1_entropy_quantile import run_experiment as run_exp1
        result = run_exp1(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            num_scenes=num_scenes,
            num_random_samples=kwargs.get("num_random_samples", 100),
            mask_ratio=kwargs.get("mask_ratio", 0.5),
            device=device,
        )
    elif exp_id == 2:
        from exp2_reconstruction_correlation import run_experiment as run_exp2
        result = run_exp2(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            num_scenes=min(num_scenes, 20),
            num_repeats=kwargs.get("num_repeats", 100),
            mask_ratio=kwargs.get("mask_ratio", 0.5),
            device=device,
        )
    elif exp_id == 3:
        from exp3_control_experiment import run_experiment as run_exp3
        result = run_exp3(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            num_scenes=num_scenes,
            num_samples_per_strategy=kwargs.get("num_samples", 50),
            num_recon_repeats=kwargs.get("num_recon_repeats", 30),
            mask_ratio=kwargs.get("mask_ratio", 0.5),
            device=device,
        )
    else:
        logger.error(f"Unknown experiment ID: {exp_id}")
        return None

    elapsed = time.time() - start_time
    logger.info(f"\nExperiment {exp_id} completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return result


def generate_comprehensive_report(output_base: str):
    report_path = os.path.join(output_base, "comprehensive_report.md")

    sections = []

    sections.append("# Motivation Verification: Comprehensive Report\n")
    sections.append("## Overview\n")
    sections.append(
        "This report summarizes the results of 3 experiments designed to verify "
        "the motivation for semantic-adaptive masking over random masking.\n\n"
        "### Removed experiments\n"
        "- **exp4 (small object)**: Removed because masking operates at voxel-block "
        "(token) level, not object level. 'Small object fully occluded' has no clear "
        "definition under token-level masking.\n"
        "- **exp5 (full-block alignment)**: Removed because Sonata's `match_neighbour` "
        "already matches all student output points (including masked blocks inferred via "
        "attention). Full-block alignment is the existing behavior, not a modification.\n"
    )

    sections.append("## Experiment Summary\n")
    sections.append("| # | Experiment | Purpose | Status |")
    sections.append("|---|-----------|---------|--------|")

    purposes = {
        1: "Prove problem exists (entropy non-uniform, random mask unfocused)",
        2: "Establish causal chain (high entropy → hard to predict), eliminate circular reasoning",
        3: "Exclude alternative explanations (inverse-adaptive control, recon-error efficiency)",
    }

    for exp_id, (_, description) in EXPERIMENTS.items():
        exp_dir = os.path.join(output_base, f"exp{exp_id}")
        report_file = os.path.join(exp_dir, f"exp{exp_id}_report.md")
        if os.path.isfile(report_file):
            status = "COMPLETED"
        elif os.path.isdir(exp_dir):
            status = "PARTIAL"
        else:
            status = "NOT RUN"
        sections.append(f"| {exp_id} | {description} | {purposes[exp_id]} | {status} |")

    sections.append("\n## Detailed Results\n")

    for exp_id in sorted(EXPERIMENTS.keys()):
        exp_dir = os.path.join(output_base, f"exp{exp_id}")
        report_file = os.path.join(exp_dir, f"exp{exp_id}_report.md")
        module_name, description = EXPERIMENTS[exp_id]

        sections.append(f"### Experiment {exp_id}: {description}\n")

        if os.path.isfile(report_file):
            with open(report_file, "r") as f:
                content = f.read()
            sections.append(content)
        else:
            sections.append("*Results not available.*")

        sections.append("")

    sections.append("\n## Conclusion\n")
    sections.append(_generate_conclusion(output_base))

    with open(report_path, "w") as f:
        f.write("\n".join(sections))

    logger.info(f"Comprehensive report saved to: {report_path}")


def _generate_conclusion(output_base: str) -> str:
    lines = []

    exp2_log = os.path.join(output_base, "exp2", "experiment.log")
    if os.path.isfile(exp2_log):
        with open(exp2_log, "r") as f:
            log_content = f.read()
        if "CAUSAL CHAIN ESTABLISHED" in log_content:
            lines.append("- **Causal chain verified**: High entropy regions are harder to predict (Exp 2)")
        elif "PARTIAL EVIDENCE" in log_content:
            lines.append("- **Causal chain partially supported**: Moderate correlation (Exp 2)")
        elif "CONTRADICTION" in log_content:
            lines.append("- **CRITICAL: Causal chain contradicted** (Exp 2) — motivation needs rethinking")

    exp3_log = os.path.join(output_base, "exp3", "experiment.log")
    if os.path.isfile(exp3_log):
        with open(exp3_log, "r") as f:
            log_content = f.read()
        if "STRONGLY SUPPORTED" in log_content:
            lines.append("- **Control experiment passed**: Focusing on difficulty is key, not just non-uniform (Exp 3)")
        elif "NOT SUPPORTED" in log_content:
            lines.append("- **CRITICAL: Adaptive masking not better than random** (Exp 3)")

    if not lines:
        lines.append("*Run all experiments to generate conclusions.*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Motivation Verification Experiments for Semantic-Adaptive Masking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/volt_sonata/pretrain-voltsonata-sam-0-base.py",
        help="Path to model config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint. If not provided, quick_pretrain will run first.",
    )
    parser.add_argument(
        "--output_base", type=str,
        default="motivation_experiments/results",
        help="Base directory for results",
    )
    parser.add_argument(
        "--exps", type=int, nargs="+", default=[1, 2, 3],
        help="Experiment IDs to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--num_scenes", type=int, default=50,
        help="Number of scenes to process per experiment",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer scenes, fewer samples, fewer pretrain epochs",
    )
    parser.add_argument(
        "--pretrain_only", action="store_true",
        help="Only run quick pretrain, then exit",
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=10,
        help="Number of epochs for quick pretrain (default: 10)",
    )
    parser.add_argument(
        "--pretrain_save_path", type=str, default=None,
        help="Save path for quick pretrain checkpoint",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs for quick pretrain",
    )
    parser.add_argument(
        "--report_only", action="store_true",
        help="Only generate report from existing results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.report_only:
        generate_comprehensive_report(args.output_base)
        return

    # ===== Resolve checkpoint =====
    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        existing = find_existing_checkpoint(args.pretrain_save_path)
        if existing:
            checkpoint_path = existing
            logger.info(f"Found existing pretrain checkpoint: {checkpoint_path}")
        else:
            pretrain_epochs = args.pretrain_epochs
            if args.quick:
                pretrain_epochs = min(pretrain_epochs, 5)

            logger.info("No checkpoint provided and none found. Running quick pretrain...")
            checkpoint_path = run_quick_pretrain(
                config_path=args.config,
                pretrain_epochs=pretrain_epochs,
                save_path=args.pretrain_save_path,
                num_gpus=args.num_gpus,
            )

            if checkpoint_path is None:
                logger.error(
                    "Quick pretrain failed! Please provide a checkpoint with --checkpoint "
                    "or fix training configuration."
                )
                return

    logger.info(f"Using checkpoint: {checkpoint_path}")

    if args.pretrain_only:
        logger.info("--pretrain_only mode: exiting after pretrain")
        return

    # ===== Run experiments =====
    kwargs = {}
    if args.quick:
        args.num_scenes = 5
        kwargs["num_random_samples"] = 20
        kwargs["num_repeats"] = 10
        kwargs["num_samples"] = 10
        kwargs["num_recon_repeats"] = 5
        logger.info("QUICK MODE: reduced sample sizes")

    results = {}
    for exp_id in args.exps:
        if exp_id not in EXPERIMENTS:
            logger.error(f"Unknown experiment ID: {exp_id}. Valid: {list(EXPERIMENTS.keys())}")
            continue

        try:
            result = run_single_experiment(
                exp_id=exp_id,
                config_path=args.config,
                checkpoint_path=checkpoint_path,
                output_base=args.output_base,
                num_scenes=args.num_scenes,
                device=args.device,
                **kwargs,
            )
            results[exp_id] = result
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}")
            import traceback
            traceback.print_exc()

    generate_comprehensive_report(args.output_base)

    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info(f"Checkpoint used: {checkpoint_path}")
    logger.info(f"Results saved to: {args.output_base}")
    logger.info(f"Comprehensive report: {os.path.join(args.output_base, 'comprehensive_report.md')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
