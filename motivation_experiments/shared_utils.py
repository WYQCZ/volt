"""
Motivation验证实验 - 共享工具模块

提供数据加载、模型构建、语义难度提取、可视化等通用功能，
所有实验脚本共用此模块。
"""
import os
import sys
import math
import logging
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def setup_project_path():
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = setup_project_path()


def setup_logging(output_dir: str, level=logging.INFO):
    os.makedirs(output_dir, exist_ok=True)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "experiment.log")),
            logging.StreamHandler(),
        ],
    )


def build_model_from_config(config_path: str, checkpoint_path: Optional[str] = None):
    from pointcept.utils.config import Config
    from pointcept.models.builder import MODELS

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model
    model = MODELS.build(model_cfg)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    return model, cfg


def build_dataloader_from_config(
    cfg, split="val", batch_size=1, num_workers=2,
    dataset_index=None, dataset_type=None,
):
    from pointcept.datasets.builder import build_dataset
    from torch.utils.data import DataLoader
    from pointcept.datasets.utils import collate_fn

    data_cfg = getattr(cfg.data, split, None)
    if data_cfg is None:
        data_cfg = cfg.data.train
        logger.warning(f"Split '{split}' not found, using train split")

    if data_cfg.type == "ConcatDataset" and (
        dataset_index is not None or dataset_type is not None
    ):
        if dataset_index is not None:
            sub_cfg = data_cfg.datasets[dataset_index]
            logger.info(f"Loading sub-dataset [{dataset_index}]: {sub_cfg.type}")
        else:
            sub_cfg = None
            for ds_cfg in data_cfg.datasets:
                if ds_cfg.type == dataset_type:
                    sub_cfg = ds_cfg
                    break
            if sub_cfg is None:
                raise ValueError(
                    f"Dataset type '{dataset_type}' not found in ConcatDataset"
                )
            logger.info(f"Loading sub-dataset by type: {sub_cfg.type}")
        dataset = build_dataset(sub_cfg)
    else:
        dataset = build_dataset(data_cfg)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn),
        pin_memory=True,
        drop_last=False,
    )
    return loader


class DifficultyExtractor:
    """从预训练模型中提取语义难度分数的核心工具类"""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_teacher_features_and_difficulty(
        self,
        data_dict: Dict[str, torch.Tensor],
        curriculum_alpha: float = 1.0,
        mask_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)

        global_point = self._build_global_point(data_dict)

        sde_active = (
            self.model.use_semantic_adaptive_masking
            and curriculum_alpha > 0
        )

        if sde_active:
            global_point_, global_feat, mid_features = (
                self.model._teacher_forward_with_mid(global_point)
            )
        else:
            global_point_ = self.model.teacher.backbone(global_point)
            global_feat = global_point_.feat
            mid_features = None

        result = {
            "teacher_feat": global_feat,
            "mid_features": mid_features,
            "global_point": global_point_,
        }

        has_token_info = hasattr(global_point_, "num_tokens")
        if has_token_info:
            result["num_tokens"] = global_point_.num_tokens
            result["token_coord"] = global_point_.token_coord if hasattr(global_point_, "token_coord") else None
            result["token_origin_coord"] = global_point_.origin_coord
            result["token_batch"] = global_point_.batch
            result["subvoxel_occupancy"] = global_point_.subvoxel_occupancy

            if (
                self.model.use_semantic_adaptive_masking
                and mid_features is not None
            ):
                block_difficulty, subvoxel_entropy = self.model.sde(
                    mid_features, global_point_.subvoxel_occupancy
                )
                result["block_difficulty"] = block_difficulty
                result["subvoxel_entropy"] = subvoxel_entropy

                token_mask, adaptive_probs = self.model.adaptive_mask_generator(
                    block_difficulty, mask_ratio, curriculum_alpha
                )
                result["adaptive_mask"] = token_mask
                result["adaptive_probs"] = adaptive_probs

        return result

    @torch.no_grad()
    def compute_reconstruction_error_single_block(
        self,
        data_dict: Dict[str, torch.Tensor],
        block_idx: int,
    ) -> float:
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)

        global_point = self._build_global_point(data_dict)

        with torch.no_grad():
            teacher_point = self.model.teacher.backbone(global_point)
            teacher_feat = teacher_point.feat

        num_tokens = teacher_point.num_tokens if hasattr(teacher_point, "num_tokens") else teacher_feat.shape[0]
        if block_idx >= num_tokens:
            return 0.0

        single_mask = torch.zeros(num_tokens, dtype=torch.bool, device=self.device)
        single_mask[block_idx] = True

        mask_global_point = self._build_global_point(data_dict)
        mask_global_point["mask"] = single_mask

        with torch.no_grad():
            student_point = self.model.student.backbone(mask_global_point)
            student_feat = student_point.feat

        error = F.mse_loss(student_feat[block_idx], teacher_feat[block_idx]).item()
        return error

    @torch.no_grad()
    def generate_random_mask(self, num_tokens: int, mask_ratio: float = 0.5) -> torch.Tensor:
        mask_num = int(num_tokens * mask_ratio)
        perm = torch.randperm(num_tokens, device=self.device)
        token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=self.device)
        token_mask[perm[:mask_num]] = True
        return token_mask

    @torch.no_grad()
    def generate_inverse_adaptive_mask(
        self,
        block_difficulty: torch.Tensor,
        mask_ratio: float = 0.5,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        median_diff = block_difficulty.median().detach()
        inverse_difficulty = 1.0 - block_difficulty
        inverse_probs = mask_ratio * (
            1.0 + torch.sigmoid((inverse_difficulty - (1.0 - median_diff)) / temperature)
        )
        inverse_probs = torch.clamp(inverse_probs, 0.05, 0.95)
        mask = torch.bernoulli(inverse_probs).bool()
        return mask, inverse_probs

    @torch.no_grad()
    def generate_density_adaptive_mask(
        self,
        occupancy: torch.Tensor,
        mask_ratio: float = 0.5,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        occupied_fraction = occupancy.squeeze(-1).mean(dim=1)
        density_probs = mask_ratio * (
            1.0 + torch.sigmoid((occupied_fraction - occupied_fraction.median()) / temperature)
        )
        density_probs = torch.clamp(density_probs, 0.05, 0.95)
        mask = torch.bernoulli(density_probs).bool()
        return mask, density_probs

    @staticmethod
    def _build_global_point(data_dict):
        from pointcept.models.utils.structure import Point
        return Point(
            feat=data_dict["global_feat"],
            coord=data_dict["global_coord"],
            origin_coord=data_dict["global_origin_coord"],
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )


class StatisticsHelper:

    @staticmethod
    def compute_quantile_mask_distribution(
        block_difficulty: np.ndarray,
        mask: np.ndarray,
        num_quantiles: int = 4,
    ) -> Dict[str, Any]:
        quantile_edges = np.quantile(
            block_difficulty, np.linspace(0, 1, num_quantiles + 1)
        )
        quantile_labels = [
            f"Q{i+1}({quantile_edges[i]:.3f}-{quantile_edges[i+1]:.3f})"
            for i in range(num_quantiles)
        ]

        mask_ratios = []
        block_counts = []
        for i in range(num_quantiles):
            if i < num_quantiles - 1:
                in_quantile = (block_difficulty >= quantile_edges[i]) & (
                    block_difficulty < quantile_edges[i + 1]
                )
            else:
                in_quantile = (block_difficulty >= quantile_edges[i]) & (
                    block_difficulty <= quantile_edges[i + 1]
                )
            total = in_quantile.sum()
            masked = (in_quantile & mask).sum()
            ratio = masked / total if total > 0 else 0.0
            mask_ratios.append(ratio)
            block_counts.append(total)

        pearson_r, pearson_p = scipy_stats.pearsonr(block_difficulty, mask.astype(float))
        spearman_r, spearman_p = scipy_stats.spearmanr(block_difficulty, mask.astype(float))

        return {
            "quantile_labels": quantile_labels,
            "mask_ratios": mask_ratios,
            "block_counts": block_counts,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "quantile_edges": quantile_edges,
        }

    @staticmethod
    def compute_efficiency_score(
        block_difficulty: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        masked_difficulty = block_difficulty[mask]
        mask_ratio = mask.mean()
        avg_entropy_masked = masked_difficulty.mean() if len(masked_difficulty) > 0 else 0.0
        efficiency = avg_entropy_masked / mask_ratio if mask_ratio > 0 else 0.0

        return {
            "mask_ratio": mask_ratio,
            "avg_entropy_masked": avg_entropy_masked,
            "efficiency_score": efficiency,
            "total_masked": mask.sum(),
            "total_blocks": len(mask),
        }

    @staticmethod
    def compute_reconstruction_correlation(
        block_difficulty: np.ndarray,
        recon_errors: np.ndarray,
    ) -> Dict[str, float]:
        pearson_r, pearson_p = scipy_stats.pearsonr(block_difficulty, recon_errors)
        spearman_r, spearman_p = scipy_stats.spearmanr(block_difficulty, recon_errors)

        sorted_idx = np.argsort(block_difficulty)
        sorted_diff = block_difficulty[sorted_idx]
        sorted_err = recon_errors[sorted_idx]
        n = len(sorted_diff)
        window = max(1, n // 20)
        smoothed_diff = np.convolve(sorted_diff, np.ones(window) / window, mode="valid")
        smoothed_err = np.convolve(sorted_err, np.ones(window) / window, mode="valid")

        return {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "smoothed_difficulty": smoothed_diff,
            "smoothed_error": smoothed_err,
        }


class VisualizationHelper:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_quantile_distribution(
        self,
        results: Dict[str, Any],
        title: str = "Mask Distribution Across Difficulty Quantiles",
        filename: str = "quantile_distribution.png",
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = results["quantile_labels"]
        ratios = results["mask_ratios"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, ratios, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Uniform 50%")
        ax.set_ylabel("Mask Ratio", fontsize=12)
        ax.set_xlabel("Difficulty Quantile", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.legend()

        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{ratio:.1%}", ha="center", va="bottom", fontsize=10)

        pr = results.get("pearson_r", 0)
        pp = results.get("pearson_p", 1)
        ax.text(0.02, 0.95, f"Pearson r={pr:.3f} (p={pp:.2e})",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_entropy_distribution(
        self,
        block_difficulty: np.ndarray,
        title: str = "Semantic Difficulty Distribution",
        filename: str = "entropy_distribution.png",
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(block_difficulty, bins=50, color="#4e79a7", edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("Difficulty Score", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title("Histogram", fontsize=13)
        axes[0].axvline(x=block_difficulty.mean(), color="red", linestyle="--",
                        label=f"Mean={block_difficulty.mean():.3f}")
        axes[0].axvline(x=np.median(block_difficulty), color="orange", linestyle="--",
                        label=f"Median={np.median(block_difficulty):.3f}")
        axes[0].legend(fontsize=9)

        sorted_diff = np.sort(block_difficulty)
        axes[1].plot(sorted_diff, np.linspace(0, 1, len(sorted_diff)), color="#e15759")
        axes[1].set_xlabel("Difficulty Score", fontsize=12)
        axes[1].set_ylabel("Cumulative Fraction", fontsize=12)
        axes[1].set_title("CDF", fontsize=13)
        q25, q50, q75 = np.quantile(block_difficulty, [0.25, 0.5, 0.75])
        for q, label in [(q25, "Q25"), (q50, "Q50"), (q75, "Q75")]:
            axes[1].axvline(x=q, color="gray", linestyle=":", alpha=0.6)
            axes[1].text(q, 1.02, f"{label}={q:.3f}", fontsize=8, ha="center")

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_scatter_entropy_vs_mask(
        self,
        block_difficulty: np.ndarray,
        mask_ratios: np.ndarray,
        strategy_name: str = "Random",
        filename: str = "scatter_entropy_vs_mask.png",
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(block_difficulty, mask_ratios, alpha=0.3, s=8, c="#4e79a7")
        ax.set_xlabel("Semantic Difficulty", fontsize=12)
        ax.set_ylabel("Mask Probability (coverage)", fontsize=12)
        ax.set_title(f"Difficulty vs Mask Probability ({strategy_name})", fontsize=13)
        ax.set_ylim(-0.05, 1.05)

        pr, pp = scipy_stats.pearsonr(block_difficulty, mask_ratios)
        ax.text(0.02, 0.95, f"Pearson r={pr:.3f} (p={pp:.2e})",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_reconstruction_vs_entropy(
        self,
        block_difficulty: np.ndarray,
        recon_errors: np.ndarray,
        pearson_r: float = 0.0,
        pearson_p: float = 1.0,
        filename: str = "reconstruction_vs_entropy.png",
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(block_difficulty, recon_errors, alpha=0.3, s=8, c="#e15759")
        axes[0].set_xlabel("Semantic Difficulty", fontsize=12)
        axes[0].set_ylabel("Reconstruction Error", fontsize=12)
        axes[0].set_title("Per-Block Scatter", fontsize=13)

        z = np.polyfit(block_difficulty, recon_errors, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(block_difficulty.min(), block_difficulty.max(), 100)
        axes[0].plot(x_fit, p(x_fit), "k--", alpha=0.8, linewidth=2,
                     label=f"Linear fit: slope={z[0]:.4f}")
        axes[0].legend(fontsize=9)
        axes[0].text(0.02, 0.95, f"Pearson r={pearson_r:.3f}\np={pearson_p:.2e}",
                     transform=axes[0].transAxes, fontsize=10, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        num_bins = 20
        bins = np.quantile(block_difficulty, np.linspace(0, 1, num_bins + 1))
        bin_centers = []
        bin_mean_err = []
        bin_std_err = []
        for i in range(num_bins):
            if i < num_bins - 1:
                mask = (block_difficulty >= bins[i]) & (block_difficulty < bins[i + 1])
            else:
                mask = (block_difficulty >= bins[i]) & (block_difficulty <= bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_mean_err.append(recon_errors[mask].mean())
                bin_std_err.append(recon_errors[mask].std())

        axes[1].errorbar(bin_centers, bin_mean_err, yerr=bin_std_err,
                         fmt="o-", color="#4e79a7", capsize=3, markersize=5)
        axes[1].set_xlabel("Semantic Difficulty (binned)", fontsize=12)
        axes[1].set_ylabel("Mean Reconstruction Error", fontsize=12)
        axes[1].set_title("Binned Mean ± Std", fontsize=13)

        fig.suptitle("Reconstruction Error vs Semantic Difficulty", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_control_comparison(
        self,
        strategies: Dict[str, Dict],
        metric_key: str = "efficiency_score",
        title: str = "Control Experiment Comparison",
        filename: str = "control_comparison.png",
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(strategies.keys())
        values = [strategies[n][metric_key] for n in names]
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values, color=colors[:len(names)])
        ax.set_ylabel(metric_key.replace("_", " ").title(), fontsize=12)
        ax.set_title(title, fontsize=14)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def plot_small_object_analysis(
        self,
        object_sizes: np.ndarray,
        full_occlusion_probs: Dict[str, np.ndarray],
        title: str = "Small Object Full Occlusion Probability",
        filename: str = "small_object_occlusion.png",
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy_name, probs in full_occlusion_probs.items():
            sorted_idx = np.argsort(object_sizes)
            ax.plot(object_sizes[sorted_idx], probs[sorted_idx],
                    marker="o", markersize=3, label=strategy_name, alpha=0.7)
        ax.set_xlabel("Object Size (# tokens)", fontsize=12)
        ax.set_ylabel("Full Occlusion Probability", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close(fig)

    def save_csv(self, data: Dict[str, Any], filename: str):
        import csv
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    value = str(value)
                writer.writerow([key, value])
        logger.info(f"Saved CSV: {filepath}")

    def save_summary_report(
        self,
        sections: Dict[str, str],
        filename: str = "summary_report.md",
    ):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.write("# Motivation Verification Experiment Report\n\n")
            for section_title, content in sections.items():
                f.write(f"## {section_title}\n\n")
                f.write(content)
                f.write("\n\n")
        logger.info(f"Saved report: {filepath}")
