import logging
from itertools import chain
from functools import partial
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler

from pointcept.models.voltsonata.curriculum_scheduler import CurriculumScheduler
from pointcept.models.voltsonata.adaptive_mask_generator import AdaptiveMaskGenerator
from pointcept.models.voltsonata.semantic_difficulty_estimator import (
    SemanticDifficultyEstimator,
)

logger = logging.getLogger(__name__)


class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=4096,
        embed_channels=512,
        num_prototypes=4096,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False
        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        similarity = self.prototype(feat)
        return similarity


@MODELS.register_module("VoltSonata-SAM")
class VoltSonataSAM(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 8,
        roll_mask_loss_weight=2 / 8,
        unmask_loss_weight=4 / 8,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=0,
        use_semantic_adaptive_masking=True,
        sde_embed_dim=None,
        sde_pos_embed_dim=64,
        sde_subvoxel_grid_size=4,
        sde_num_prototypes=256,
        sde_prototype_temperature=0.5,
        sde_mask_temperature=0.5,
        sde_topk_percentile=0.1,
        sde_use_subvoxel_attention=True,
        sde_teacher_feature_layer=None,
        curriculum_warmup_ratio=0.2,
        curriculum_schedule="cosine",
        full_block_alignment=True,
    ):
        super().__init__()
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view
        self.use_semantic_adaptive_masking = use_semantic_adaptive_masking
        self.full_block_alignment = full_block_alignment

        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        self.mask_jitter = mask_jitter

        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None
        self.student_temp = student_temp

        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        self.match_max_k = match_max_k
        self.match_max_r = match_max_r
        self.up_cast_level = up_cast_level

        assert unmask_loss_weight + mask_loss_weight + roll_mask_loss_weight > 0
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        assert num_global_view == 1 or num_global_view == 2

        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}

        student_backbone = build_model(backbone)
        backbone_copy = backbone.copy() if isinstance(backbone, dict) else backbone
        backbone_copy.update(teacher_custom)
        teacher_backbone = build_model(backbone_copy)

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        if use_semantic_adaptive_masking:
            if sde_embed_dim is None:
                sde_embed_dim = head_in_channels
            if sde_teacher_feature_layer is None:
                sde_teacher_feature_layer = getattr(
                    student_backbone, "depth", 12
                ) // 2
            self.sde_teacher_feature_layer = sde_teacher_feature_layer

            self.sde = SemanticDifficultyEstimator(
                embed_dim=sde_embed_dim,
                pos_embed_dim=sde_pos_embed_dim,
                subvoxel_grid_size=sde_subvoxel_grid_size,
                num_prototypes=sde_num_prototypes,
                prototype_temperature=sde_prototype_temperature,
                topk_percentile=sde_topk_percentile,
                use_subvoxel_attention=sde_use_subvoxel_attention,
            )
            self.adaptive_mask_generator = AdaptiveMaskGenerator(
                mask_temperature=sde_mask_temperature,
            )
            self.curriculum_scheduler = CurriculumScheduler(
                warmup_ratio=curriculum_warmup_ratio,
                curriculum_schedule=curriculum_schedule,
            )

            sde_param_count = self.sde.get_param_count()
            logger.info(
                f"SDE parameter count: {sde_param_count} "
                f"({'OK' if sde_param_count <= 800000 else 'EXCEEDS 0.8M limit!'})"
            )
        else:
            self.sde = None
            self.adaptive_mask_generator = None
            self.curriculum_scheduler = None
            self.sde_teacher_feature_layer = None

        self.curriculum_alpha = 0.0

    def before_train(self):
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)

        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = curr_step

        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

        if self.use_semantic_adaptive_masking:
            self.curriculum_scheduler.before_train(total_steps, curr_step)

    def before_step(self):
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()

        if self.use_semantic_adaptive_masking:
            self.curriculum_alpha = self.curriculum_scheduler.step()
        else:
            self.curriculum_alpha = 0.0

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size", self.mask_size, self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio", self.mask_ratio, self.mask_ratio_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/teacher_temp", self.teacher_temp, self.teacher_temp_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/momentum", self.momentum, self.momentum_scheduler.iter,
            )
            if self.use_semantic_adaptive_masking:
                self.trainer.writer.add_scalar(
                    "params/curriculum_alpha",
                    self.curriculum_alpha,
                    self.curriculum_scheduler.current_step,
                )

    def after_step(self):
        with torch.no_grad():
            m = self.momentum
            student_param_list = list(self.student.parameters())
            teacher_param_list = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

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

    def generate_mask(self, coord, offset):
        batch = offset2batch(offset)
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, counts = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    def generate_mask_at_token_level(self, num_tokens, device):
        mask_ratio = self.mask_ratio
        mask_num = int(num_tokens * mask_ratio)
        perm = torch.randperm(num_tokens, device=device)
        token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        token_mask[perm[:mask_num]] = True
        return token_mask

    def generate_adaptive_mask(self, teacher_mid_features, num_tokens, occupancy):
        if teacher_mid_features is None:
            teacher_mid_features = torch.zeros(
                num_tokens, self.sde.subvoxel_recovery.embed_dim,
                device=occupancy.device,
            )

        if occupancy is None:
            P3 = self.sde.subvoxel_recovery.num_subvoxels
            occupancy = torch.ones(
                num_tokens, P3, 1, device=teacher_mid_features.device, dtype=torch.float32
            )

        block_difficulty, subvoxel_entropy = self.sde(
            teacher_mid_features, occupancy
        )
        token_mask, adaptive_probs = self.adaptive_mask_generator(
            block_difficulty, self.mask_ratio, self.curriculum_alpha
        )

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "mask/difficulty_mean", block_difficulty.mean().item(),
                self.curriculum_scheduler.current_step,
            )
            self.trainer.writer.add_scalar(
                "mask/difficulty_median", block_difficulty.median().item(),
                self.curriculum_scheduler.current_step,
            )
            self.trainer.writer.add_scalar(
                "mask/adaptive_mask_ratio", token_mask.float().mean().item(),
                self.curriculum_scheduler.current_step,
            )
            self.trainer.writer.add_scalar(
                "mask/curriculum_alpha", self.curriculum_alpha,
                self.curriculum_scheduler.current_step,
            )

        return token_mask, block_difficulty, adaptive_probs

    @torch.no_grad()
    def match_neighbour(self, view1_coord, view1_offset, view2_coord, view2_offset):
        index2, distance = pointops.knn_query(
            1,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index1 = torch.arange(
            index2.shape[0], device=index2.device, dtype=torch.long
        ).unsqueeze(-1)
        index = torch.cat([index1, index2], dim=-1)[
            distance.squeeze(-1) < self.match_max_r
        ]
        return index

    @torch.no_grad()
    def roll_point(self, point):
        n = self.num_global_view
        bs = len(point.offset) // self.num_global_view
        data_dict = {}
        for key in point.keys():
            if key in ["feat", "coord", "origin_coord", "batch"]:
                value = point[key].split(offset2bincount(point.offset).tolist())
                value = chain(*[value[n * b : n * (b + 1)][::-1] for b in range(bs)])
                if key == "batch":
                    value = [torch.ones_like(v) * i for i, v in enumerate(value)]
                data_dict[key] = torch.cat(list(value), dim=0)
        return Point(data_dict)

    def up_cast(self, point):
        for _ in range(self.up_cast_level):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def _teacher_forward_with_mid(self, global_point):
        if hasattr(self.teacher.backbone, "return_mid_feature"):
            self.teacher.backbone.return_mid_feature = True
            self.teacher.backbone.mid_feature_layer = self.sde_teacher_feature_layer
            result = self.teacher.backbone(global_point)
            if isinstance(result, tuple):
                global_point_, mid_features = result
                if isinstance(global_point_, Point):
                    global_feat = global_point_.feat
                else:
                    global_feat = global_point_
            else:
                global_point_ = result
                global_feat = result.feat if isinstance(result, Point) else result
                mid_features = None
        else:
            result = self.teacher.backbone(global_point)
            if isinstance(result, Point):
                global_point_ = result
                global_feat = result.feat
            else:
                global_point_ = global_point
                global_point_.feat = result
                global_feat = result
            mid_features = None

        if self.up_cast_level > 0:
            global_point_ = self.up_cast(global_point_)
            global_feat = global_point_.feat

        return global_point_, global_feat, mid_features

    def forward(self, data_dict, return_point=False):
        if return_point:
            result = self.teacher.backbone(data_dict)
            if isinstance(result, Point):
                point = result
            elif isinstance(result, tuple):
                point = result[0] if isinstance(result[0], Point) else result[0]
            else:
                point = result
            if isinstance(point, Point):
                for _ in range(self.up_cast_level):
                    assert "pooling_parent" in point.keys()
                    assert "pooling_inverse" in point.keys()
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                    point = parent
            return dict(point=point)

        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            # Step 1: Teacher forward
            # Only extract mid-features when SDE is actually active (curriculum_alpha > 0)
            sde_active = self.use_semantic_adaptive_masking and self.curriculum_alpha > 0
            if sde_active:
                global_point_, global_feat, mid_features = (
                    self._teacher_forward_with_mid(global_point)
                )
            else:
                global_point_ = self.teacher.backbone(global_point)
                global_point_ = self.up_cast(global_point_)
                global_feat = global_point_.feat
                mid_features = None

            # Step 2: Generate token-level mask
            # Masking operates at Volt token granularity (no dimension mismatch)
        has_token_info = hasattr(global_point_, "num_tokens")
        if has_token_info:
            num_tokens = global_point_.num_tokens
            point_to_token = global_point_.point_to_token
            subvoxel_occupancy = global_point_.subvoxel_occupancy
        else:
            num_tokens = global_point_.feat.shape[0]
            point_to_token = None
            subvoxel_occupancy = None

        if self.use_semantic_adaptive_masking and self.curriculum_alpha > 0 and has_token_info:
            token_mask, block_difficulty, adaptive_probs = (
                self.generate_adaptive_mask(
                    mid_features, num_tokens, subvoxel_occupancy,
                )
            )
        elif has_token_info:
            token_mask = self.generate_mask_at_token_level(
                num_tokens, global_point.coord.device
            )
        else:
            global_mask, global_cluster = self.generate_mask(
                global_point.coord, global_point.offset
            )
            token_mask = None

        with torch.no_grad():
            if has_token_info and token_mask is not None:
                mask_for_backbone = token_mask
            elif has_token_info:
                mask_for_backbone = token_mask
            else:
                mask_for_backbone = global_mask

            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None and not has_token_info:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter
                    ),
                    max=self.mask_jitter * 2,
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=mask_for_backbone,
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            result_dict = dict(loss=[])

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            with torch.no_grad():
                global_point_.feat = self.teacher.mask_head(global_feat)

            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                # Note: Sonata's match_neighbour already covers ALL points in the
                # student output (including masked regions inferred via attention),
                # so the loss inherently operates over all blocks (full block alignment).
                # The "full_block_alignment" flag is kept for API compatibility only;
                # setting it to False does NOT restrict the loss to unmasked blocks,
                # the behavior is always equivalent to full block alignment.
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp,
                        dim=-1,
                    ),
                    dim=-1,
                )

                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(
                    roll_mask_loss * self.roll_mask_loss_weight
                )

        if self.unmask_loss_weight > 0:
            with torch.no_grad():
                global_point_.feat = self.teacher.unmask_head(global_feat)

            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)

            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )

            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss in result_dict.values():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return result_dict
