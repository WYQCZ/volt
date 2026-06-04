"""Ablation: disable subvoxel attention in SDE"""
_base_ = ["./pretrain-voltsonata-sam-0-base.py"]
model = dict(
    sde_use_subvoxel_attention=False,
)
