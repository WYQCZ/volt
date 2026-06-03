"""Ablation: disable full block alignment (use original Sonata loss)"""
_base_ = ["./pretrain-voltsonata-sam-0-base.py"]
model = dict(
    full_block_alignment=False,
)
