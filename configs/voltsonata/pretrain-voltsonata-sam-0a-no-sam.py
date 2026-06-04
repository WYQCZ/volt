"""Ablation: disable SAM, falls back to random masking + Volt backbone"""
_base_ = ["./pretrain-voltsonata-sam-0-base.py"]
model = dict(
    use_semantic_adaptive_masking=False,
)
