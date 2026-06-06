"""Ablation: disable full block alignment flag (note: Sonata's match_neighbour already covers all points, so this flag is kept for API compatibility only and disabling it does not change the actual loss behavior)"""
_base_ = ["./pretrain-voltsonata-sam-0-base.py"]
model = dict(
    full_block_alignment=False,
)
