"""
Volt backbone + random masking (no SAM) on ScanNet only.
For motivation validation: baseline model to compare against SAM.
"""
_base_ = ["./pretrain-voltsonata-sam-0-base.py"]

# Disable SAM -> pure random masking
model = dict(
    use_semantic_adaptive_masking=False,
)

# Single dataset: ScanNet only
data = dict(
    train=dict(
        type="ScanNetDataset",
        split=["train", "val", "test"],
        data_root="data/scannet",
        transform={{_base_.transform}},
        test_mode=False,
        loop=1,
    )
)

# Single dataset ~1/6 data volume of full 6-dataset pretrain,
# so scale epoch proportionally: 200 * 6 / 1 ≈ 1200, but 200 is
# sufficient for motivation validation (loss convergence visible by epoch 100).
epoch = 200
eval_epoch = 200
