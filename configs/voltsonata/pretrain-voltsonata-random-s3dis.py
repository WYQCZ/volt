"""
Volt backbone + random masking (no SAM) on S3DIS only.
For motivation validation: baseline model to compare against SAM.
S3DIS: ~272 rooms, ~3.3 GB raw data, much smaller than ScanNet.

Training step analysis:
  S3DIS has 272 rooms, batch_size_per_gpu=48 → 6 steps/epoch
  Official 6-dataset pretrain: 8422 scenes, 176 steps/epoch × 200 epoch = 35,200 steps
  To match equivalent steps: S3DIS needs ~5867 epochs
  - 1200 epoch (7,200 steps) ≈ 41 6DS-epochs: sufficient for motivation trend
  - 2000 epoch (12,000 steps) ≈ 68 6DS-epochs: recommended for convergence
"""
_base_ = ["./pretrain-voltsonata-sam-0-base.py"]

# Disable SAM -> pure random masking
model = dict(
    use_semantic_adaptive_masking=False,
    backbone=dict(in_channels=6),
)

batch_size = 2
num_worker = 8
gradient_accumulation_steps = 8

# Override transform: remove 'normal' from view_keys and Collect
transform = [
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "color"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(type="RandomColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02, p=0.8),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="RandomColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02, p=0.8),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.02}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_color",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color"),
        local_feat_keys=("local_coord", "local_color"),
    ),
]

data = dict(
    train=dict(
        _delete_=True,
        type="S3DISDataset",
        split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        data_root="data/s3dis",
        transform=transform,
        test_mode=False,
        loop=1,
    )
)

# batch_size=2, 2 GPU → 1/GPU, 173 rooms → 86 mini-batches/epoch
# gradient_accumulation=8 → effective batch=16, ~10 optimizer updates/epoch
# 160 epochs × 10 ≈ 1600 optimizer updates total
# Estimated time: ~96 hours
epoch = 160
eval_epoch = 160
