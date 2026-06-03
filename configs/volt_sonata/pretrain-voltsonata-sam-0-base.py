"""
Default configuration for pretraining VoltSonata-SAM model
Semantic-Adaptive Masking for self-supervised pretraining
Dataset: ScanNet v2, ScanNet++, S3DIS, HM3D, ArkitScene, Structured3D
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 96
num_worker = 96
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False

# model settings
model = dict(
    type="VoltSonata-SAM",
    backbone=dict(
        type="VoltAdapter",
        in_channels=9,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        return_mid_feature=True,
        mid_feature_layer=6,
        mask_token=True,
        subvoxel_grid_size=4,
        token_level_output=True,
    ),
    teacher_custom=dict(
        drop_path=0.0,
    ),
    head_in_channels=384,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=0,
    # Semantic-Adaptive Masking parameters
    use_semantic_adaptive_masking=True,
    sde_embed_dim=384,
    sde_pos_embed_dim=64,
    sde_subvoxel_grid_size=4,
    sde_num_prototypes=256,
    sde_prototype_temperature=0.5,
    sde_mask_temperature=0.5,
    sde_topk_percentile=0.1,
    sde_use_subvoxel_attention=True,
    sde_teacher_feature_layer=6,
    curriculum_warmup_ratio=0.2,
    curriculum_schedule="cosine",
    full_block_alignment=True,
)

# scheduler settings
epoch = 200
base_lr = 0.004
lr_decay = 0.9

base_wd = 0.04
final_wd = 0.2

dec_depths = model["backbone"]["depth"]
param_dicts = [
    dict(
        keyword=f"blocks.{b}.",
        lr=base_lr * lr_decay ** (dec_depths - b - 1),
    )
    for b in range(dec_depths)
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
transform = [
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "color", "normal"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
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
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
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
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]

data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="ScanNetDataset",
                split=["train", "val", "test"],
                data_root="data/scannet",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            dict(
                type="ScanNetPPDataset",
                split=[
                    "train_grid1mm_chunk6x6_stride3x3",
                    "val_grid1mm_chunk6x6_stride3x3",
                    "test_grid1mm_chunk6x6_stride3x3",
                ],
                data_root="data/scannetpp",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            dict(
                type="S3DISDataset",
                split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
                data_root="data/s3dis",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            dict(
                type="DefaultDataset",
                split=["Training", "Validation"],
                data_root="data/arkitscenes",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            dict(
                type="HM3DDataset",
                split=["train", "val"],
                data_root="data/hm3d",
                transform=transform,
                test_mode=False,
                force_label=False,
                loop=1,
            ),
            dict(
                type="Structured3DDataset",
                split=["train", "val", "test"],
                data_root="data/structured3d",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
        ],
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
