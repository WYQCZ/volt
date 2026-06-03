#!/bin/bash
# ============================================================
# Volt Motivation Experiments - 远程服务器一键部署脚本
#
# 服务器配置:
#   GPU: Tesla V100S (compute capability 7.0)
#   Driver: 510.39.01 → Max CUDA runtime = 11.6
#   OS: Ubuntu 20.04, glibc 2.31
#   conda: 24.5.0, base Python 3.12.4
#
# 版本链 (严格受驱动限制):
#   Python 3.11 (spconv-cu116 无 cp312 wheel)
#   PyTorch 1.13.1+cu116 (最后支持 cu116 的官方版本)
#   torchvision 0.14.1
#   spconv-cu116 2.3.6
#   timm 0.9.12 (最后兼容 torch 1.x 的版本)
#   torch-scatter/cluster: torch-1.13.1+cu116
#   flash-attn: 不可用 (需 CUDA >= 11.8), 代码已有 fallback
#   peft/transformers: 不安装 (代码已改为懒加载, 实验不需要)
# ============================================================

set -e

VOLT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "Project root: $VOLT_ROOT"

echo "=========================================="
echo "Step 1: 创建 conda 环境 (Python 3.11)"
echo "=========================================="
conda create -n volt python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate volt

echo "=========================================="
echo "Step 2: 安装 CUDA 11.6 Toolkit (conda, 用于编译 pointops)"
echo "=========================================="
conda install -c nvidia/label/cuda-11.6.2 cuda-toolkit -y 2>/dev/null || {
    echo "conda nvidia channel 不可用, 尝试 cudatoolkit..."
    conda install cudatoolkit=11.6 -y
}
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version || echo "WARNING: nvcc not found, pointops 编译可能失败"

echo "=========================================="
echo "Step 3: 安装 PyTorch 1.13.1+cu116"
echo "=========================================="
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

echo "=========================================="
echo "Step 4: 安装 PyG 依赖 (torch-scatter, torch-cluster)"
echo "=========================================="
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install torch-geometric

echo "=========================================="
echo "Step 5: 安装 spconv-cu116 (稀疏卷积)"
echo "=========================================="
pip install spconv-cu116

echo "=========================================="
echo "Step 6: 安装项目 Python 依赖"
echo "=========================================="
pip install \
    "timm>=0.9.12,<1.0.0" \
    scipy matplotlib einops addict yapf \
    h5py pyyaml tensorboard tensorboardx \
    plyfile termcolor open3d imageio \
    albumentations packaging pillow pandas \
    safetensors wheel

echo "=========================================="
echo "Step 7: 编译 pointops (CUDA 扩展)"
echo "=========================================="
cd "${VOLT_ROOT}/libs/pointops"
python setup.py install 2>&1 || {
    echo "ERROR: pointops 编译失败!"
    echo "请确认 nvcc 可用: nvcc --version"
    echo "如果缺少 gcc: apt install build-essential"
    exit 1
}
cd "${VOLT_ROOT}"

echo "=========================================="
echo "Step 8: 编译 pointgroup_ops (CUDA 扩展)"
echo "=========================================="
cd "${VOLT_ROOT}/libs/pointgroup_ops"
python setup.py install 2>&1 || echo "WARNING: pointgroup_ops 编译失败 (非关键)"
cd "${VOLT_ROOT}"

echo "=========================================="
echo "Step 9: 编译 pointseg (C++ 扩展, 无 CUDA)"
echo "=========================================="
cd "${VOLT_ROOT}/libs/pointseg"
python setup.py install 2>&1 || echo "WARNING: pointseg 编译失败 (非关键)"
cd "${VOLT_ROOT}"

echo "=========================================="
echo "Step 10: 验证环境"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')

try:
    import pointops
    print('pointops: OK')
except ImportError as e:
    print(f'pointops: FAILED - {e}')

try:
    import spconv
    print('spconv: OK')
except ImportError as e:
    print(f'spconv: FAILED - {e}')

try:
    import flash_attn
    print('flash_attn: OK')
except ImportError:
    print('flash_attn: NOT INSTALLED (will use fallback attention)')

try:
    import torch_scatter
    print('torch_scatter: OK')
except ImportError as e:
    print(f'torch_scatter: FAILED - {e}')

try:
    import timm
    print(f'timm: {timm.__version__}')
except ImportError as e:
    print(f'timm: FAILED - {e}')

try:
    from pointcept.models.builder import MODELS
    print(f'pointcept MODELS: OK (registry: {MODELS.name})')
except Exception as e:
    print(f'pointcept MODELS: FAILED - {e}')
"

echo ""
echo "=========================================="
echo "部署完成!"
echo "=========================================="
echo ""
echo "运行实验:"
echo "  conda activate volt"
echo "  cd ${VOLT_ROOT}/motivation_experiments"
echo ""
echo "  # 快速验证 (1-2min, 1场景)"
echo "  python run_all.py --config ../configs/volt_sonata/pretrain-voltsonata-sam-0-base.py --quick --exps 1"
echo ""
echo "  # 正式运行全部 (3个实验)"
echo "  python run_all.py --config ../configs/volt_sonata/pretrain-voltsonata-sam-0-base.py --num_scenes 50"
echo ""
echo "注意:"
echo "  - flash-attn 不可用 (CUDA 11.6 限制), 已自动 fallback 到手动 attention"
echo "  - V100 不支持 bfloat16, 已自动降级为 float16"
echo "  - 无需安装 peft/transformers (实验不需要, 已改为懒加载)"
