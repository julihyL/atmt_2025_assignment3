# 1. 清理已有模块
module purge

# 2. 加载 GPU 模块和 CUDA 12.4.1
module load gpu
module load cuda/12.4.1

# 3. 加载 mamba
module load mamba

# 4. 设置本地缓存目录在 /scratch
export CONDA_PKGS_DIRS=/scratch/zhidhu/conda_pkgs

# 5. 创建环境（只安装必要包，快速）
mamba create -p /scratch/zhidhu/atmt1 \
    -c pytorch -c nvidia \
    python=3.12 \
    pytorch pytorch-cuda=12.4 \
    sentencepiece sacrebleu tqdm numpy pyyaml \
    -y

# 6. 激活环境
conda activate /scratch/zhidhu/atmt1

# 7. 测试 PyTorch GPU 是否可用
python -c "import torch; print('CUDA version:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"