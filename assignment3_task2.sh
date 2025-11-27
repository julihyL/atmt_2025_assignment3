#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_put/out_assignment3_task2_%j.out

module load gpu
module load mamba
source activate /scratch/zhidhu/atmt_new
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# PREPARE DATA
# python preprocess.py \
#     --source-lang cz \
#     --target-lang en \
#     --raw-data ~/shares/cz-en/data/raw \
#     --dest-dir ./cz-en/data/prepared \
#     --model-dir ./cz-en/tokenizers \
#     --test-prefix test \
#     --train-prefix train \
#     --valid-prefix valid \
#     --src-vocab-size 8000 \
#     --tgt-vocab-size 8000 \
#     --src-model ./cz-en/tokenizers/cz-bpe-8000.model \
#     --tgt-model ./cz-en/tokenizers/en-bpe-8000.model

# # TRAIN
# python train.py \
#     --cuda \
#     --data cz-en/data/prepared/ \
#     --src-tokenizer ./atmt_a3_MQA/tokenizers/cz-bpe-8000.model \
#     --tgt-tokenizer ./atmt_a3_MQA/tokenizers/en-bpe-8000.model \
#     --source-lang cz \
#     --target-lang en \
#     --batch-size 64 \
#     --arch transformer \
#     --max-epoch 7 \
#     --log-file cz-en/logs/train_task2.log \
#     --save-dir cz-en/checkpoints_task2/ \
#     --ignore-checkpoints \
#     --encoder-dropout 0.1 \
#     --decoder-dropout 0.1 \
#     --dim-embedding 256 \
#     --attention-heads 4 \
#     --dim-feedforward-encoder 1024 \
#     --dim-feedforward-decoder 1024 \
#     --max-seq-len 300 \
#     --n-encoder-layers 3 \
#     --n-decoder-layers 3 

# TRANSLATE
python translate_assign5.py \
    --cuda \
    --beam-size 1,3,5 \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer ./atmt_a3_MQA/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer ./atmt_a3_MQA/tokenizers/en-bpe-8000.model \
    --checkpoint-path ./atmt_a3_MQA/checkpoints/checkpoint_last.pt \
    --output ./out_put/output_task2_opt.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/atomt.pilot.s3it.uzh/cz-en/data/raw/test.en
