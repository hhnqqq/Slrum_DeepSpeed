#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=vip_gpu_ailab
#SBATCH -A ai4multi
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpugpu
#SBATCH --mail-type=end

source /home/bingxing2/apps/tools/modules/init/profile.sh
module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate hhn

export NCCL_Algo=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

worker_addrs=($(scontrol show hostnames))
master_addr=${worker_addrs[0]}

export RANK=$SLURM_PROCID
export GROUP_RANK=$SLURM_NODEID
export MASTER_ADDR=$master_addr
export MASTER_PORT=29501
export WORLD_SIZE=`expr $SLURM_NNODES * $SLURM_GPUS_ON_NODE`

base_options="--train-dataset-name dna_pretrain_2k \
--eval-dataset-name gue_human \
--model-name llama2 \
--tokenizer-name base \
--output-path /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/output \
--tokenizer-path /home/bingxing2/ailab/scx6mh7/workspace/dnabert2/merged_tokenizer.model \
--ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/llama/llama2.pth \
--tb-log-dir /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/tb_files/new_runs_1 \
--partial-ckpt-path /home/bingxing2/ailab/scx6mh7/workspace/dnabert2/merged_embedding.ckpt \
"

enable_list=("weight_a" "weight_b" "norm" "embedding" "output")

options="$base_options \
    --experiment-name llama2_dna_pretrain_merged_tokenizer \
    --show-loss-step 1 \
    --show-avg-loss-step 1 \
    --mode pretrain \
    --from-pretrained \
    --epochs 5 \
    --batch-size-per-gpu 48 \
    --eval-batch-size-per-gpu 48 \
    --eval-interval 10 \
    --save-interval 10000 \
    --bf16 \
    --warmup 0.03 \
    --variant 7b \
    --device cuda \
    --max-len 700 \
    --max-src-len 700 \
    --eval-max-len 700 \
    --eval-max-src-len 700 \
    --seed 42 \
    --ds-config-path /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/ds_config/pp_config.json \
    --lr 1e-4 \
    --lr-decay-ratio 0.1 \
    --auto-warmup-steps 100 \
    --auto-warmup-rate 0.05 \
    --use-lora \
    --use-lora-plus \
    --lora-rank 128 \
    --activation-checkpoint \
    --atten-type flash_atten \
    --tensorboard \
    --diy-optimizer \
    --save-trainable \
    --enable-list \
    "


for item in "${enable_list[@]}"; do
    options+=" \"$item\""
done


echo ${options}
srun bash -c 'torchrun --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_ON_NODE --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-backend=c10d /home/bingxing2/ailab/scx6mh7/workspace/MyTransformers/train/u_train.py ${options}'

set +x
