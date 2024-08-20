# 多节点训练的主程序
#!/home/bingxing2/ailab/scx6mh7/peng_tmp_test/miniconda3/envs/hhn/bin/python
import sys
import os
import click
import time

from common.utils.functional_tools import ensure_directory_exists

uuid = str(time.time())

srcipt_path = os.path.split(os.path.realpath(__file__))[0]
configs_path = os.path.join(srcipt_path, "configs")
ensure_directory_exists(configs_path)
ensure_directory_exists(os.path.join(configs_path, uuid))
ensure_directory_exists(os.path.join(srcipt_path, 'logs'))

# ------------配置命令行参数，设置了默认值方便使用-----------------
@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--num_nodes', type=int, default=1, help='Number of nodes')
@click.option('--gpu_per_nodes', type=int, default=4, help='Number of GPUs per node')
@click.option('--job_name', type=str, default=f"{os.environ.get('USER')}_job", help='job name')
@click.option('--zero', type=int, default=2, help='Stage of Deepspeed Zero++')
@click.option('--partition', type=str, default="vip_gpu_ailab", help='partition name')
@click.option('--group', type=str, default="ai4bio", help='job name')
@click.option('--conda_env', type=str, default="hhn", help='conda environment')
@click.option('--workspace', type=str, default="/home/bingxing2/ailab/scx6mh7/workspace/MyTransformers", help='main program dir')
@click.argument('**kwargs', nargs=-1, type=click.UNPROCESSED)
def main(num_nodes, gpu_per_nodes, job_name,zero,partition,group, conda_env, workspace, **kwargs):
    CMD_START = -1
    for i,j in enumerate(sys.argv[1:]):
        if(j.endswith('train.py')):
            CMD_START = i
            break
    CMD = " ".join(sys.argv[CMD_START+1:])

# -------------------要运行的bash命令-------------------------
    BASHCOMMAND = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --qos gpugpu
#SBATCH -N {num_nodes}
#SBATCH --gres=gpu:{gpu_per_nodes}
#SBATCH -p {partition}
#SBATCH -A {group}

#SBATCH --output={srcipt_path}/logs/%j.out
#SBATCH --error={srcipt_path}/logs/%j.err

export NCCL_ALGO=Ring #NCCL通信算法
export NCCL_MAX_NCHANNELS=16 #NCCL最大通道数
export NCCL_MIN_NCHANNELS=16 #NCCL最小通道数
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml #NCCL拓扑文件路径
export NCCL_IB_HCA=mx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23 # InfiniBand通信的超时时间
export NCCL_IB_RETRY_CNT=7 # InfiniBand通信的重试次数

export PYTHONFAULTHANDLER=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_CUDNN_ENABLED=1
export TORCH_CUDNN_ENABLED=1
export TORCH_DISTRIBUTED_CUDNN_BENCHMARK=1
export TORCH_CUDNN_BENCHMARK=1

export SLURM_NNODES={num_nodes}
export GPUS_PER_NODE={gpu_per_nodes}
export SACC_HOME={srcipt_path}
export SACC_UUID={uuid}
echo $SACC_HOME
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export MASTER_PORT=$(python $SACC_HOME/get_good_port.py) # 调用get_good_port.py获取ds主端口，直接使用16666也可以的
export GRADIO_SERVER_PORT=10046
export RANK=$SLURM_PROCID
export GROUP_RANK=$SLURM_NODEID
export WORLD_SIZE=`expr $SLURM_NNODES * $SLURM_GPUS_ON_NODE`

echo "master addr: $MASTER_ADDR"

module load compilers/cuda/12.1
module load nccl/2.18.3-1_cuda12.1
module load compilers/gcc/12.2.0
module load cudnn/8.9.5.29_cuda12.x 
module load tensorboard/2.11.2   
module load anaconda/2021.11
source activate {conda_env}

# 配置ds多节点地址
srun python $SACC_HOME/gen_ds_hosts.py

CMD="{CMD}"

python $SACC_HOME/gen_bash.py "$CMD"


echo "Start time: `date`"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo $SLURM_JOB_NAME
echo $CMD > $SACC_HOME/logs/$SLURM_JOB_ID.cmd.log


srun bash $SACC_HOME/configs/{uuid}/run.sh > $SACC_HOME/logs/$SLURM_JOB_ID.python.log

'''

    # 将bash命令写入脚本中
    with open(f'{srcipt_path}/configs/{uuid}/run_slurm.sh', 'w') as f:
        f.write(BASHCOMMAND)

    # 将slurm的输出定向到本库的logs文件夹中
    os.system(f'sbatch {srcipt_path}/configs/{uuid}/run_slurm.sh > {srcipt_path}/logs/$SLURM_JOB_ID.log')
    os.system(f'parajobs')

if __name__ == '__main__':
    click.disable_unicode_literals_warning = True
    main()


