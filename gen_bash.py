import os
import sys
import socket

CMD = sys.argv[1]
# CMD = '/home/bingxing2/ailab/scx6mh7/workspace/sacc_beijingcloud/test.py'

# COMMAND = f'''
# deepspeed --num_nodes {os.environ.get('SLURM_NNODES')} \
#     --num_gpus {os.environ.get('GPUS_PER_NODE')} \
#     --master_addr {socket.gethostbyname(os.environ.get('MASTER_ADDR'))} \
#     --master_port {int(os.environ.get('MASTER_PORT'))} \
#     --hostfile {os.environ.get('SACC_HOME')}/configs/{os.environ.get('SACC_UUID')}/ds_hosts \
#     --launcher SLURM \
#     --no_ssh_check \
#     --force_multi \
#     {CMD}
# '''

COMMAND = f'''
torchrun  --nnodes={os.environ.get('SLURM_NNODES')} \
    --nproc-per-node={os.environ.get('SLURM_GPUS_ON_NODE')} \
    --rdzv-endpoint={os.environ.get('MASTER_ADDR')}:${os.environ.get('MASTER_PORT')} \
    --rdzv-backend=c10d \
    {CMD}
'''

run_sh_path = f"{os.environ.get('SACC_HOME')}/configs/{os.environ.get('SACC_UUID')}/run.sh"
with open(run_sh_path, "w") as f:
    f.write(COMMAND)