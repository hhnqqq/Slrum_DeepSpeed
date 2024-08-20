import os
import torch
import deepspeed
import torch.distributed as dist
from argparse import Namespace

def init_dist(args):
    if int(os.environ.get('SLURM_NNODES', 1)) > 1:
        local_rank = int(os.environ['RANK']) % int(os.environ['SLURM_GPUS_ON_NODE'])
        args.local_rank = local_rank
        os.environ['LOCAL_RANK'] = str(local_rank)
    if args.device == 'cuda':
        if args.local_rank == -1:
            device = torch.device("cuda")
        else:
            deepspeed.init_distributed(dist_backend="nccl")
            args.world_size = dist.get_world_size()
            args.global_rank = dist.get_rank()
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
    else:
        device = 'cpu'
    args.device = device
    return args
    
args = Namespace()
args.device = 'cuda'
init_dist(args)
print(f'global_rank: {args.global_rank}, local_rank: {args.local_rank}\n')