import os

import torch
import torch.distributed as dist

if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ["SLURM_LOCALID"])
    node_id = int(os.environ["SLURM_NODEID"])
    rank = node_id * ngpus_per_node + local_rank

    master_addr = os.environ["MASTER_ADDR"]
    init_method = f'tcp://{master_addr}:21452'

    world_size = int(os.environ["SLURM_JOB_NUM_NODES"]) * ngpus_per_node

    print(f"Rank {rank} starting initialization")
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} initialized, waiting for barrier")
    dist.barrier()
    print(f'Rank {rank} synced')
    dist.destroy_process_group()
    print(f'Rank {rank} destroyed')