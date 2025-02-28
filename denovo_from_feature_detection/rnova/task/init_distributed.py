from datetime import timedelta

import torch.distributed as dist
import torch
import os

def initialize_distributed(server='cc'):
    if server == 'bsi':
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=180))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        return local_rank
    elif server == 'cc':
        ngpus_per_node = torch.cuda.device_count()
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_id = int(os.environ["SLURM_NODEID"])
        rank = node_id * ngpus_per_node + local_rank

        master_addr = os.environ["MASTER_ADDR"]
        init_method = f'tcp://{master_addr}:21457'

        world_size = int(os.environ["SLURM_JOB_NUM_NODES"]) * ngpus_per_node

        dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=rank, timeout=timedelta(minutes=180))
        torch.cuda.set_device(local_rank)
        dist.barrier()

        print(f'Rank {rank} initialized')

        return local_rank
    else:
        raise NotImplementedError