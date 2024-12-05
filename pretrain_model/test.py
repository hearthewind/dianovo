import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.random.randn(500, 10)
        self.label = np.sum(self.data, axis=1)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    source, target = zip(*batch)
    source = torch.Tensor(np.array(source))
    target = torch.Tensor(np.array(target))
    return source, target

def move2device(var):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    return var.to(device)

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

def get_model():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    model = torch.nn.Linear(10, 1).to(device)
    return DDP(model, device_ids=[local_rank], output_device=local_rank)

def main(total_epochs, batch_size, learning_rate):
    local_rank = int(os.environ["LOCAL_RANK"])

    ddp_setup()
    model = get_model()

    train_dataset = TrainDataset()
    train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             sampler=DistributedSampler(train_dataset, shuffle=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    for epoch in range(total_epochs):
        train_data.sampler.set_epoch(epoch)
        for source, target in train_data:
            source = move2device(source)
            target = move2device(target)
            optimizer.zero_grad()
            with autocast():
                output = model(source)
                loss = F.mse_loss(output, target.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if local_rank == 0:
            print(f"Epoch {epoch} loss {loss.item()}")

    destroy_process_group()


if __name__ == '__main__':
    main(1000, 16, 0.001)

# how to run the program?
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 test.py
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:12378 --nnodes=1 --nproc_per_node=2 test.py
