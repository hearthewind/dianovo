import torch

def get_lr_scheduler(optimizer, num_warmup_steps):
    """
    Create a learning rate scheduler for a warmup phase.

    :param optimizer: Optimizer linked to model parameters.
    :param num_warmup_steps: Number of steps for the warmup phase.
    :return: Learning rate scheduler.
    """

    assert num_warmup_steps > 1

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(num_warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
