import os
import torch

def get_available_device():
    """Search for (preferred) devices, in decreasing preference:
    ['cuda', 'mps', 'cpu']
    """
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:  # default to cpu
        device = torch.device('cpu')
    return device

def distribute_model(model, device):
    # if device is distributed cuda, return DDP(model, ...)
    # else, do nothing
    return torch.nn.parallel.\
        DistributedDataParallel(model, device_ids=[device], output_device=device)

def get_model_state_dict(model, device):
    # if distributed return model.module.state_dict()
    # else
    return model.state_dict()

def save_process(device):
    retval = True
    if isinstance(device, int):
        if device > 0:
            retval = False
    return retval
        
def ddp_setup(rank: int, world_size: int):
    """
    Copy/paste from pytorch for supporting model distribution
    across resources.

   Args:
      rank: Unique identifier of each process
      world_size: Total number of processes

    backend in ['nccl', 'gloo'], from pytorch documentation,
      'nccl' is for distributing across GPU, 'gloo' for CPU
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.cuda.set_device(rank)
    print(f"setup device{rank}")
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size)

