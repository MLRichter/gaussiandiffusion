import torch
import os
from pathlib import Path

from torch._C._distributed_c10d import TCPStore
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.target_schedulers import SNR_Scheduler


def adjust_print_freq(print_every: int, update_freq: int) -> int:
    if print_every % update_freq != 0:
        divisble_print_every = ((print_every // update_freq) * (update_freq+1))
        print("Adjusting print every to", divisble_print_every)
        print((divisble_print_every % update_freq))
        assert (divisble_print_every % update_freq) == 0.0
        return int(divisble_print_every)
    else:
        print("No adjustment to print frequency is necessary")
        return print_every


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_root(rank: int) -> bool:
    return rank == 0


def is_distributed(world_size: int) -> bool:
    return world_size != 1


def init_distributed_mode(
        dist_on_itp: bool, world_size: int = None, tcp=False, dist_url: str = 'env://'):

    if dist_on_itp:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['WORLD_SIZE'] = str(world_size)
    else:
        print('Not using distributed mode')
        return 1, 0, 0, None, None

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    print("Training with World Size:", world_size, "using",  dist_backend)
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, dist_url, gpu), flush=True)
    if tcp:
        if "MASTER_ADDR" not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
        print('| creating TCP-store at {}, port: {}, rank: {}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], rank))
        store = TCPStore(host_name=os.environ['MASTER_ADDR'], port=int(os.environ['MASTER_PORT']), world_size=world_size, is_master=rank == 0)
        torch.distributed.init_process_group(backend=dist_backend, store=store,
                                             world_size=world_size, rank=rank)
    else:
        import time
        if rank >= 0:
            sleeptime = (world_size+1+-rank)
            print('| creating process group, waiting for:', sleeptime)

            time.sleep(sleeptime)
        print('| creating process group')
        torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                             world_size=world_size, rank=rank)
    print('| creating subgroups for generator and discriminator')
    model_group = torch.distributed.new_group(ranks=list(range(world_size)))
    disc_group = torch.distributed.new_group(ranks=list(range(world_size)))
    print("| setup complete, waiting for all node-processes to finish")
    torch.distributed.barrier()
    print("| done")
    setup_for_distributed(rank == 0)
    return world_size, rank, gpu,  model_group, disc_group


def check_autoresume_possible(log_dir: str):
    return (Path(log_dir) / "models" / "model.pt").exists()


def autoresume(log_dir: str,
               model: Module,
               optim: Optimizer,
               desc_optim: Optimizer,
               sched: LRScheduler,
               desc_sched: LRScheduler,
               scaler_m: torch.cuda.amp.GradScaler,
               scaler_d: torch.cuda.amp.GradScaler,
               snr_sched: SNR_Scheduler
               ) -> int:
    weight_path = Path(log_dir) / "models" / "model.pt"
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])  # , strict=False)
    start_iter = checkpoint['last_iter'] + 1
    optim.load_state_dict(checkpoint['optim_state_dict'])
    sched.last_epoch = checkpoint['last_iter']
    scaler_m.load_state_dict(checkpoint["scaler_m"])
    print("Resuming from", weight_path, "at", start_iter, "steps")
    return start_iter


def load_model_for_finetuneing(log_dir: str, model: Module, disc: Module, finetune_disc: bool = True) -> int:
    weight_path = Path(log_dir) / "models" / "model.pt"
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])  # , strict=False)
    if finetune_disc:
        disc.load_state_dict(checkpoint['disc_state_dict'])  # , strict=False)
    print("Finetuning from", weight_path, "(model and discriminator)" if finetune_disc else "(model only)")

def create_log_dir(log_dir: str) -> None:
    log_path = Path(log_dir)
    model_path = log_path / "models"
    image_path = log_path / "images"
    model_path.mkdir(parents=True, exist_ok=True)
    image_path.mkdir(parents=True, exist_ok=True)