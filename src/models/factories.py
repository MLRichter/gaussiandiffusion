from warmup_scheduler import GradualWarmupScheduler

from src import datasets
from src.models import model_registry
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
from torch import nn
from fastxtend.optimizer.stableadam import StableAdam
from src.models import text_encoder_registry, vae_registry
from optimi.stableadamw import StableAdamW
from timm.optim import Lamb


def get_dataset(dataset_name: str, path: str, subset: str, size: int , crop_size: int, *args, **kwargs):
    return datasets.__dict__[dataset_name](path, subset, size, crop_size, *args, **kwargs)


def get_model(model_name: str, device: str, *args, **kwargs) -> nn.Module:
    return model_registry.__dict__[model_name](device, *args, **kwargs)


def get_text_embedding(embedding_name: str, device: str, *args, **kwargs) -> nn.Module:
    return text_encoder_registry.__dict__[embedding_name](device, *args, **kwargs)


def get_latent_encoder(vae_name: str, weight_path: str, device: str, *args, **kwargs) -> nn.Module:
    return vae_registry.__dict__[vae_name](weight_pat=weight_path, device=device, *args, **kwargs)


def get_optimizer(opt: str, model: nn.Module, lr: float, *args, **kwargs) -> optim.Optimizer:
    if "adam" == opt.lower():
        optimizer = optim.Adam(model.parameters(), lr=lr, *args, **kwargs)
    elif "adamw" == opt.lower():
        optimizer = optim.AdamW(model.parameters(), lr=lr, *args, **kwargs)
    elif "stableadam" == opt.lower():
        optimizer = StableAdamW(model.parameters(), lr=lr, *args, **kwargs)
    elif "lamb" == opt.lower():
        optimizer = Lamb(model.parameters(), lr=lr, *args, **kwargs)
    else:
        raise ValueError(f"Unknown Optimizer {opt}")
    return optimizer


def get_scheduler(sched: str, optim: optim.Optimizer, warmup_updates: int, total_updates: int, *args, **kwargs) -> optim.lr_scheduler.LRScheduler:
    if "gradual_warmup" == sched.lower():
        scheduler = GradualWarmupScheduler(optim, multiplier=1, total_epoch=warmup_updates, *args, **kwargs)
    elif "cosine" == sched.lower():
        scheduler = CosineAnnealingLR(optim, T_max=total_updates, *args, **kwargs)
    else:
        raise ValueError(f"Unknown Scheduler {sched}")
    return scheduler