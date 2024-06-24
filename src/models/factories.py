from typing import Tuple

import torch
from tokenizers.tokenizers import Tokenizer
from warmup_scheduler import GradualWarmupScheduler

from src import datasets
from src.models import model_registry
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
from torch import nn
from fastxtend.optimizer.stableadam import StableAdam
from src.models import text_embedding_registry, vae_registry
from optimi.stableadamw import StableAdamW
from timm.optim import Lamb


def get_dataset(dataset_name: str, path: str, subset: str, size: int , crop_size: int, *args, **kwargs):
    return datasets.__dict__[dataset_name](path, subset, size, crop_size, *args, **kwargs)


def get_model(model_name: str, device: str, *args, **kwargs) -> nn.Module:
    model = model_registry.__dict__[model_name](*args, **kwargs)
    model.to(device)
    return model


def get_text_embedding(embedding_name: str, device: str, *args, **kwargs) -> Tuple[nn.Module, Tokenizer]:
    model, tokenizers = text_embedding_registry.__dict__[embedding_name](device, *args, **kwargs)
    model.eval().requires_grad_(False)
    return model, tokenizers


def get_latent_encoder(vae_name: str, weight_path: str, device: str, *args, **kwargs) -> nn.Module:
    vae = vae_registry.__dict__[vae_name](device=device, *args, **kwargs)
    checkpoint = torch.load(weight_path)
    vae.load_state_dict(checkpoint['state_dict'])  # , strict=False)
    vae.eval().requires_grad_(False)
    return vae


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