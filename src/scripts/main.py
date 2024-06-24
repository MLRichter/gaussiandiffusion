import argparse

import torch

from torch.utils.data import DataLoader

from torchtools.utils import Diffuzz2
from torchtools.utils.diffusion2 import DDPMSampler

from src.logging_utils import JsonLogger, JsonWandbLogger, DummyLogger
from src.models.factories import get_model, get_dataset, get_optimizer, get_scheduler, \
    get_latent_encoder, get_text_embedding

from src.training.train_txt_guided_ldm import train
from src.training.utils import create_log_dir, check_autoresume_possible, autoresume, adjust_print_freq, \
    load_model_for_finetuneing
from src.training.utils import is_distributed, init_distributed_mode




def main():
    parser = argparse.ArgumentParser(description='Training configuration')

    # basic training parameters
    parser.add_argument('--dataset_path', default='../../../coco2017/', type=str, help='Path to dataset')
    parser.add_argument('--log_dir', default='../../output/test_gas2', type=str, help='Directory for logging')
    parser.add_argument('--start_iter', default=None, type=int, help='Iteration to start training from')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--update_freq', default=4, type=int, help='Update frequency')
    parser.add_argument('--total_updates', default=100000, type=int, help='Total number of updates')
    parser.add_argument('--warmup_updates', default=100000, type=int, help='Number of warmup updates')

    # resolution
    parser.add_argument('--train_size', default=256, type=int, help='Training image size')
    parser.add_argument('--train_crop_size', default=128, type=int, help='Training crop size')
    parser.add_argument('--val_size', default=256, type=int, help='Validation image size')
    parser.add_argument('--val_crop_size', default=128, type=int, help='Validation crop size')

    # loss weights
    # model selection
    parser.add_argument('--model_name', default='standard_ctms_v1', type=str, help='Model name')
    parser.add_argument('--latent_encoder_name', default='standard_ctms_v1', type=str, help='Model name')
    parser.add_argument('--latent_encoder_weights', default='standard_ctms_v1', type=str, help='Model name')
    parser.add_argument('--text_encoder_name', default='standard_ctms_v1', type=str, help='Model name')
    parser.add_argument('--loss_target', default='v', type=str, help='loss target')
    parser.add_argument('--loss_weighting', default='p2', type=str, help='loss target')

    # scheduler and optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer for the model')
    parser.add_argument('--scheduler', default='gradual_warmup', type=str, help='LR Scheduler')

    # dataset
    parser.add_argument('--dataset_name', default='coco_dataset', type=str, help='Dataset name')
    parser.add_argument('--seed', default=30071993, type=int, help='Random seed')

    # general training settings
    parser.add_argument('--print_every', default=6000, type=int, help='Print frequency')
    parser.add_argument('--wandb', action='store_true', help='enable logging using wandb')
    parser.add_argument('--finetune', default=None, type=str, help='First loads weights from the specified checkpoint. Does NOT overwrite autoresume')
    parser.add_argument('--autorestart', default=True, type=bool, help='Auto-restart')

    # hardware specific stuff
    parser.add_argument('--use_32', action='store_true', help='disable mixed precision')
    parser.add_argument('--use_bfloat16', action='store_true', help='Use bfloat16')
    parser.add_argument('--use_float16', dest='feature', action='store_false')

    parser.add_argument('--workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--disable_grad_scaling', action='store_true', help='disables grad scaling')


    # distributed stuff
    parser.add_argument('--world_size', default=1, type=int, help='World size for distributed training')
    parser.add_argument('--dist_on_itp', default=False, type=bool, help='Distributed on ITP')
    parser.add_argument('--tcp', default=False, type=bool, help='Use TCP')
    parser.add_argument('--dist_url', default='env://', type=str, help='URL for distributed training')


    args = parser.parse_args()

    # Print all configurations as the first thing in the main function
    print("Configuration Parameters:")
    full_config_dict = {}
    for arg in vars(args):
        print(f"| \t{arg}: {getattr(args, arg)}")
        full_config_dict[arg] = getattr(args, arg)


    dataset_path = args.dataset_path #"../../../coco2017/"
    log_dir = args.log_dir #"../../output/test_gas2"
    start_iter = args.start_iter #None
    lr: float = args.lr #1e-4
    batch_size: int = args.batch_size #2
    update_freq: int = args.update_freq #2
    total_updates = args.total_updates #100000
    warmup_updates = args.warmup_updates #100000
    train_size: int = args.train_size #256
    train_crop_size: int = args.train_crop_size #128

    loss_target = args.loss_target
    loss_weighting = args.loss_weighting

    val_size: int = args.val_size #256
    val_crop_size: int = args.val_crop_size #256
    use_bfloat16: bool = args.use_bfloat16 #True

    workers: int = args.workers #4
    model_name: str = args.model_name #"standard_ctms_v1"
    latent_encoder_name = args.latent_encoder_name
    latent_encoder_weights = args.latent_encoder_weights
    text_encoder_name = args.text_encoder_name

    dataset_name: str = args.dataset_name #"coco_dataset"
    optimizer: str = args.optimizer
    scheduler: str = args.scheduler
    seed: int = args.seed #30071993

    print_every: int = args.print_every #3000

    device: str = args.device
    finetune: str = args.finetune
    autorestart = args.autorestart #True
    world_size: int = args.world_size #1
    dist_on_itp: bool = args.dist_on_itp #False
    tcp: bool = args.tcp #False
    dist_url: str = args.dist_url #'env://',
    disable_grad_scaling: bool = args.disable_grad_scaling


    # TODO: Rebuild for Diffuzz impl
    if loss_target == 'e':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0.0001, 0.9999))
        diffuzz_sampler = DDPMSampler(diffuzz, mode=loss_target)
    elif loss_target == 'v':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0, 1 - 1e-7))
        diffuzz_sampler = None
    else:
        raise ValueError(f"{loss_target} loss_target not recognized")

    print_every = adjust_print_freq(print_every=print_every, update_freq=update_freq)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world_size, rank, local_rank, m_group, d_group = init_distributed_mode(dist_on_itp=dist_on_itp, world_size=world_size, tcp=tcp, dist_url=dist_url)


    latent_encoder = get_latent_encoder(vae_name=latent_encoder_name, device=device, weight_path=latent_encoder_weights)
    model = get_model(model_name=model_name, device=device)
    txt_tokenizer, txt_model = get_text_embedding(embedding_name=text_encoder_name, device=device)

    dataset_train = get_dataset(dataset_name=dataset_name, path=dataset_path,
                                subset="train", size=train_size, crop_size=train_size)

    dataset_val = get_dataset(dataset_name=dataset_name, path=dataset_path,
                                subset="val", size=val_size, crop_size=val_size)


    optimizer = get_optimizer(optimizer, model, lr=lr)
    scheduler = get_scheduler(scheduler, optimizer, warmup_updates=warmup_updates, total_updates=total_updates)
    scaler_m = torch.cuda.amp.GradScaler(enabled=not disable_grad_scaling)

    if is_distributed(world_size=world_size):
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
        train_sample_kwargs = {"sampler": sampler_train}
    else:
        train_sample_kwargs = {"shuffle" : True}

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=workers, pin_memory=True, **train_sample_kwargs)
    dataloader_val = DataLoader(dataset_val, batch_size=10, num_workers=workers, pin_memory=True, shuffle=False)

    model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_batch_size = batch_size * world_size * update_freq

    create_log_dir(log_dir)
    if finetune is not None:
        load_model_for_finetuneing(finetune, model=model)
    if check_autoresume_possible(log_dir) and autorestart:
        start_iter = autoresume(log_dir, model=model, optim=optimizer,
                                sched=scheduler, scaler_m=scaler_m)
        print("Resuming from", start_iter)
    else:
        print("No resuming checkpoint was found, starting from scratch")
    if args.wandb and rank == 0:
        logger = JsonWandbLogger(save_path=log_dir, **full_config_dict)
    elif rank == 0:
        logger = JsonLogger(save_path=log_dir, **full_config_dict)
    else:
        logger = DummyLogger()

    model_without_ddp = model
    if args.wandb and rank == 0:
        import wandb
        wandb.watch(model)
    if is_distributed(world_size):
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], process_group=m_group, broadcast_buffers=True, find_unused_parameters=True)


    print()
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', model_n_parameters)
    print("LR = %.8f" % lr)
    print("(Total) Batch size = %d" % total_batch_size)
    print("(Local) Batch size = %d" % batch_size)
    print("Update frequent = %d" % update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of total training samples = %d" % (total_updates * total_batch_size))
    print("Number of total training batches = %d" % total_updates)


    train(
        latent_encoder=latent_encoder,
        diffusion_model=model,
        text_model=txt_model,
        tokenizer=txt_tokenizer,
        diffuzz_sampler=diffuzz_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader_train,
        val_dataloader=dataloader_val,
        diffuzz=diffuzz,
        logger=logger,
        crop=train_crop_size,
        val_crop=val_crop_size,
        input_dim=train_crop_size,
        log_path=log_dir,
        loss_weighting=loss_weighting,
        device=device,
        loss_target=loss_target,
        dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
        update_freq=update_freq,
        start_iter=start_iter,
        total_updates=total_updates,
        print_every=print_every,
        rank=rank
    )


if __name__ == '__main__':
    main()