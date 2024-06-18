import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import wandb
import os
import shutil
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
import time

import webdataset as wds
from webdataset.handlers import warn_and_continue

from torch.distributed import init_process_group, destroy_process_group, new_group, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL
import torch.multiprocessing as mp

from torchtools.utils import Diffuzz, Diffuzz2
from torchtools.utils.diffusion2 import DDPMSampler
from torchtools.transforms import SmartCrop
from modules_experimental_1 import StageX
from utils import WebdatasetFilterHumans

# PARAMETERS
updates = 300000  # 500000
warmup_updates = 10000
batch_size = 128  # 1024 # 2048 # 4096
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 500 * grad_accum_steps
lr = 1e-4

dataset_path = "pipe:aws s3 cp s3://laion-west/humans-7M-with-blip-caps+aesthetics+nsfw/00000{1..5499}.tar -"
clip_image_model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
output_path = "../../output/experimental/exp1/"
checkpoint_path = "../../models/experimental/exp1.pt"
target = "v"

wandv_project = "wurstchen_experimental"
wandv_entity = "babbleberns"
wandb_run_name = "exp1d"
wandb_config = {
    "model_type": 'asymmetric [[2, 4], [36, 6]]',
    "target": f'{target}-target',
    "image_size": "256x256",
    "batch_size": batch_size,
    "warmup_updates": warmup_updates,
    "lr": lr,
    "description": "Simple clip2img unet on Humans-7m, preliminary experiments."
}

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    SmartCrop(256, randomize_p=0.3, randomize_q=0.2)
])
clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
])


def identity(x):
    return x


def get_scores(x_json):
    try:
        similarity = torch.tensor(x_json.get('similarity', 0.0))
        pwatermark = torch.tensor(x_json.get('pwatermark', 0.0))
        width, height = x_json.get('width', 1.0), x_json.get('height', 1.0)
        owidth, oheight = x_json.get('original_width', 1.0), x_json.get('original_height', 1.0)
        return similarity, pwatermark, torch.tensor(width / height), torch.tensor(min(owidth, oheight, width, height))
    except:
        return torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0)


def ddp_setup(rank, world_size, n_node, node_id):  # <--- DDP
    rk = int(os.environ.get("SLURM_PROCID"))
    print(f"Rank {rk} setting device to {rank}")
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rk, world_size=world_size * n_node,
        init_method="file:///fsx/home-pablo/src/wurstchen_experimental/dist_file_experimental_exp1",
    )
    print(f"[GPU {rk}] READY")


def train(gpu_id, world_size, n_nodes):
    node_id = int(os.environ["SLURM_PROCID"]) // world_size
    is_main_node = int(os.environ.get("SLURM_PROCID")) == 0
    ddp_setup(gpu_id, world_size, n_nodes, node_id)  # <--- DDP
    device = torch.device(gpu_id)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- PREPARE DATASET ---
    # PREPARE DATASET
    dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=warn_and_continue
    ).select(
        WebdatasetFilterHumans(min_size=16, max_pwatermark=0.5, unsafe_threshold=0.99)
    ).shuffle(690, handler=warn_and_continue).decode(
        "pilrgb", handler=warn_and_continue
    ).to_tuple(
        "jpg", "combined_txt", "json", handler=warn_and_continue
    ).map_tuple(
        transforms, identity, get_scores, handler=warn_and_continue
    )
    real_batch_size = batch_size // (world_size * n_nodes * grad_accum_steps)
    dataloader = DataLoader(dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True)

    if is_main_node:
        print("REAL BATCH SIZE / DEVICE:", real_batch_size)

    # --- PREPARE MODELS ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) if os.path.exists(checkpoint_path) else None
    except RuntimeError as e:
        if os.path.exists(f"{checkpoint_path}.bak"):
            os.remove(checkpoint_path)
            shutil.copyfile(f"{checkpoint_path}.bak", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise e

    # - utils -
    # diffuzz = Diffuzz(device=device)
    if target == 'e':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0.0001, 0.9999))
        diffuzz_sampler = DDPMSampler(diffuzz, mode=target)
    elif target == 'v':
        diffuzz = Diffuzz2(device=device, scaler=1, clamp_range=(0, 1 - 1e-7))

    # - vae -
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    vae.eval().requires_grad_(False)

    # - CLIP text encoder
    # clip_text_model = CLIPTextModel.from_pretrained(clip_text_model_name).to(device).eval().requires_grad_(False)
    # clip_tokenizer = AutoTokenizer.from_pretrained(clip_text_model_name)
    clip_image_model = CLIPVisionModel.from_pretrained(clip_image_model_name).to(device).eval().requires_grad_(False)

    # - Diffusive Imagination Combinatrainer, a.k.a. Risotto -
    # generator = StageX().to(device)
    generator = StageX(blocks=[[2, 4], [36, 6]], c_hidden=[512, 1024], nhead=[-1, 16], level_config=['CT', 'CTA'],
                       dropout=[0, 0.1]).to(device)
    if checkpoint is not None:
        generator.load_state_dict(checkpoint['state_dict'])
    generator = DDP(generator, device_ids=[gpu_id], output_device=device)  # <--- DDP

    if is_main_node:  # <--- DDP
        print("Num trainable params:", sum(p.numel() for p in generator.parameters() if p.requires_grad))

    # - SETUP WANDB -
    if is_main_node:  # <--- DDP
        run_id = checkpoint['wandb_run_id'] if checkpoint is not None else wandb.util.generate_id()
        wandb.init(project=wandv_project, name=wandb_run_name, entity=wandv_entity, id=run_id, resume="allow",
                   config=wandb_config)

    # SETUP OPTIMIZER, SCHEDULER & CRITERION
    optimizer = optim.AdamW(generator.parameters(), lr=lr)  # , eps=1e-7, betas=(0.9, 0.95))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Failed loading optimizer, skipping...")
        scheduler.last_epoch = checkpoint['scheduler_last_step']

    start_iter = 1
    grad_norm = torch.tensor(0, device=device)
    if is_main_node:  # <--- DDP
        if checkpoint is not None:
            start_iter = checkpoint['scheduler_last_step'] * grad_accum_steps + 1
            print("RESUMING TRAINING FROM ITER ", start_iter)
            wandb.alert(title=f"Training {run_id} resumed",
                        text=f"Training {run_id} resumed from step {checkpoint['scheduler_last_step']}")
        else:
            wandb.alert(title=f"Training {run_id} started", text=f"Training {run_id} started")

    ema_loss = None
    if checkpoint is not None:
        ema_loss = checkpoint['metrics']['ema_loss']

    if checkpoint is not None:
        del checkpoint  # cleanup memory
        torch.cuda.empty_cache()

        # -------------- START TRAINING --------------
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters + 1)) if is_main_node else range(start_iter, max_iters + 1)  # <--- DDP
    generator.train()
    for it in pbar:
        images, captions, scores = next(dataloader_iterator)
        images = images.to(device)
        _, _, _, sca = scores
        sca = (sca.to(device) / min(images.size(-1), images.size(-2))).clamp(max=1)

        if np.random.rand() < 0.05:
            sca = torch.zeros_like(sca)

        with torch.no_grad():
            clip_image_embeddings = torch.zeros(images.size(0), 1, 1280).to(device)
            rand_idx = np.random.rand(clip_image_embeddings.size(0)) > 0.05
            if rand_idx.any():
                clip_image_embeddings[rand_idx] = clip_image_model(
                    clip_preprocess(images[rand_idx])).pooler_output.unsqueeze(1)

            t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
            latents = vae.encode(images).latent_dist.mode()
            noised_latents, noise = diffuzz.diffuse(latents, t)
            target_v = diffuzz.get_v(latents, t, noise)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if target == 'e':
                pred_e = diffuzz.noise_from_v(noised_latents, pred_v, t)
                loss = nn.functional.mse_loss(pred_e, noise, reduction='none').mean(dim=[1, 2, 3])
                loss_adjusted = (loss * diffuzz.p2_weight(t)).mean() / grad_accum_steps
            elif target == 'v':
                pred_v = generator(noised_latents, t, clip_image_embeddings, sca=sca)
                loss = nn.functional.mse_loss(pred_v, target_v, reduction='none').mean(dim=[1, 2, 3])
                loss_adjusted = loss.mean() / grad_accum_steps

        if it % grad_accum_steps == 0 or it == max_iters:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with generator.no_sync():
                loss_adjusted.backward()

        if not np.isnan(loss.mean().item()):
            ema_loss = loss.mean().item() if ema_loss is None else ema_loss * 0.99 + loss.mean().item() * 0.01

        if is_main_node:  # <--- DDP
            pbar.set_postfix({
                'bs': images.size(0),
                'loss': ema_loss,
                'raw_loss': loss.mean().item(),
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })
            wandb.log({
                'loss': ema_loss,
                'raw_loss': loss.mean().item(),
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'total_steps': scheduler.last_epoch,
            })

            if np.isnan(loss.mean().item()) or np.isnan(grad_norm.item()):
                wandb.alert(
                    title=f"NaN value encountered in training run {run_id}",
                    text=f"Loss {loss.mean().item()} - Grad Norm {grad_norm.item()}. Run {run_id}",
                    wait_duration=60 * 30
                )

        if is_main_node and (
                scheduler.last_epoch == 1 or scheduler.last_epoch % print_every == 0 or it == max_iters):  # <--- DDP
            if np.isnan(loss.mean().item()):
                tqdm.write(f"Skipping sampling & checkpoint because the loss is NaN")
                wandb.alert(title=f"Skipping sampling & checkpoint for training run {run_id}",
                            text=f"Skipping sampling & checkpoint at {scheduler.last_epoch} for training run {run_id} iters because loss is NaN")
            else:
                tqdm.write(f"ITER {it}/{max_iters} - loss {ema_loss}")

                generator.eval()
                images, captions, scores = next(dataloader_iterator)
                while images.size(0) < 8:  # 8
                    _images, _captions, _scores = next(dataloader_iterator)
                    images = torch.cat([images, _images], dim=0)
                    captions += _captions
                images, captions = images[:8].to(device), captions[:8]
                sca = torch.ones(images.size(0), device=device)
                with torch.no_grad():
                    # clip stuff
                    clip_image_embeddings = clip_image_model(clip_preprocess(images)).pooler_output.unsqueeze(1)
                    clip_image_embeddings_uncond = torch.zeros_like(clip_image_embeddings)
                    # ---

                    t = (1 - torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
                    latents = vae.encode(images).latent_dist.mode()
                    noised_latents, noise = diffuzz.diffuse(latents, t)

                    if target == 'e':
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            pred_e = generator(noised_latents, t, clip_image_embeddings, sca=sca)
                        pred = \
                        diffuzz.undiffuse(noised_latents, t, torch.zeros_like(t), pred_e, sampler=diffuzz_sampler)[0]
                    elif target == 'v':
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            pred_v = generator(noised_latents, t, clip_image_embeddings, sca=sca)
                        pred = diffuzz.x0_from_v(noised_latents, pred_v, t)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        *_, (sampled, _, _) = diffuzz.sample(generator, {
                            'clip': clip_image_embeddings, "sca": sca,
                        }, latents.shape, unconditional_inputs={
                            'clip': clip_image_embeddings_uncond, "sca": sca,
                        }, cfg=7, sample_mode=target)

                    noised_images = vae.decode(noised_latents).sample.clamp(0, 1)
                    pred_images = vae.decode(pred).sample.clamp(0, 1)
                    sampled_images = vae.decode(sampled).sample.clamp(0, 1)
                generator.train()

                torchvision.utils.save_image(torch.cat([
                    torch.cat([i for i in images.cpu()], dim=-1),
                    torch.cat([i for i in noised_images.cpu()], dim=-1),
                    torch.cat([i for i in pred_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images.cpu()], dim=-1)
                ], dim=-2), f'{output_path}{it:06d}.jpg')

                try:
                    os.remove(f"{checkpoint_path}.bak")
                except OSError:
                    pass

                try:
                    os.rename(checkpoint_path, f"{checkpoint_path}.bak")
                except OSError:
                    pass

                torch.save({
                    'state_dict': generator.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_last_step': scheduler.last_epoch,
                    'iter': it,
                    'metrics': {
                        'ema_loss': ema_loss,
                    },
                    'wandb_run_id': run_id,
                }, checkpoint_path)

                if scheduler.last_epoch % 20000 == 0:
                    chkpt_extension = f"_{scheduler.last_epoch // 1000}k.pt"
                    shutil.copyfile(checkpoint_path, checkpoint_path.replace(".pt", chkpt_extension))

                log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(images[i])] for i in
                            range(len(images))]
                log_table = wandb.Table(data=log_data, columns=["Captions", "Sampled", "Orig"])
                wandb.log({"Log": log_table})

    destroy_process_group()  # <--- DDP


if __name__ == '__main__':
    print("Launching Script")
    world_size = torch.cuda.device_count()
    n_node = 1
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print("Detecting", torch.cuda.device_count(), "GPUs for each of the", n_node, "nodes")
    train(local_rank, world_size, n_node)



