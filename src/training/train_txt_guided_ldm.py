import os

import torch
import torchvision
from tokenizers import Tokenizer
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchtools.utils.diffusion2 import DDPMSampler
from torchvision.transforms import RandomCrop
from torchtools.utils import Diffuzz2
from tqdm import tqdm
import time
from src.logging_utils import Logger
from src.training.utils import is_root
from transformers import AutoTokenizer, CLIPTextModel
import numpy as np
import lpips


def stratified_random_sampling(num_samples):
    # Divide the interval [0, 1] into `num_samples` strata
    strata_edges = torch.linspace(0, 1, num_samples + 1)

    # Sample one random point within each stratum and perturb it
    samples = []
    for i in range(num_samples):
        start, end = strata_edges[i], strata_edges[i + 1]
        sample = torch.distributions.Uniform(start, end).sample()
        samples.append(sample)

    return torch.tensor(samples)

def integer_sampling(num_samples, device):
    timesteps = torch.randint(0, 1000, (num_samples,), device=device)
    timesteps = timesteps.long()
    return timesteps

def train(
        latent_encoder: Module,
        diffusion_model: Module,
        text_model: CLIPTextModel,
        tokenizer: AutoTokenizer,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        diffuzz_sampler: DDPMSampler,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        diffuzz: Diffuzz2,
        logger: Logger,
        crop: RandomCrop,
        val_crop: int,
        input_dim: int,
        log_path: str,
        loss_weighting: str = "p2",
        device: str = "cuda",
        loss_target: str = 'v',
        dtype=torch.bfloat16,
        update_freq: int = 1,
        start_iter: int = 1,
        total_updates: int = 1000000,  # 1M Updates
        print_every: int = 2000,
        rank: int = 0,
):
    latent_encoder.to(device)
    diffusion_model.to(device)
    text_model.to(device)
    ema_loss, ema_mse_loss, ema_g_loss, ema_var_loss, ema_lpips_loss = None, None, None, None, None
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, total_updates), initial=start_iter if start_iter == 1 else start_iter+1, total=total_updates)
    time.sleep(1)
    diffusion_model.train()
    print("Using", dtype, "for mixed precision")


    total_gradient_updates: int = 0
    for step, i in enumerate(pbar):

        try:
            images, captions = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(dataloader)
            images, captions = next(dataloader_iterator)
        images = images.to(device)
        images = crop(images)  # train on 128x128 crops of the images

        # TODO: Implementing Training

        # get text embeddings
        with torch.no_grad():
            rand_idx = np.random.rand(len(captions)) > 0.05
            captions = [str(c) if keep else "" for c, keep in zip(captions, rand_idx)]
            clip_tokens = tokenizer(captions, truncation=True, padding="max_length",
                                    max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            clip_text_embeddings = text_model(**clip_tokens, output_hidden_states=True).last_hidden_state

            #t = (1-torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
            #t = stratified_random_sampling(images.size(0)).to(device).add(0.001).clamp(0.001, 1.0)
            t = integer_sampling(images.size(0), device)
            t = diffuzz.scale_t(t, input_dim/256)
            latents = latent_encoder.encode(images)[0]
            noised_latents, noise = diffuzz.diffuse(latents, t)
            target_v = diffuzz.get_v(latents, t, noise)

        if loss_target == 'e':
            if loss_weighting == 'p2':
                loss_w = diffuzz.p2_weight(t)
            else:
                loss_w = 1.0
            pred_e = diffusion_model(noised_latents, t, clip_text_embeddings, return_dict=False)
            loss = nn.functional.mse_loss(pred_e, noise, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_w).mean() / update_freq
        elif loss_target == 'v':
            if loss_weighting == 'truncatedSNR':
                loss_w = diffuzz.truncated_snr_weight(t, min=1.0) / (diffuzz.truncated_snr_weight(t, min=None) + 1)
            else:
                loss_w = 1.0
            pred_v = diffusion_model(noised_latents, t, clip_text_embeddings, return_dict=False)
            loss = nn.functional.mse_loss(pred_v, target_v, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_w).mean() / update_freq
        else:
            raise NotImplementedError()

        if i % update_freq == 0:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with diffusion_model.no_sync():
                loss_adjusted.backward()

        if not np.isnan(loss.mean().item()):
            ema_loss = loss.mean().item() if ema_loss is None else ema_loss * 0.99 + loss.mean().item() * 0.01
        grad_norm = nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)


        # Logging
        metrics = {
            'bs': images.size(0),
            'gas': update_freq,
            'tbs': images.size(0) * update_freq,
            'tgu': total_gradient_updates,
            'loss': ema_loss / images.size(0),
            'grad_norm': grad_norm.item() / images.size(0),
            'lr': optimizer.param_groups[0]['lr'],
            'total_steps': scheduler.last_epoch,
        }
        pbar.set_postfix(metrics)

        if (i == 0 or i % print_every == 0) and is_root(rank):





            # TODO: Generate Images
            validate(
                latent_encoder=latent_encoder,
                diffusion_model=diffusion_model,
                txt_tokenizer=tokenizer,
                txt_model=text_model,
                diffuzz=diffuzz,
                loss_target=loss_target,
                dtype=dtype,
                val_dataloader=val_dataloader,
                val_crop=val_crop,
                device=device,
                diffuzz_sampler=diffuzz_sampler,
                logger=logger,
                itr=i
            )

            logger.log_metrics(metrics=metrics)


def validate(latent_encoder: nn.Module,
             diffusion_model: nn.Module,
             txt_tokenizer: AutoTokenizer,
             txt_model: CLIPTextModel,
             diffuzz: Diffuzz2,
             loss_target: str,
             dtype,
             val_dataloader: DataLoader,
             val_crop: int,
             device: str,
             diffuzz_sampler: DDPMSampler,
             logger: Logger,
             itr: int,
             num_collages: int = 10
             ):
    lpips_criterion = lpips.LPIPS(net='vgg').to(device)
    diffusion_model.eval()
    # TODO: Validation code goes here
    pbar = tqdm(range(len(val_dataloader)), "Evaluating")
    losses = []
    lpips_losses = []
    with torch.no_grad():
        for step, _ in enumerate(pbar):

            try:
                images, captions = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(val_dataloader)
                images, captions = next(dataloader_iterator)

            images = images.to(device)

            # clip stuff
            # ---
            rand_idx = np.random.rand(len(captions)) > 0.05
            captions = [str(c) if keep else "" for c, keep in zip(captions, rand_idx)]
            clip_tokens = txt_tokenizer(captions, truncation=True, padding="max_length",
                                    max_length=txt_tokenizer.model_max_length, return_tensors="pt").to(device)
            clip_text_embeddings = txt_model(**clip_tokens, output_hidden_states=True).last_hidden_state

            clip_tokens_uncond = txt_tokenizer([""]*len(captions), truncation=True, padding="max_length",
                                        max_length=txt_tokenizer.model_max_length, return_tensors="pt").to(device)
            clip_text_embeddings_uncond = txt_model(**clip_tokens_uncond, output_hidden_states=True).last_hidden_state

            t = (1-torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
            t = diffuzz.scale_t(t, val_crop/256)
            latents = latent_encoder.encode(images)[0]
            noised_latents, noise = diffuzz.diffuse(latents, t)
            if loss_target == 'e':
                with torch.cuda.amp.autocast(dtype=dtype):
                    pred_e = diffusion_model(noised_latents, t, clip_text_embeddings, return_dict=False)

                pred = diffuzz.undiffuse(noised_latents, t, torch.zeros_like(t), pred_e, sampler=diffuzz_sampler)[0]
            elif loss_target == 'v':
                with torch.cuda.amp.autocast(dtype=dtype):
                    pred_v = diffusion_model(noised_latents, t, clip_text_embeddings, return_dict=False)

                pred = diffuzz.x0_from_v(noised_latents, pred_v, t)

            with torch.cuda.amp.autocast(dtype=dtype):
                result_cfg_15 = list(diffuzz.sample(diffusion_model, {
                    'encoder_hidden_states': clip_text_embeddings, 'return_dict': False
                }, latents.shape, unconditional_inputs={
                    'encoder_hidden_states': clip_text_embeddings_uncond, 'return_dict': False
                }, cfg=1.5))[-1][0]

                #print("DEBUG")
                #print(type(result_cfg_15), result_cfg_15)

                result_cfg_15_uncond = list(diffuzz.sample(diffusion_model, {
                    'encoder_hidden_states': clip_text_embeddings_uncond, 'return_dict': False
                }, latents.shape, unconditional_inputs={
                    'encoder_hidden_states': clip_text_embeddings_uncond, 'return_dict': False
                }, cfg=1.5))[-1][0]

                result_cfg_7 = list(diffuzz.sample(diffusion_model, {
                    'encoder_hidden_states': clip_text_embeddings, 'return_dict': False
                }, latents.shape, unconditional_inputs={
                    'encoder_hidden_states': clip_text_embeddings_uncond, 'return_dict': False
                }, cfg=7))[-1][0]


                noised_images = torch.cat(
                    [latent_encoder
                     .decode
                     (noised_latents[i:i + 1])
                     .clamp(0, 1) for i in range(len(noised_latents))], dim=0)

                pred_images = torch.cat(
                    [latent_encoder
                     .decode
                     (pred[i:i + 1])
                     .clamp(0, 1) for i in range(len(pred))], dim=0)
                
                if step <= num_collages:
                    cfg_15_images = torch.cat(
                        [latent_encoder
                        .decode(result_cfg_15[i:i + 1])
                        .clamp(0, 1) for i in range(len(result_cfg_15))], dim=0)

                    cfg_15_uncond_images = torch.cat(
                        [latent_encoder
                        .decode
                        (result_cfg_15_uncond[i:i + 1])
                        .clamp(0, 1) for i in range(len(result_cfg_15))], dim=0)

                    cfg_7_images = torch.cat(
                        [latent_encoder
                        .decode
                        (result_cfg_7[i:i + 1])
                        .clamp(0, 1) for i in range(len(result_cfg_15))], dim=0)

                
                    img = torch.cat([
                        torch.cat([i for i in images.cpu()], dim=-1),
                        torch.cat([i for i in noised_images.cpu()], dim=-1),
                        torch.cat([i for i in pred_images.cpu()], dim=-1),
                        torch.cat([i for i in cfg_15_images.cpu()], dim=-1),
                        torch.cat([i for i in cfg_15_uncond_images.cpu()], dim=-1),
                        torch.cat([i for i in cfg_7_images.cpu()], dim=-1)
                    ], dim=-2)
                    save_path = os.path.join(logger.save_path, "images", f"{itr}-{step}.jpg")
                    logger.log_images(img, save_path)
                denoise_recon_loss = nn.functional.mse_loss(pred_images, images).detach().cpu().item()
                lpips_loss = lpips_criterion(pred_images, images).mean().cpu().item()
                lpips_losses.append(lpips_loss)
                losses.append(denoise_recon_loss)
        logger.log_metrics({
            "val_mse_loss": np.mean(losses),
            "val_lpips_loss": np.mean(lpips_loss)
        })

    diffusion_model.train()
