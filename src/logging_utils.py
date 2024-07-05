import json
import pickle

import wandb
from pathlib import Path
from typing import Dict, Protocol, List

import torch
import torchvision
from torchvision.utils import make_grid


def _postprocess_model_name(model_name: str) -> str:
    return " ".join([component.capitalize() if component != 'vae' else component.upper() for component in model_name.split("_")])


class Logger(Protocol):

    def log_metrics(self, metrics: Dict):
        ...

    def log_images(self, img: torch.Tensor, save_path: str):
        ...


class PrintLogger:

    def __init__(self, **kwargs):
        pass

    def log_metrics(self, metrics: Dict):
        print(metrics)

    def log_images(self, img: torch.Tensor, save_path: str):
        torchvision.utils.save_image(img, save_path)


class DummyLogger:

    def __init__(self, **kwargs):
        pass

    def log_metrics(self, metrics: Dict):
        pass

    def log_images(self, img: torch.Tensor, save_path: str):
        pass


class JsonLogger:

    def __init__(self, save_path: str, **kwargs):
        self.save_path = save_path

    def log_metrics(self, metrics: Dict):
        with open(Path(self.save_path) / "logs.json", "a+") as fp:
            json.dump(metrics, fp)
            fp.write("\n")

    def log_images(self, img: torch.Tensor, save_path: str):
        torchvision.utils.save_image(img, save_path)


class JsonWandbLogger:
    def __init__(self, save_path: str, project_name: str = "VAE-Diffusion-Train", **kwargs):
        name = _postprocess_model_name(f"{kwargs['latent_encoder_name']}-{kwargs['model_name']}-{kwargs['text_encoder_name']}")

        self.save_path = save_path
        if (Path(self.save_path) / "id.wandb").exists():
            with open(Path(self.save_path) / "id.wandb", "rb") as fp:
                wandb_id = pickle.load(fp)
        else:
            wandb_id = wandb.util.generate_id()
            with open(Path(self.save_path) / "id.wandb", "wb") as fp:
                pickle.dump(wandb_id, fp)
        wandb.init(project=project_name, name=name, config=kwargs, id=wandb_id, resume="allow")

    def log_metrics(self, metrics: Dict):
        wandb.log(metrics)
        with open(Path(self.save_path) / "logs.json", "a+") as fp:
            json.dump(metrics, fp)
            fp.write("\n")

    def log_images(self, img: torch.Tensor, save_path: str):
        #from skimage.io import imsave
        #from cv2 import imwrite
        torchvision.utils.save_image(img, save_path)
        #grid = make_grid(img)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        #ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        #imsave(str(save_path), ndarr)
        #imwrite(str(save_path), ndarr)

        wandb.log({'Example Batch {}'.format(Path(save_path).with_suffix("").name): wandb.Image(img)})