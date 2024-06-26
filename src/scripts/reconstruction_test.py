import torchvision

from src.models.vae_registry import patch_vae_v1_compound_small, patch_vae_v1_width_small
from skimage.data import astronaut
import numpy as np
import torch

img = astronaut() / astronaut().max()
print(img.shape)

model: torch.nn.Module = patch_vae_v1_width_small(device="cuda")
state_dict = torch.load("model.pt")['state_dict']
model.load_state_dict(state_dict=state_dict)
img = torch.from_numpy(img).swapaxes(0, 2).swapaxes(2, 1).unsqueeze(0).float().cuda()
pred, _, _, _ = model(img)
enc, _, _, _ = model.encode(img)
dec = model.decode(enc)
collage = torch.cat([img, pred, dec], dim=-1)
print(collage.size())
save_path = "collage.png"
torchvision.utils.save_image(collage, save_path)
