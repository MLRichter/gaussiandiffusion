import torch
import numpy as np

from torch import nn

from src.models.building_block import SimpleBlock, LayerNorm2d, GlobalBatchNorm2d, TotalBatchNorm2d


class AutoEncoder(nn.Module):
    def __init__(self, c_in=3, c=8*64, c_latent=16, encoder_block=9, decoder_block=18, compression_factor=8,
                 attention_heads=None, dropout=0.1, is_vae: bool = False, decoder_denorm: bool = False, enc_norm: str = 'bn'):
        super().__init__()    
        
        levels = int(np.log2(compression_factor))
        
        # ENCODER
        self.in_smoother = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_in, c_in, kernel_size=7)
        )
        encoder_blocks = [
            nn.PixelUnshuffle(compression_factor),
            nn.Conv2d(c_in*compression_factor**2, 4*c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4*c, c, kernel_size=1),
        ]
        encoder_blocks += [SimpleBlock(c, attention_heads, dropout=dropout) for _ in range(encoder_block)]
        encoder_blocks += [
            nn.Conv2d(c, c*4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c*4, c_latent*2, kernel_size=1)
        ]
        self.encoder = nn.Sequential(*encoder_blocks)
        if enc_norm.lower() == "bn":
            self.norm = nn.SyncBatchNorm(c_latent, affine=False) if not is_vae else nn.Identity()
        elif enc_norm.lower() == "gbn":
            if decoder_denorm:
                raise ValueError("Denormalization does not work with Global Batch Norm")
            self.norm = GlobalBatchNorm2d(1, affine=False) if not is_vae else nn.Identity()
        elif enc_norm.lower() == "tbn":
            if decoder_denorm:
                raise ValueError("Denormalization does not work with Global Batch Norm")
            self.norm = TotalBatchNorm2d(1, affine=False) if not is_vae else nn.Identity()
        elif enc_norm.lower() == "ln":
            self.norm = LayerNorm2d(c_latent) if not is_vae else nn.Identity()

        
        # DECODER
        decoder_blocks = [
            nn.Conv2d(c_latent, c*4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c*4, c, kernel_size=1)
        ]
        decoder_blocks += [SimpleBlock(c, attention_heads, dropout=dropout, reflection_pad=True) for _ in range(decoder_block)]
        decoder_blocks += [
            nn.Conv2d(c, 4*c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4*c, c_in*compression_factor**2, kernel_size=1),
            nn.PixelShuffle(compression_factor),
        ]   
        self.decoder = nn.Sequential(*decoder_blocks)
        self.out_smoother = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_in, c_in, kernel_size=7)
        )
        self.is_vae = is_vae
        self.decoder_denorm = decoder_denorm

    def denorm(self, x):
        mu, var = self.norm.running_mean, self.norm.running_var
        x = torch.nn.functional.batch_norm(x, torch.zeros_like(mu), 1/var)
        x = torch.nn.functional.batch_norm(x, -mu, torch.ones_like(var))
        return x
        
    def encode(self, x):
        x = self.in_smoother(x)
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        std = (logvar * 0.5).exp()
        x = torch.randn_like(std) * std + mu
        return self.norm(x), mu, std, logvar

    def decode(self, x):
        if self.decoder_denorm:
            x = self.denorm(x)
        x = self.decoder(x)
        return x #return self.out_smoother(x)

    def forward(self, x):
        x, mu, std, logvar = self.encode(x)
        x = self.decoder(x)
        return x, mu, std, logvar


class Discriminator(nn.Module):
    def __init__(self, c_in=3, c=8 * 64, blocks=8, compression_factor=8, attention_heads=None, dropout=0.1):
        super().__init__()
        # ENCODER
        self.in_smoother = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_in, c_in, kernel_size=7)
        )
        encoder_blocks = [
            nn.PixelUnshuffle(compression_factor),
            nn.Conv2d(c_in * compression_factor ** 2, 4 * c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * c, c, kernel_size=1),
        ]
        encoder_blocks += [SimpleBlock(c, attention_heads, dropout=dropout) for _ in range(blocks)]
        encoder_blocks += [
            nn.Conv2d(c, c * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c * 4, 1, kernel_size=1)
        ]
        self.encoder = nn.Sequential(*encoder_blocks)

    def forward(self, x):
        x = self.in_smoother(x)
        return self.encoder(x)


if __name__ == '__main__':
    from time import time
    from discrimintors import Discriminator
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    model = AutoEncoder().to(device)
    # model = AutoEncoder(c=8*64, encoder_block=4, decoder_block=12, attention_heads=8).to(device)
    print("Model trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    disc = Discriminator().to(device)
    print("Disc trainable params:", sum(p.numel() for p in disc.parameters() if p.requires_grad))
    x = torch.randn(64, 3, 128, 128, device=device)
    model.eval(), disc.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        start_time = time()
        enc = model.encode(x)[0]
        print(enc.shape, enc.mean().item(), enc.std().item())
        pred = model.decode(enc)
        d = disc(pred)
        print("ELAPSED:", round(time() - start_time, 3), "s")
        print(pred.shape, d.shape)
    model.train(), disc.train()
