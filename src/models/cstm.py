import torch
import numpy as np

from torch import nn

from src.models.building_block import SimpleBlock


class AutoEncoder(nn.Module):
    def __init__(self, c_in=3, c=8 * 64, c_latent=16, encoder_block=4, decoder_block=12, compression_factor=8,
                 attention_heads=8, dropout=0.1, is_vae: bool = False):
        super().__init__()

        levels = int(np.log2(compression_factor))

        # ENCODER
        encoder_blocks = [
            nn.PixelUnshuffle(2),
            nn.Conv2d(c_in * 4, 4 * c // (2 ** (levels - 1)), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * c // (2 ** (levels - 1)), c // (2 ** (levels - 1)), kernel_size=1),
        ]
        for i in range(levels - 1):
            encoder_blocks += [
                *[SimpleBlock(c // (2 ** (levels - 1 - i)), None, dropout=dropout) for _ in range(2)],
                nn.Conv2d(c // (2 ** (levels - 1 - i)), c // (2 ** (levels - 2 - i)), kernel_size=4, stride=2,
                          padding=1)
            ]
        encoder_blocks += [SimpleBlock(c, attention_heads, dropout=dropout) for _ in range(encoder_block)]
        encoder_blocks += [
            nn.Conv2d(c, c * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c * 4, c_latent * 2, kernel_size=1)
        ]
        self.encoder = nn.Sequential(*encoder_blocks)
        # normalize only if it is a vae
        self.norm = nn.BatchNorm2d(c_latent, affine=False) if not is_vae else nn.Identity()
        self.is_vae = is_vae

        # DECODER
        decoder_blocks = [
            nn.Conv2d(c_latent, c * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c * 4, c, kernel_size=1)
        ]
        decoder_blocks += [SimpleBlock(c, attention_heads, dropout=dropout) for _ in range(decoder_block)]
        for i in range(levels - 1):
            decoder_blocks += [
                nn.ConvTranspose2d(c // (2 ** i), c // (2 ** (i + 1)), kernel_size=4, stride=2, padding=1),
                *[SimpleBlock(c // (2 ** (i + 1)), None, dropout=dropout) for _ in range(2)],
            ]
        decoder_blocks += [
            nn.Conv2d(c // (2 ** (levels - 1)), 4 * c // (2 ** (levels - 1)), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * c // (2 ** (levels - 1)), c_in * 4, kernel_size=1),
            nn.PixelShuffle(2),
        ]
        self.decoder = nn.Sequential(*decoder_blocks)

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        std = (logvar * 0.5).exp()
        x = torch.randn_like(std) * std + mu
        return self.norm(x), mu, std, logvar

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x, mu, std, logvar = self.encode(x)
        x = self.decoder(x)
        return x, mu, std, logvar


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
