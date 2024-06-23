import torch

from src.models.cstm import AutoEncoder as CTSM
from src.models.patch_cstm import AutoEncoder as PatchCTSM


def standard_vae_v1(device: str, *args, **kwargs):
    return CTSM(is_vae=True).to(device)


def standard_ctms_v1(device: str, *args, **kwargs):
    return CTSM().to(device)


def patch_snvae_v1(device: str, *args, **kwargs):
    return PatchCTSM().to(device)


def patch_snvae_v2(device: str, *args, **kwargs):
    return PatchCTSM(decoder_denorm=True).to(device)


def patch_snvae_v3(device: str, *args, **kwargs):
    return PatchCTSM(decoder_denorm=False, enc_norm="ln").to(device)


def patch_snvae_v4(device: str, *args, **kwargs):
    return PatchCTSM(decoder_denorm=False, enc_norm="gbn").to(device)


def patch_snvae_v5(device: str, *args, **kwargs):
    return PatchCTSM(decoder_denorm=False, enc_norm="tbn").to(device)

def patch_vae_v1(device: str, *args, **kwargs):
    return PatchCTSM(is_vae=True).to(device)

# WIDTH SCALING

def patch_vae_v1_width_tiny(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=2*64, c_latent=16, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_width_small(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=4*64, c_latent=16, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_width_base(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=8*64, c_latent=16, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_width_c4_base(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=8*64, c_latent=4, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_width_c8_base(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=8*64, c_latent=8, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_width_large(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=2*8*64, c_latent=16, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_width_xlarge(device: str, *args, **kwargs):
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=4*8*64, c_latent=16, encoder_block=9, decoder_block=18, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


# COMPOUND SCALING

def patch_vae_v1_compound_tiny(device: str, *args, **kwargs):
    scale_factor = 3
    depth_enc = (scale_factor + 1)
    depth_dec = depth_enc * 2
    width = scale_factor * 64
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=width, c_latent=16, encoder_block=depth_enc, decoder_block=depth_dec, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_compound_small(device: str, *args, **kwargs):
    scale_factor = 5
    depth_enc = (scale_factor + 1)
    depth_dec = depth_enc * 2
    width = scale_factor * 64
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=width, c_latent=16, encoder_block=depth_enc, decoder_block=depth_dec, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_compound_base(device: str, *args, **kwargs):
    scale_factor = 8
    depth_enc = (scale_factor + 1)
    depth_dec = depth_enc * 2
    width = scale_factor * 64
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=width, c_latent=16, encoder_block=depth_enc, decoder_block=depth_dec, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_compound_large(device: str, *args, **kwargs):
    scale_factor = 13
    depth_enc = (scale_factor + 1)
    depth_dec = depth_enc * 2
    width = scale_factor * 64
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=width, c_latent=16, encoder_block=depth_enc, decoder_block=depth_dec, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


def patch_vae_v1_compound_xlarge(device: str, *args, **kwargs):
    scale_factor = 21
    depth_enc = (scale_factor + 1)
    depth_dec = depth_enc * 2
    width = scale_factor * 64
    return PatchCTSM(
        is_vae=True,
        c_in=3, c=width, c_latent=16, encoder_block=depth_enc, decoder_block=depth_dec, compression_factor=8, attention_heads=None, dropout=0.1).to(device)


# ImageNet-scaling:     depth_scale_factor = 1.2, wdith_scale_factor = 1.1


if __name__ == "__main__":
    #model = standard_vae_v1("cpu")
    #print("Model trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000, "M")

    model_factories = [
        patch_vae_v1_width_tiny,
        patch_vae_v1_compound_tiny,
        patch_vae_v1_width_small,
        patch_vae_v1_compound_small,
        patch_vae_v1_width_base,
        patch_vae_v1_compound_base,
        patch_vae_v1_width_large,
        patch_vae_v1_compound_large,
        patch_vae_v1_width_xlarge,
        patch_vae_v1_compound_xlarge,
    ]

    for model_factory in model_factories:
        model = model_factory("cpu")
        print(model_factory.__name__, "Model trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000, "M")
        dummy_input = torch.zeros((1, 3, 128, 128))
        x, mu, std, logvar = model(dummy_input)
        print(x.size(), mu.size(), std.size(), logvar.size())
        print()