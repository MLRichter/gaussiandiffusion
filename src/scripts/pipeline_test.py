from diffusers import StableDiffusionPipeline

vae = ...
tokenizer, text_encoder = ...
unet = ...


pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )