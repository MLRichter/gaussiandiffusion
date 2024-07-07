import PIL.Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from src.models.factories import get_model, get_text_embedding, get_latent_encoder

unet = get_model("unet_40M", "cuda:0").module
text_encoder, tokenizer = get_text_embedding("vith_text_encoder", "cuda:0")
vae = get_latent_encoder('patch_vae_v1_width_tiny', None, "cuda:0")
noise_scheduler = DDPMScheduler(prediction_type='v_prediction')

vae.return_tuple = True

tokenizer.added_tokens_encoder = tokenizer.all_special_tokens_extended
pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
#pipeline._execution_device = "cuda:0"

img: PIL.Image.Image = pipeline(["this is a test", 'this is test']).images[0]
img.save(open("file.png", "wb"))
print(img)