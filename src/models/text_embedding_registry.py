from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
from typing import Tuple


def vith_text_encoder(device: str) -> Tuple[AutoTokenizer, CLIPTextModel]:
    clip_text_model_name = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
    clip_text_model = CLIPTextModel.from_pretrained(clip_text_model_name).to(device).eval().requires_grad_(False)
    clip_tokenizer = AutoTokenizer.from_pretrained(clip_text_model_name)
    return clip_text_model, clip_tokenizer


if __name__ == '__main__':
    txt_model, tokenizer = vith_text_encoder("cpu")
    captions = ["I am sorry Dave! I cannot do that."]
    clip_tokens = tokenizer(captions, truncation=True, padding="max_length",
                                 max_length=tokenizer.model_max_length, return_tensors="pt")

    embeddings = txt_model(**clip_tokens, output_hidden_states=True).last_hidden_state
    print(embeddings.size())