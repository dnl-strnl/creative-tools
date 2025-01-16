from diffusers import StableDiffusionUpscalePipeline
import os
import torch

from base import Handler as BaseHandler

class Handler(BaseHandler):
    def __init__(self, **kwargs):
        super().__init__()

    def load_model(self, model_dir, device):
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            pretrained_model_name_or_path=os.path.join(model_dir, 'model'),
            variant="fp16",
            torch_dtype=torch.float16
            use_safetensors=True,
        ).to(device)
