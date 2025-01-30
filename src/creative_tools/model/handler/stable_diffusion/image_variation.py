from diffusers import StableDiffusionImageVariationPipeline
import torch

from base import Handler as BaseHandler

class Handler(BaseHandler):
    def __init__(self, **kwargs):
        self._context = None
        self.device = None
        self.initialized = False
        self.model = None

    def load_model(self, model_dir, device):
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
              "lambdalabs/sd-image-variations-diffusers",
              revision="v2.0",
        ).to(device)
