from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

from base import Handler as BaseHandler

class Handler(BaseHandler):
    def __init__(self, **kwargs):
        self._context = None
        self.device = None
        self.initialized = False
        self.model = None

    def load_model(self, model_dir, device):
        model_id = "stabilityai/stable-diffusion-3.5-large"

        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        )

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        ).to(device)
