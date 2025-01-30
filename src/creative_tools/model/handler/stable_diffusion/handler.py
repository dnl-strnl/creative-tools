from diffusers import DiffusionPipeline
import os
import torch

from base import Handler as BaseHandler

class Handler(BaseHandler):
    def __init__(self, **kwargs):
        self._context = None
        self.device = None
        self.initialized = False
        self.model = None

    def load_model(self, model_dir, device):
        model_path = os.path.join(model_dir, 'model')
        lora_path = os.path.join(model_dir, 'lora')

        self.pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

        self.pipe.safety_checker = lambda images, **kwargs: \
            (images, [False] * len(images))

        lora_weights = os.path.join(
            lora_path, weight_name := "pytorch_lora_weights.safetensors"
        )
        if os.path.exists(lora_weights):
            self.pipe.load_lora_weights(lora_path, weight_name=weight_name)
