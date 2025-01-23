import base64
from diffusers import DiffusionPipeline
import io
import json
import numpy as np
import os
from PIL import Image
import sys
import torch

from ts.torch_handler.base_handler import BaseHandler

class Handler(BaseHandler):
    def __init__(self, **kwargs):
        self._context = None
        self.device = None
        self.initialized = False
        self.model = None

    def load_model(self, model_dir, device):
        model_path = os.path.join(props.get('model_dir'), 'model')
        self.pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)

    def initialize(self, context):
        self.manifest = context.manifest
        props = context.system_properties
        if torch.cuda.is_available() and props.get('gpu_id') is not None:
            self.device = torch.device(f'cuda:{props.get("gpu_id")}')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.load_model(props.get('model_dir'), self.device)
        self.initialized = True

    def handle(self, data, context):
        processed_data = self.preprocess(data)
        if processed_data is None:
            return [dict(body=json.dumps(dict(error='No input data provided.')))]
        model_output = self.inference(processed_data)
        return self.postprocess(model_output)

    def preprocess(self, data):
        if len(data) == 0:
            return None
        json_data = data[0].get('body')
        if isinstance(json_data, (str, bytes)):
            json_data = json.loads(json_data)
        return json_data

    def inference(self, data):
        prompt = data.get('prompt', '')
        seed = int(data.get('seed', np.random.randint(0, 2**64, dtype=np.uint64)))
        num_images_per_prompt = data.pop('batch', 1)

        base64_image = data.pop('image', None)
        try:
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as decode_image_prompt_exception:
            raise ValueError(f"{decode_image_prompt_exception=}")
            image = None
        kwargs = {k:v for k,v in data.items() if not k in ['prompt','seed','batch']}
        if image is not None:
            kwargs['image'] = image

        with torch.no_grad():
            images = self.pipe(
                prompt,
                num_images_per_prompt=num_images_per_prompt,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                **kwargs
            ).images
        return images, seed

    def postprocess(self, inference_output):
        images, seed = inference_output
        output = []
        for image in images:
            bytes = io.BytesIO()
            image.save(bytes, format='PNG')
            output.append(base64.b64encode(bytes.getvalue()).decode())
        return [dict(body=json.dumps(dict(images=output, seed=seed)))]
