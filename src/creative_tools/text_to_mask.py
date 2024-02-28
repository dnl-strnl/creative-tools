from creative_tools import *
import cv2
import glob
import hydra
import json
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from omegaconf import DictConfig
import os
from PIL import Image
from pycocotools import mask as coco_mask
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
import tqdm
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import uuid


def mask_overlay(image, mask, alpha_blend=0.5, color_map='jet'):
    '''
    resize instance segmentation and alpha-blended overlay masks.
    '''
    black_and_white_mask = (255 * mask).astype(np.uint8)

    mask_image = Image.fromarray(black_and_white_mask).convert('L')

    resized_mask = mask_image.resize(image.size)

    mask_cmap = plt.get_cmap(color_map)(Normalize()(resized_mask))
    mask_cmap = (mask_cmap[:, :, :3] * 255).astype(np.uint8)

    rgb_mask = Image.fromarray(mask_cmap)

    overlay_image = Image.blend(
        image.convert('RGBA'), rgb_mask.convert('RGBA'), alpha=alpha_blend
    )

    return resized_mask, overlay_image


def annotate_instance(binary_mask, prompt=''):
    '''
    extract COCO-like annotation metadata from a binary instance mask.
    '''
    mask = np.asfortranarray(binary_mask)
    rlen = coco_mask.encode(mask)
    mask = coco_mask.decode(rlen)
    area = int(coco_mask.area(rlen))
    bbox = list(cv2.boundingRect(mask.astype(np.uint8)))

    if 'counts' in rlen and isinstance(rlen['counts'], bytes):
        rlen['counts'] = rlen['counts'].decode('utf-8')

    instance = dict(
        area=area,
        bbox=bbox,
        segmentation=rlen,
        prompt=prompt,
    )
    return instance


@hydra.main(config_path=configure_path(__file__), config_name='default')
def main(cfg: DictConfig):
    image_files = load_images(cfg.image)

    outdir = os.path.join(cfg.output, cfg.model.replace('/','_'))
    os.makedirs(outdir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model)
    model = CLIPSegForImageSegmentation.from_pretrained(cfg.model)

    prompts = str(cfg.text).split(str(cfg.delimiter))

    annotations, aliases, skipped = [], [], []
    overlays = {k:[] for k in prompts}

    bar = tqdm.tqdm(image_files, total=len(image_files))
    for fname in bar:
        try:
            while (alias := id_image()) in aliases: pass
            aliases.append(alias)

            image = Image.open(fname).convert('RGB')
            images = [image] * len(prompts)
            inputs = processor(text=prompts, images=images, padding=True, return_tensors='pt')

            with torch.no_grad(): outputs = model(**inputs)
            output = F.softmax(outputs.logits, dim=0)

            for i,(prompt, probabilities) in enumerate(zip(prompts, output)):
                mask = (probabilities > cfg.confidence_threshold).numpy()

                if not np.any(mask) or mask.sum() < cfg.min_instance_area:
                    continue

                metadata = annotate_instance(mask, prompt=prompt)
                metadata['image'] = alias

                instance_id = f'{alias}-{format_prompt(prompt)}-{str(i).zfill(4)}'
                metadata['instance'] = instance_id

                logging.info(f'{fname}: {metadata}')
                annotations.append(metadata)

                resized_mask, overlay_image = mask_overlay(
                    image, mask, cfg.mask_alpha, cfg.mask_colormap
                )

                if cfg.mask:
                    resized_mask.save(os.path.join(outdir, f'{instance_id}_mask.png'))

                if cfg.overlay:
                    overlay_path = os.path.join(outdir, f'{instance_id}_overlay.png')
                    overlay_image.save(overlay_path)
                    overlays[prompt].append(overlay_path)

        except Exception as ex:
            print(ex, fname)
            skipped.append(fname)

    if cfg.grid:
        for prompt, paths in overlays.items():
            imgs = [T.ToTensor()(Image.open(f).convert('RGB')) for f in paths]
            if not imgs: continue
            grid = make_grid(imgs, padding=2, pad_value=0)
            T.ToPILImage()(grid).save(
                os.path.join(outdir, f'{format_prompt(prompt)}-overlay.png')
            )

    with open(os.path.join(outdir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    main()
