from collections import Counter
from creative_tools import *
from datasets import load_dataset
import glob
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
from PIL import Image
import re
import shutil
import torchvision.transforms as T
from torchvision.utils import make_grid
import tqdm
import uuid


def get_image_hue(image_path, format='HSV'):
    '''
    calculates the average hue of an image.
    '''
    arr = np.array(Image.open(image_path).convert())
    hue = arr[..., format.index('H')].mean()
    return hue

def resize_images_uniform(images):
    '''
    uniformly resizes a list of images to the most common resolution in the set.
    '''
    image_tensors = [T.ToTensor()(Image.open(image).convert('RGB')) for image in images]
    sizes = [(image.shape[1], image.shape[2]) for image in image_tensors]
    size = Counter(sizes).most_common(1)[0][0]
    resized = [T.Resize(size)(image) for image in image_tensors]
    return resized

def dataset_grid_preview(images, grid_size=5):
    '''
    generate a square image grid with a random subset of dataset images.
    '''
    # sort preview images by hue
    images_with_hues = [(image, get_image_hue(image)) for image in images]
    images_sorted_by_hue = sorted(images_with_hues, key=lambda x: x[1])

    # select evenly spaced images across the hue range
    total_images = len(images_sorted_by_hue)
    indices = np.linspace(0, total_images - 1, grid_size ** 2, dtype=int)
    selected_images = [images_sorted_by_hue[idx][0] for idx in indices]

    # resize images uniformly to the most representative resolution
    resized = resize_images_uniform(selected_images)
    grid = make_grid(resized, nrow=grid_size, padding=2, pad_value=0)

    # downscale the grid image to the resolution of a single sample
    grid = T.ToPILImage()(grid).resize(Image.open(selected_images[0]).size)
    return grid

def save_split(split, split_file):
    '''
    saves a list of data samples in JSON format, if present.
    '''
    if len(split):
        with open(split_file, 'w') as f:
            for sample in split:
                f.write(f'{json.dumps(sample)}\n')

def split_dataset(dataset, traintest_split=0.90, testval_split=0.75):
    '''
    splits a list of samples comprising a dataset into training, testing, and validation sets.
    '''
    np.random.shuffle(dataset)

    # split the dataset into training and evaluation sets
    split_index = int(len(dataset) * traintest_split)
    train_set, temp_test_set = dataset[:split_index], dataset[split_index:]

    # further split the evaluation set into testing and validation sets
    test_split_index = int(len(temp_test_set) * testval_split)
    test_set, val_set = temp_test_set[:test_split_index], temp_test_set[test_split_index:]

    return train_set, val_set, test_set

@hydra.main(config_path=configure_path(__file__), config_name='default')
def main(cfg: DictConfig):

    image_files = load_images(cfg.input)

    outdir = os.path.join(cfg.output, cfg.name)
    os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)

    samples, prompts, aliases, skipped, original = [], [], [], [], []

    bar = tqdm.tqdm(image_files, total=len(image_files))
    for fname in bar:
        try:
            fname = fname.replace('\n','')
            image_source = os.path.basename(os.path.dirname(fname))
            # default to extracting prompt from filename.
            prompt = Path(fname).stem

            if cfg.regex:
                match = re.search(cfg.regex, fname)
                if match:
                    prompt = match.group(1)
                else:
                    if cfg.require_text:
                        raise Exception(f'{cfg.regex} >> {fname}')
                    else:
                        prompt = ''

            prompts.append(prompt)
            bar.set_description(f'{prompt=}'[:50])

            while (alias := unique_id()) in aliases: pass
            aliases.append(alias)

            local_file = os.path.join('images', f'{alias}.png')
            rename = os.path.join(outdir, local_file)
            original.append(fname)

            sample_dict = dict(
                file_name=local_file,
                alias=alias,
                text=[prompt],
                image_source=image_source,
                prompt_source='human'
            )

            samples.append(sample_dict)
            shutil.copy(fname, rename)
        except Exception as ex:
            print(ex, fname)
            skipped.append(fname)

    print(f'{len(samples)=}')

    if prompt_file := True:
        with open(os.path.join(outdir, 'prompts.txt'), 'w') as f:
            for prompt in prompts:
                if prompt.strip(): f.write(f'{prompt}\n')

    print('creating image grid preview...')
    grid_file = os.path.join(outdir, f'{cfg.name}-preview.png')
    dataset_grid = dataset_grid_preview(original, cfg.preview_grid)
    dataset_grid.save(grid_file)
    print(f'done. created {grid_file}')

    if cfg.traintest_split is None and cfg.testval_split is None:
        save_split(samples, data_files := dict(train=['metadata.jsonl']))
    else:
        train_set, val_set, test_set = split_dataset(
            samples, cfg.traintest_split, cfg.testval_split
        )

    save_split(train_set, train := str(Path(outdir) / 'train.jsonl'))
    save_split(val_set, val := str(Path(outdir) / 'val.jsonl'))
    save_split(test_set, test := str(Path(outdir) / 'test.jsonl'))
    save_split(train_set + val_set + test_set, str(Path(outdir) / 'data.jsonl'))

    dataset = load_dataset('json', data_files=dict(train=train, val=val, test=test))

    samples = dataset['train']
    print(f'{samples[0]=}\n{skipped=}')

    return dataset

if __name__ == "__main__":
    main()
