import cv2
from datasets import load_dataset
import glob
import numpy as np
import os
from pathlib import Path
from PIL import Image
import uuid

# default function for uniquely identifying IDs, filenames, or other artifacts.
unique_id = lambda width=4: str(uuid.uuid4())[:width]
# default function for converting a text "prompt" to a concise, spaceless string.
format_prompt = lambda prompt:prompt.replace(' ', '_')[:64]

def configure_path(path):
    tool_path = Path(path).resolve()
    proj_root = tool_path.parents[2]
    conf_path = proj_root / f'config/{tool_path.stem}'
    return str(conf_path)

def load_images(input_path, search_pattern='**/*.[jp][pn][eg]*', default_split='train'):
    if isinstance(input_path, list):
        return input_path

    if isinstance(input_path, str):
        if os.path.isfile(input_path):
            return [input_path]
        elif os.path.isdir(input_path):
            return glob.glob(os.path.join(input_path, search_pattern), recursive=True)

    dataset = load_dataset(input_path, split=default_split)
    return [sample['file_path'] for sample in dataset]

def array_to_image(array: np.ndarray) -> Image:
    """
    Converts a NumPy array to a PIL Image.

    :param array: input image array.
    :type array: np.ndarray
    :return: RGB-formatted PIL Image.
    :rtype: PIL.Image
    """
    rgb_img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return pil_img

def image_to_array(image:Image) -> np.ndarray:
    """
    Converts a PIL Image to a NumPy array.

    :param array: input image.
    :type image: PIL.Image
    :return: BGR-formatted NumPy array.
    :rtype: np.ndarray
    """
    rgb_image = np.asarray(image)
    cv2_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return cv2_img

def image_to_bytes(image) -> bytes:
    """
    Converts a PIL Image to PNG bytes.

    :param image: input image.
    :type image: PIL.Image
    :return: PNG-formatted image as bytes.
    :rtype: bytes
    """
    bytes_ = io.BytesIO()
    image.save(bytes_, format="PNG")
    bytes_.seek(0)
    data = bytes_.read()
    return data