from datasets import load_dataset
import glob
import os
from pathlib import Path
import uuid

id_image = lambda width=4: str(uuid.uuid4())[:width]
format_prompt = lambda prompt:prompt.replace(' ', '_')[:64]

def configure_path(path):
    tool_path = Path(path).resolve()
    proj_root = tool_path.parents[2]
    conf_path = proj_root / f'config/{tool_path.stem}'
    return str(conf_path)

def load_images(input_path, search_pattern= f'**/*.png', default_split='train'):
    if isinstance(input_path, list):
        return input_path
    elif isinstance(input_path, str) and os.path.isfile(input_path):
        return [input_path]
    elif isinstance(input_path, str) and os.path.isdir(input_path):
        found = glob.glob(os.path.join(input_path, search_pattern), recursive=True)
        if not found:
            found = glob.glob(os.path.join(input_path, f'*.png'), recursive=False)
        return found
    else:
        dataset = load_dataset(input_path, split=default_split)
        return [sample['file_path'] for sample in dataset]
