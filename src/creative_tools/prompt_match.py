import glob
import hydra
import io
import json
import logging
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
import pprint
import random
import requests
import shutil
import tqdm
from typing import Any, Dict, List
import uuid

from creative_tools import configure_path
from creative_tools.database import PromptDB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@hydra.main(config_path=configure_path(__file__), config_name='default')
def main(cfg: DictConfig):

    # instantiate prompt database manager.
    pdbm = PromptDB(
        db_name=cfg.db_name,
        db_path=cfg.db_path,
        text_embedding_model_address=cfg.text_embedding_model_addr
    )

    # if a prompt file is provided, load and add prompts to the database.
    if cfg.prompt_file:
        try:
            pdbm.load_prompts(cfg.prompt_file)
        except KeyboardInterrupt:
            print('gracefully exiting...')

    # create an image-text dataset from a directory of images, an existing set of
    # prompts and a joint image-text embedding model...
    if cfg.mode == 'dataset':
        dataset_dict = pdbm.process_dataset(
            top_k=cfg.top_k,
            target_dir=cfg.images_dir,
            images_dir=cfg.output_dir,
            image_source=cfg.image_source,
            traintest_split=cfg.traintest_split,
            testval_split=cfg.testval_split,
        )

        output_dir = Path(cfg.output_dir)
        log.info(f'saving to {output_dir}')

        # save the entire unsplit dataset.
        pdbm.save_split(dataset_dict['samples'], output_dir / 'data.jsonl')

        # save each dataset split that exists.
        for split,data in dataset_dict['splits']:
            if data:
                pdbm.save_split(data, output_dir / f'{split}.jsonl')

        log.info(f"processed {len(samples)} images, skipped {len(skipped)} images.")
        log.info(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    elif cfg.mode == 'iterate':
        for image_path in glob.glob(target_dir,'*'):
            matches = pdbm.find_similar_prompts(image_path, top_k=cfg.top_k)
            pprint.pprint(matches, indent=2)
            input()
    else:
        raise ValueError(f"unknown {cfg.mode=}")

if __name__ == "__main__":
    main()
