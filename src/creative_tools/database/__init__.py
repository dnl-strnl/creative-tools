import chromadb
import glob
import json
import numpy as np
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple
import tqdm
import uuid
import logging

from creative_tools import unique_id
from creative_tools.embeddings import Embedder

log = logging.getLogger(__name__)

class PromptDB:
    def __init__(
        self,
        db_path: str,
        db_name: str,
        text_embedding_model_address: str,
        text_embedding_dimension: int = 512,
        batch_size: int = 64,
        output_dir: str = os.getcwd(),
    ):
        self.db_path = db_path
        self.db_name = db_name
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.embedder = Embedder(
            embedding_model_address=text_embedding_model_address,
            embedding_dimension=text_embedding_dimension,
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.db_name,
            embedding_function=self.embedder
        )
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_existing_prompts(self) -> set:
        try:
            results = self.collection.get()
            existing_prompts = set(results['documents']) if results['documents'] else set()
        except Exception as prompt_retrieval_error:
            log.error(f"{prompt_retrieval_error=}")
            existing_prompts = set()
        finally:
            return list(existing_prompts)

    def load_prompts(self, prompt_file: str) -> None:
        try:
            # read and deduplicate new prompts.
            existing_prompts = self.get_existing_prompts()
            with open(prompt_file, 'r') as f:
                new_prompts = list(set([line.strip() for line in f if line.strip()]))

            prompts_to_add = [p for p in new_prompts if p not in existing_prompts]

            log.info(f"found {len(new_prompts)} prompts in file.")
            log.info(f"skipping {len(new_prompts) - len(prompts_to_add)} existing.")
            log.info(f"adding {len(prompts_to_add)} new prompts to database...")

            for i in tqdm.tqdm(range(0, len(prompts_to_add), self.batch_size)):
                batch = prompts_to_add[i:i + self.batch_size]
                try:
                    self.collection.add(
                        documents=batch,
                        metadatas=[
                            dict(type='prompt', text=prompt) for prompt in batch
                        ],
                        ids=[str(uuid.uuid4()) for _ in batch]
                    )
                    log.info(f"processed batch of {len(batch)} prompts.")
                except Exception as add_prompt_to_db_error:
                    log.error(f"{add_prompt_to_db_error=}")

        except Exception as load_prompts_error:
            log.error(f"{load_prompts_error}")
            raise

    @staticmethod
    def split_dataset(
        dataset: List[Dict],
        traintest_split: float = 0.90,
        testval_split: float = 0.90
    ) -> Tuple:
        dataset_copy = dataset.copy()
        np.random.shuffle(dataset_copy)

        split_index = int(len(dataset_copy) * traintest_split)
        train_set, temp_test_set = dataset_copy[:split_index], dataset_copy[split_index:]

        test_split_index = int(len(temp_test_set) * testval_split)
        test_set, val_set = temp_test_set[:test_split_index], temp_test_set[test_split_index:]

        return train_set, val_set, test_set

    @staticmethod
    def save_split(split: List[Dict], split_file: str) -> None:
        if len(split) > 0:
            with open(split_file, 'w') as f:
                for sample in split:
                    f.write(f'{json.dumps(sample)}\n')

    def find_similar_prompts(self, image_path: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Query the database to find top-k similar prompts to an input image.
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"{image_path=}")

            log.info(f"{image_path}: searching for similar prompts...")

            results = self.collection.query(
                query_texts=[image_path],
                n_results=top_k,
                include=['metadatas', 'distances', 'documents']
            )

            formatted_results = []
            for i,(doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'prompt': doc,
                    'datapath': image_path,
                    'metadata': metadata,
                    'distance': float(distance),
                })
                if i == 0: log.info(f'prompt: {doc}')

            return formatted_results

        except Exception as prompt_match_error:
            log.error(f"{prompt_match_error=}")
            raise

    def process_dataset(
        self,
        target_dir: str,
        images_dir: str,
        image_source: str,
        top_k: int = 5,
        traintest_split: float = 0.90,
        testval_split: float = 0.75
    ) -> None:

        os.makedirs(images_dir, exist_ok=True)

        aliases = []
        samples = []
        skipped = []
        for image_path in tqdm.tqdm(glob.glob(f'{target_dir}/*.png')):
            try:
                # generate unique image ID.
                while (alias := unique_id()) in aliases: pass
                aliases.append(alias)

                new_filename = f'{alias}.png'
                # copy image to new location.
                new_image_path = os.path.join(images_dir, new_filename)

                shutil.copy2(image_path, new_image_path)
                # get similar prompts.
                matches = self.find_similar_prompts(image_path, top_k=top_k)
                # aggregate top-k matched prompts for image.
                text = [x['prompt'] for x in matches]

                # add the sample to the dataset list.
                samples.append(dict(
                    filename = f'images/{new_filename}',
                    alias = alias,
                    text = text,
                    image_source = image_source,
                    prompt_source = self.db_name,
                ))

                log.info(f"processed {image_path} -> {new_filename}")
            except Exception as image_path_error:
                log.error(f"{image_path_error=}")
                skipped.append(image_path)

        train, val, test = self.split_dataset(
            samples, traintest_split, testval_split
        )

        aux_output = dict(samples=samples, skipped=skipped)
        output = dict(splits=dict(train=train, val=val, test=test), **aux_output)
        return output
