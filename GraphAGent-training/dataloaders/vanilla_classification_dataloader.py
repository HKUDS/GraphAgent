import copy
import json
import logging
from typing import Dict
import torch
import transformers
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os.path as osp
import glob
from .utils_llama3 import preprocess_llama3
from pathlib import Path


class VanillaClassificationDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):
        super(VanillaClassificationDataset, self).__init__()
        logging.warning("Loading data...")

        self.tokenizer = tokenizer

        if Path(data_path).parts[-1].endswith('.json'):
            ann_data_paths = [Path(data_path)]

        else:
            ann_data_paths = glob.glob(osp.join(data_path, '**/*.json'), recursive=True)
            assert len(ann_data_paths) > 0, f"Need to have one ann file for each graph"

        ann_data = []
        for ann_file in ann_data_paths:
            ann_data_i = json.load(open(ann_file, "r", encoding= "utf-8"))
            ann_data.extend(ann_data_i)

        self.ann_data = []
        for ann_entry in ann_data:
            self.ann_data.append(ann_entry)

        logging.warning("Formatting inputs...Skip in lazy mode")

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        prompt_entry = self.ann_data[i]

        token_dict = preprocess_llama3(
            copy.deepcopy([prompt_entry["conversations"]]),
            self.tokenizer)
        
        if isinstance(i, int):
            token_dict = dict(input_ids=token_dict["input_ids"][0],
                             labels=token_dict["labels"][0])

        token_dict['id'] = prompt_entry['id']

        return token_dict