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
from .utils import preprocess_graph_Hetero, preprocess_v1


class ProcessedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processed_data_path: str, sample_size: int = -1):
        super(ProcessedDataset, self).__init__()
        logging.warning("Loading data...")

        self.processed_data = torch.load(processed_data_path)
        
        invalid_ids = []
        for i, data in enumerate(self.processed_data):
            if data["input_ids"].shape[0] >= 8192:
                invalid_ids.append(i)
        for i in invalid_ids[::-1]:
            del self.processed_data[i]
        logging.warning(f"Removed {len(invalid_ids)} samples with input_ids >= 8192")

        invalid_ids = []
        for i, data in enumerate(self.processed_data):
            # if all labels are -100
            if torch.all(data["labels"] == -100):
                invalid_ids.append(i)
        for i in invalid_ids[::-1]:
            del self.processed_data[i]
        logging.warning(f"Removed {len(invalid_ids)} samples with labels all -100")

        if sample_size > 0:
            self.processed_data = self.processed_data[:sample_size]
            logging.warning(f"Sampled {sample_size} samples for {processed_data_path}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.processed_data[i]
