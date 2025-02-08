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
from .utils_llama3 import preprocess_llama3


class NLPRWDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_config: dict,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict):
        super(NLPRWDataset, self).__init__()
        logging.warning("Loading data...")

        self.tokenizer = tokenizer
        self.graph_cfg = graph_cfg

        ann_data_paths = glob.glob(osp.join(data_path, '**/*.json'), recursive=True)
        assert len(ann_data_paths) > 0, f"Need to have one ann file for each graph"

        ann_data = []
        for ann_file in ann_data_paths:
            ann_data_i = json.load(open(ann_file, "r", encoding= "utf-8"))
            ann_data.extend(ann_data_i)

        self.graph_path = data_config["graph_data"]
        self.graph_data_dict = torch.load(self.graph_path)
        valid_prompt_idx = list(self.graph_data_dict.keys())
        ann_data = {entry["id"]: entry for entry in ann_data}
        self.ann_data = []
        for prompt_idx in valid_prompt_idx:
            try:
                self.ann_data.append(ann_data[prompt_idx])
            except KeyError:
                logging.warning(f"Prompt {prompt_idx} not found in the ann file. Skipping.")

        self.hetero_key_order = data_config['hetero_key_order']

        self.node_edge_type_emb_dict = torch.load(data_config['node_edge_type_emb_dict'])

        logging.warning("Formatting inputs...Skip in lazy mode")

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        prompt_entry = self.ann_data[i]
        graph_dlist = self.graph_data_dict[prompt_entry["id"]]

        cur_token_lens = []
        for graph_dict in graph_dlist:
            for key in self.hetero_key_order:
                cur_token_lens.append(graph_dict.x_dict[key].shape[0])
        assert type(cur_token_lens[0]) == int, f"Need to be int, not {type(cur_token_lens[0])}"

        sources = preprocess_graph_Hetero(
            copy.deepcopy([prompt_entry["conversations"]]),
            self.graph_cfg, cur_token_lens)

        token_dict = preprocess_llama3(
            sources,
            self.tokenizer)
            # systems=["A conversation between a human and an intelligent chatbot."] * len(sources))
        
        if isinstance(i, int):
            token_dict = dict(input_ids=token_dict["input_ids"][0],
                             labels=token_dict["labels"][0])

        token_dict['id'] = prompt_entry['id']
        token_dict['graph_data'] = graph_dlist
        token_dict['hetero_key_order'] = [self.hetero_key_order] * len(graph_dlist)
        token_dict['edge_feas_dict'] = self.node_edge_type_emb_dict["edge_type_emb_dict"]
        token_dict['node_feas_dict'] = self.node_edge_type_emb_dict["node_type_emb_dict"]

        return token_dict