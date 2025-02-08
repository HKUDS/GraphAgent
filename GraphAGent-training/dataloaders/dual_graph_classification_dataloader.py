import copy
import json
import logging
from typing import Dict
import torch
import transformers
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import os.path as osp
import glob
from .utils import preprocess_graph_Hetero, DEFAULT_GRAPH_TOKEN
from .utils_llama3 import preprocess_llama3

class GraphDataXDict(object):
    def __init__(self, x_dict):
        self.x_dict = x_dict

class DualGraphClassificationDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_config: dict,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict):
        super(DualGraphClassificationDataset, self).__init__()
        logging.warning("Loading data...")

        self.tokenizer = tokenizer
        self.graph_cfg = graph_cfg

        ann_data = json.load(open(data_path, "r", encoding= "utf-8"))

        self.higpt_graph_data = data_config["higpt_graph_data"]
        self.higpt_graph_data_dict = torch.load(self.higpt_graph_data)

        self.skg_graph_data_dict = torch.load(data_config["skg_graph_data"])

        self.ann_data = []
        for ann_entry in ann_data:
            try:
                # import pdb; pdb.set_trace()
                self.higpt_graph_data_dict[ann_entry["id"]]
                self.skg_graph_data_dict[ann_entry["id"]]
            except KeyError:
                logging.warning(f"Graph {ann_entry['id']} not found in the graph file. Skipping.")
            else:
                self.ann_data.append(ann_entry)

        self.hetero_key_order = data_config['hetero_key_order']

        self.node_edge_type_emb_dict = torch.load(data_config['node_edge_type_emb_dict'])

        logging.warning("Formatting inputs...Skip in lazy mode")

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        prompt_entry = self.ann_data[i]
        higpt_graph_dict = self.higpt_graph_data_dict[prompt_entry["id"]]
        skg_graph_dict = self.skg_graph_data_dict[prompt_entry["id"]]

        cur_token_lens = []
        for key in prompt_entry["graph"]["keys_order"]:
            cur_token_lens.append(higpt_graph_dict.x_dict[key].shape[0])
        for key in prompt_entry["skg_graph"]["keys_order"]:
            cur_token_lens.append(skg_graph_dict.x_dict[key].shape[0])
        assert type(cur_token_lens[0]) == int, f"Need to be int, not {type(cur_token_lens[0])}"

        # print("cur_token_lens", cur_token_lens)
        graph_token_num = prompt_entry["conversations"][0]["value"].count(DEFAULT_GRAPH_TOKEN)
        assert (len(cur_token_lens) == graph_token_num), f"Number of tokens in prompt and graph do not match. Len: {len(cur_token_lens)} vs {graph_token_num}"

        sources = preprocess_graph_Hetero(
            copy.deepcopy([prompt_entry["conversations"]]),
            self.graph_cfg, cur_token_lens)

        token_dict = preprocess_llama3(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
            token_dict = dict(input_ids=token_dict["input_ids"][0],
                             labels=token_dict["labels"][0])

        token_dict['id'] = prompt_entry['id']

        merged_graph_dict = {}
        higpt_keys_order = [s + "_higpt" for s in prompt_entry["graph"]["keys_order"]]
        skg_keys_order = [s + "_skg" for s in prompt_entry["skg_graph"]["keys_order"]]
        for key in higpt_graph_dict.x_dict.keys():
            merged_graph_dict[key + "_higpt"] = higpt_graph_dict.x_dict[key]
        for key in skg_graph_dict.x_dict.keys():
            merged_graph_dict[key + "_skg"] = skg_graph_dict.x_dict[key]
        
        merged_graph_dict_pyg = HeteroData()
        for key in merged_graph_dict.keys():
            merged_graph_dict_pyg[key].x = merged_graph_dict[key]

        token_dict['graph_data'] = merged_graph_dict_pyg
        token_dict['hetero_key_order'] = higpt_keys_order + skg_keys_order
        return token_dict