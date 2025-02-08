
from dataloaders.nlp_rw_dataloader import NLPRWDataset
import transformers
import yaml
from model.graph_action_agent import conversation as conversation_lib
from pl_train import DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN


# load dataset configs from config/data_config.yaml
dataset_config = {}
with open("config/data_config.yaml", "r") as f:
    dataset_config = yaml.safe_load(f)


tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/hpc2hdd/home/xzou428/Yuhao/llama3-8b-instruct",
    model_max_length=8192,
    padding_side="right",
    # use_fast=False,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
conversation_lib.default_conversation = conversation_lib.conv_templates["llama-3"]

tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

train_data = NLPRWDataset(
            tokenizer=tokenizer,
            data_path="dataset/NLP_rw_new_batch/instruction",
            data_config=dataset_config["NLP_related_works_new_batch"],
            graph_cfg=dict(
                is_graph=True,
                use_graph_start_end=True,
            ),
        )

train_data_list = []

for i in range(len(train_data)):
    train_data_list.append(train_data[i])