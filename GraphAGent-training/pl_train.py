import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch

import transformers
from torch.utils.data import DataLoader

from model.graph_action_agent import conversation as conversation_lib

from lightning.pytorch.strategies import FSDPStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from lightning.pytorch import Trainer, seed_everything
from model.graph_action_agent.pl_model import GraphAgent_pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
from lightning import LightningDataModule
from torch import nn
from dataloaders.abstract_dataloader import ProcessedDataset
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import ConcatDataset

IGNORE_INDEX = -100
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=True)
    tune_embed_tokens: bool = field(default=True)
    full_finetune: bool = field(default=True)
    graph_tower: Optional[str] = field(default="MetaHGT_imdb_dblp_epoch5")
    graph_select_layer: Optional[int] = field(default=-2)
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=True)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")
    tune_gnn: bool = field(default=False)
    graph_hidden_size: int = field(default=768)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = True
    is_graph: bool = True
    graph_root: Optional[str] = field(default=None)
    hetero_key_path: Optional[str] = field(default=None)
    num_shot: Optional[int] = field(default=0)
    data_name: Optional[str] = field(default=None)


@dataclass
class TrainingArguments:
    gpus: Optional[str] = field(default="0,1,2,3")

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    strategy: str = field(default="fsdp")

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool = False

    resume: Optional[str] = field(default=None)

    adam_epsilon: float = field(default=1e-8)
    warmup_steps: int = field(default=0)
    num_workers: int = field(default=16)

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    output_dir: str = field(default="./checkpoints/graphchat-gt-graphmatch-7b")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)

    save_every_n_epochs: int = field(default=2)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=1)
    tf32: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)

    flash_attn: bool = field(default=True)


class SaveGraphProjectorCallback(Callback):
    def __init__(self, output_dir, keys_to_match):
        self.output_dir = output_dir
        self.keys_to_match = keys_to_match

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # 准备保存模型权重
        _state_dict = pl_module.state_dict()

        weight_to_save = {}
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in self.keys_to_match):
                weight_to_save[k] = v

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存 graph projector 的权重
        torch.save(weight_to_save, os.path.join(self.output_dir, "graph_projector.bin"))


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "graph_data" in instances[0]:
            graph_data_batch = [instance["graph_data"] for instance in instances]
            key_order_batch = [instance["hetero_key_order"] for instance in instances]

        batch["graph_data"] = graph_data_batch
        batch["hetero_key_order"] = key_order_batch
        batch["id"] = [instance["id"] for instance in instances]

        return batch


class GraphAgentLitDataModule(LightningDataModule):
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizer, data_args, training_args
    ) -> None:
        super().__init__()
        self.training_args = training_args
        self.data_args = data_args
        data_name = data_args.data_name

        self.loaded_data = None

        if data_name == "stage_2_mix":
            self.pr_expert_data = ProcessedDataset(
                "dataset/ICLR_peer_review_stage2/train_data_list_8192_llama3.pt"
            )

            self.sc_expert_data = ProcessedDataset(
                "dataset/Arxiv2023_instruction_stage2/train_data_list_8192_llama3.pt"
            )

            self.expert_data = ProcessedDataset(
                "dataset/NLP_rw_new_batch/instruction/train_data_list_1500_8192_ans_gt_256_llama3.pt"
            )

            self.stage_2_mix_data = ConcatDataset(
                [self.expert_data, self.pr_expert_data, self.sc_expert_data]
            )
            self.loaded_data = self.stage_2_mix_data
            print("*********** Training with stage2 MIX data ***********")

        elif data_name == "stage_1_mix_with_higpt":
            stage_1_higpt_data = ConcatDataset(
                [
                    ProcessedDataset(
                        "dataset/stage_1/instruct_ds_node_matching_imdb/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/stage_1/instruct_ds_matching_movie/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/stage_1/instruct_ds_matching_author/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/stage_1/instruct_ds_matching_paper/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/stage_1/instruct_ds_node_matching_DBLP/train_data_list_8192.pt"
                    ),
                ]
            )
            stage_1_rw_data = ConcatDataset(
                [
                    ProcessedDataset(
                        "dataset/NLP_rw_stage_1/instruction_meta_node_cross_type/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/NLP_rw_stage_1/instruction_meta_node_key_results/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/NLP_rw_stage_1/instruction_meta_node_methodology/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/NLP_rw_stage_1/instruction_meta_node_paper/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/NLP_rw_stage_1/instruction_meta_node_research_background/train_data_list_8192.pt"
                    ),
                    ProcessedDataset(
                        "dataset/NLP_rw_stage_1/instruction_meta_node_research_question/train_data_list_8192.pt"
                    ),
                ]
            )
            self.loaded_data = ConcatDataset([stage_1_higpt_data, stage_1_rw_data])
            print("*********** Training with stage1 HiGPT+RW MIX data ***********")

        elif data_name.startswith("stage_2_dual_graph_imdb_few_shot"):
            num_shots = int(data_name.split("_")[-1])
            self.loaded_data = ProcessedDataset(
                f"dataset/IMDB_fewshot_train/{num_shots}_shot_train.pt"
            )
            print(
                f"*********** Training with stage2 dual graph imdb few shot data with {num_shots} shots ***********"
            )

        else:
            raise ValueError(f"Unsupported data name: {data_name}")

        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.loaded_data,
            batch_size=self.training_args.per_device_train_batch_size,
            num_workers=self.training_args.num_workers,
            collate_fn=self.data_collator,
            prefetch_factor=4,
            pin_memory=True,
            shuffle=True,
        )


def train():
    seed_everything(42)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.flash_attn:
        from model.graph_action_agent.llama2_flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )

        replace_llama_attn_with_flash_attn()

    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(",")]
    devices = training_args.gpus
    num_devices = len(devices)
    batch_size = training_args.per_device_train_batch_size * num_devices

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if model_args.version == "v1":
        eot = "<|eot_id|>"
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        tokenizer.pad_token = eot
        tokenizer.pad_token_id = eot_id
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "llama-3"
        ]
    else:
        raise ValueError(f"Unsupported template version: {model_args.version}")

    model = GraphAgent_pl(training_args, model_args, data_args, tokenizer)

    data_module = GraphAgentLitDataModule(tokenizer, data_args, training_args)

    if num_devices > 1:
        if training_args.strategy == "fsdp":
            strategy = FSDPStrategy(
                auto_wrap_policy={LlamaDecoderLayer, nn.Embedding},  #
                activation_checkpointing_policy={LlamaDecoderLayer, nn.Embedding}
                if training_args.gradient_checkpointing
                else None,  #
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=True,
            )
        else:
            strategy = training_args.strategy
    else:
        strategy = "auto"

    csv_logger = CSVLogger(save_dir=training_args.output_dir)

    model_precision = (
        "16" if training_args.fp16 else ("bf16" if training_args.bf16 else "32")
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=csv_logger.log_dir,
        filename=model_args.model_save_name,
        monitor="train_loss",
        every_n_epochs=training_args.save_every_n_epochs,
        save_top_k=-1,
        save_weights_only=True,
    )

    trainer = Trainer(
        default_root_dir=training_args.output_dir,
        max_epochs=int(training_args.num_train_epochs),
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        logger=[csv_logger],
        precision=model_precision,
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
    )
    resume = training_args.resume
    if resume:
        model.load_state_dict(torch.load(resume)["state_dict"])
        print(f"******************* Loaded from {resume} *******************")

    print("******************* Params that are NOT bf16 *******************")
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            print(name, param.dtype)

    print(
        "******************* Params that are enabled for updating *******************"
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
