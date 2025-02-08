import torch
from lightning.pytorch import LightningModule
from transformers import get_cosine_schedule_with_warmup, AutoConfig
from torch.optim import AdamW
from model.graph_action_agent.model import HeteroGraphLLMForCausalLM
import numpy as np


class GraphAgent_pl(LightningModule):
    def __init__(
        self,
        training_args,
        model_args,
        data_args,
        tokenizer,
        **kwargs,
    ):
        super().__init__()
        self.predict_output_dir = kwargs.get("predict_output_dir", None)

        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        compute_dtype = (
            torch.float16
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )

        bnb_model_from_pretrained_args = {}

        if model_args.graph_tower is not None:
            hf_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            hf_config.graph_hidden_size = model_args.graph_hidden_size
            hf_config.graph_select_layer = model_args.graph_select_layer

            self.model = HeteroGraphLLMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=hf_config,
                **bnb_model_from_pretrained_args,
            )
        else:
            raise ValueError("graph_tower field is required")

        self.model.config.use_cache = False
        if model_args.freeze_backbone:
            self.model.model.requires_grad_(False)

        model_graph_dict = self.model.get_model().initialize_graph_modules(
            graph_tower=model_args.graph_tower,
            graph_select_layer=model_args.graph_select_layer,
            pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter,
            fsdp=None,
        )
        self.model.get_graph_tower().to(dtype=compute_dtype)

        data_args.is_graph = True

        self.model.config.tune_graph_mlp_adapter = (
            training_args.tune_graph_mlp_adapter
        ) = model_args.tune_graph_mlp_adapter
        if model_args.tune_graph_mlp_adapter:
            self.model.requires_grad_(False)
            for p in self.model.get_model().graph_projector.parameters():
                p.requires_grad = True
            if model_args.tune_gnn:
                for p in self.model.get_model().graph_tower.parameters():
                    p.requires_grad = True

        self.model.config.freeze_graph_mlp_adapter = (
            training_args.freeze_graph_mlp_adapter
        )
        if training_args.freeze_graph_mlp_adapter:
            for p in self.model.get_model().graph_projector.parameters():
                p.requires_grad = False

        self.model.config.use_graph_start_end = (
            data_args.use_graph_start_end
        ) = model_args.use_graph_start_end
        training_args.use_graph_start_end = model_args.use_graph_start_end
        self.model.initialize_graph_tokenizer(
            use_graph_start_end=model_args.use_graph_start_end,
            tokenizer=tokenizer,
            device="cuda",
            tune_embed_tokens=model_args.tune_embed_tokens,
            pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter,
            model_args=model_args,
        )

        params_no_grad = [
            n for n, p in self.model.named_parameters() if not p.requires_grad
        ]

        if training_args.bf16:
            self.model.to(torch.bfloat16)
        elif training_args.fp16:
            self.model.to(torch.float16)

        print(
            "************************** parameters: #",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        tuned_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                tuned_params.append(name)
        print(tuned_params)

        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        bs = len(batch["input_ids"])
        batch_id = batch.pop("id")
        loss_dict = self.model(**batch)
        loss = loss_dict["loss"]
        if np.isnan(loss.item()):
            print(batch["input_ids"])
            print(batch["labels"])
            raise ValueError("loss is nan")
        log_dict = {f"train_loss": loss.item()}
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if p.requires_grad
                ],
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_args.learning_rate, eps=1e-3
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
