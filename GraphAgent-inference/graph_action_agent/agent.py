from graph_action_agent.graphllm.graphllm import HeteroGraphLLMForCausalLM
import transformers
from transformers import get_cosine_schedule_with_warmup, StoppingCriteria, AutoConfig, pipeline
import torch
from graph_tokenizer.build_input import build_input
from graph_tokenizer.graph_tokenizer import DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import os
from colorama import Fore, Style

class GraphActionAgent:
    def __init__(self):
        model_path = os.environ['GRAPH_ACTION_MODEL_PATH']
        
        hf_config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            self.llm = HeteroGraphLLMForCausalLM(config=hf_config)

        self.llm.model.graph_tower = "MetaHGT_imdb_dblp_epoch5"
        self.llm = load_checkpoint_and_dispatch(self.llm, model_path, device_map="auto", dtype=torch.bfloat16, no_split_module_classes=["LlamaDecoderLayer"])
        self.llm.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=8192,
        )
        eot = "<|eot_id|>"
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        tokenizer.pad_token = eot
        tokenizer.pad_token_id = eot_id
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
        self.tokenizer = tokenizer

    def invoke(self, user_instruction, pyg_graph, task_type):
        action_agent_input = build_input(user_instruction, pyg_graph, task_type, self.tokenizer)
        action_agent_output = self.inference(action_agent_input)
        return action_agent_output

    @torch.no_grad()
    def inference(self, input):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        input_ids = input["input_ids"].to(self.llm.device)

        output_ids = self.llm.generate(
                input_ids,
                attention_mask=input["attention_mask"],
                graph_data=input["graph_data"],
                hetero_key_order = input["hetero_key_order"], 
                do_sample=True,
                temperature=0.6,
                max_new_tokens=2048,
                top_p=0.9,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
        )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs