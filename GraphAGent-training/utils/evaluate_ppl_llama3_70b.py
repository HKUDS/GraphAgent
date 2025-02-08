# %%
import evaluate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from torch.nn import CrossEntropyLoss
from evaluate import logging
import numpy as np
from pathlib import Path
import json


class Perplexity():

    def compute(
        self, predictions, model, tokenizer, batch_size: int = 4, add_start_token: bool = True, max_length=None,
        output_dir = None
    ):

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0))
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()
            if output_dir is not None:
                with open(output_dir / "ppls.json", "w") as f:
                    json.dump({"perplexities": ppls}, f)

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

perplexity = Perplexity()

model = AutoModelForCausalLM.from_pretrained(
                "/hpc2hdd/home/xzou428/Yuhao/llama3-70b-instruct", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
tokenizer = AutoTokenizer.from_pretrained(
            "/hpc2hdd/home/xzou428/Yuhao/llama3-70b-instruct"
        )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
# %cd /hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/
import os 
os.chdir("/hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/")
from pathlib import Path
import torch
import json

res_dict = {}
predict_file = "inference/llama3-70b_nlp_new_batch_rw_vanilla/all_res.json"
ground_truth_file = "dataset/NLP_rw_new_batch/instruction/test_data_list_1500_8192_ans_gt_256_llama3.pt"

predict_res = json.load(open(predict_file))
predict_res = {int(list(entry.keys())[0]):list(entry.values())[0] for entry in predict_res}

ground_truths = torch.load(ground_truth_file)
for data_entry in ground_truths:
    if data_entry["id"] in predict_res:
        res_dict[data_entry["id"]] = {
            "predict": predict_res[data_entry["id"]],
            "ground_truth": data_entry["ground_truth"]
        }

print(len(res_dict))
res_dict = {k: v for k, v in res_dict.items() if "ground_truth" in v}
print(len(res_dict))

predictions = [v["predict"] for v in res_dict.values()]

results = perplexity.compute(predictions=predictions, model=model, tokenizer=tokenizer, batch_size=2,
                                output_dir = Path(predict_file).parent)
print(results)


# %%
import os 
os.chdir("/hpc2hdd/home/xzou428/Yuhao/HiGPT-tune-lightning/")
from pathlib import Path
import torch
import json

res_dict = {}
predict_file = "inference/llama3_nlp_new_batch_rw_vanilla/all_res.json"
ground_truth_file = "dataset/NLP_rw_new_batch/instruction/test_data_list_1500_8192_ans_gt_256_llama3.pt"

predict_res = json.load(open(predict_file))
predict_res = {int(list(entry.keys())[0]):list(entry.values())[0] for entry in predict_res}

ground_truths = torch.load(ground_truth_file)
for data_entry in ground_truths:
    if data_entry["id"] in predict_res:
        res_dict[data_entry["id"]] = {
            "predict": predict_res[data_entry["id"]][0]['generated_text'][1]["content"],
            "ground_truth": data_entry["ground_truth"]
        }

print(len(res_dict))
res_dict = {k: v for k, v in res_dict.items() if "ground_truth" in v}
print(len(res_dict))

predictions = [v["predict"] for v in res_dict.values()]

results = perplexity.compute(predictions=predictions, model=model, tokenizer=tokenizer, batch_size=2,
                                output_dir = Path(predict_file).parent)
print(results)