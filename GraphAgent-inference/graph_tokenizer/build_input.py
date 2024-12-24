from graph_tokenizer.graph_tokenizer import DEFAULT_GRAPH_TOKEN
from graph_tokenizer.utils import preprocess_graph_Hetero, preprocess_llama3_inference
import copy
import torch

def build_prompt(user_instruction, pyg_graph, task_type):
    node_type_repr = []
    hetero_keys_order = []

    for node_type in pyg_graph.node_types:
        hetero_keys_order.append(node_type)
        node_type_repr_i = f"\"{node_type}\" nodes: {DEFAULT_GRAPH_TOKEN}"
        node_type_repr.append(node_type_repr_i)

    node_type_repr = "; ".join(node_type_repr)

    prompt_template = f"System: You are a powerful AI assistant facilited with the ability to read and comprehend graphs. You can help with diverse tasks, where graphs serve as useful references and structural representations. The current task is: {task_type}\nUser: \n{user_instruction}\nHeterogeneous Knowledge Graph: {node_type_repr}"

    return prompt_template, hetero_keys_order

def build_input(user_instruction, pyg_graph, task_type, tokenizer):
    prompt_text, hetero_keys_order = build_prompt(user_instruction, pyg_graph, task_type)

    # import pdb; pdb.set_trace()
    cur_token_lens = []
    for key in hetero_keys_order:
        cur_token_lens.append(pyg_graph.x_dict[key].shape[0])
    assert type(cur_token_lens[0]) == int, f"Need to be int, not {type(cur_token_lens[0])}"

    convs = [{"from": "human", "value": prompt_text}]

    sources = preprocess_graph_Hetero(
        copy.deepcopy([convs]), cur_token_lens)

    token_dict = preprocess_llama3_inference(
        sources,
        tokenizer)
    
    # add chat template
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assistant_start_tokens = tokenizer.encode(assistant_start, return_tensors="pt", add_special_tokens=False)
    token_dict["input_ids"] = torch.cat((token_dict["input_ids"], assistant_start_tokens), dim=1)
    token_dict["attention_mask"] = token_dict["input_ids"].ne(tokenizer.pad_token_id)

    # print(tokenizer.batch_decode(token_dict["input_ids"], skip_special_tokens=False))
    # print(token_dict["attention_mask"].tolist())
    
    return {
        **token_dict,
        "prompt": prompt_text,
        "graph_data": pyg_graph,
        "hetero_key_order": hetero_keys_order
    }