from typing import Dict, Sequence, List


import transformers

from graph_action_agent.graphllm import conversation as conversation_lib


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def preprocess_graph_Hetero(
    sources: Sequence[str],
    cur_token_lens: List[int],
    use_graph_start_end: bool = True,
) -> Dict:
    graph_token_lens = cur_token_lens

    for source in sources:

        for sentence in source:
            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                # build replace_tokens
                replace_tokens = []
                for i, token_len in enumerate(graph_token_lens):
                    replace_token = DEFAULT_GRAPH_PATCH_TOKEN * token_len
                    if use_graph_start_end:
                        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
                    replace_tokens.append(replace_token)

                for i, replace_token in enumerate(replace_tokens):
                    index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                    sentence["value"] = sentence["value"][:index] + replace_token + sentence["value"][index+len(DEFAULT_GRAPH_TOKEN):]

    return sources


from multiprocessing import Pool
from typing import Dict, Sequence

import torch
from graph_action_agent.graphllm import conversation as conversation_lib
from graph_action_agent.graphllm.conversation import SeparatorStyle
import transformers

IGNORE_TOKEN_ID = -100


def apply_prompt_template(sources, systems=None):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if systems and systems[i]:
            conv.set_system_message(systems[i])
        prompt = conv.get_prompt()
        conversations.append(prompt)
    return conversations, conv


def tokenize_conversations(conversations, tokenizer):
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False # This is important https://github.com/lm-sys/FastChat/pull/3289
    ).input_ids
    targets = input_ids.clone()
    return input_ids, targets


def get_prompt_separator(conv):
    if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        user_turn_separator = conv.sep2
        assistant_turn_separator = conv.roles[1] + ": "

    elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        user_turn_separator = conv.sep2
        assistant_turn_separator = conv.roles[1] + ": "

    elif conv.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
        if conv.sep2 is None:
            user_turn_separator = conv.roles[0] + ": "
        else:
            user_turn_separator = conv.sep2

        assistant_turn_separator = conv.roles[1] + ": "

    elif conv.sep_style == SeparatorStyle.LLAMA2:
        user_turn_separator = conv.sep2
        assistant_turn_separator = conv.roles[1] + " "

    elif conv.sep_style == SeparatorStyle.LLAMA3:
        user_turn_separator = f"<|start_header_id|>{conv.roles[0]}<|end_header_id|>"
        assistant_turn_separator = (
            f"<|start_header_id|>{conv.roles[1]}<|end_header_id|>"
        )

    elif conv.sep_style == SeparatorStyle.CHATML:
        if conv.sep2 is None:
            user_turn_separator = conv.sep + "\n"
        else:
            user_turn_separator = conv.sep2 + "\n"

        assistant_turn_separator = conv.roles[1] + "\n"

    return user_turn_separator, assistant_turn_separator


def mask_targets(conversations, targets, tokenizer, conv):
    # import pdb; pdb.set_trace()
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # if tokenizer.eos_token is None:
        #     cur_len = 0
        # elif tokenizer.eos_token is not None and target[0] != tokenizer.bos_token_id:
        #     cur_len = 0
        # elif tokenizer.eos_token is not None and target[0] == tokenizer.bos_token_id:
        #     cur_len = 1

        cur_len = 1
        # target[:cur_len] = IGNORE_TOKEN_ID
        target[:cur_len] = IGNORE_TOKEN_ID
        user_turn_separator, assistant_turn_separator = get_prompt_separator(conv)
        turns = conversation.split(user_turn_separator)
        for i, turn in enumerate(turns):
            if (
                i < len(turns) - 1 and turn == ""
            ):  # Last turn is the user_turn_separator
                break

            if (
                tokenizer.bos_token is not None and turn == tokenizer.bos_token
            ):  # Already masked
                continue

            if i != 0:
                turn = user_turn_separator + turn

            turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)

            if assistant_turn_separator in turn:
                parts = turn.rsplit(assistant_turn_separator)
                parts[0] += assistant_turn_separator
            else:
                parts = [turn]

            instruction_len = len(
                tokenizer(parts[0], add_special_tokens=False).input_ids
            )

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len+2: # Important: plus two <eot_id>
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return targets


def preprocess_llama3(
    sources, tokenizer: transformers.PreTrainedTokenizer, **kwargs
) -> Dict:
    systems = None if not kwargs else kwargs.get("systems", None)

    # If the data volume is small, process it directly in the main thread
    if len(sources) <= 1000:
    # if True:
        conversations, conv = apply_prompt_template(sources, systems)
        # print(f"Conversations: {conversations}")
        # print(f"Conv: {conv}")
        input_ids, targets = tokenize_conversations(conversations, tokenizer)
        targets = mask_targets(conversations, targets, tokenizer, conv)
    else:  # If the data volume is large, use multithreading for processing
        with Pool() as p:
            conversations, conv = p.apply_async(
                apply_prompt_template, (sources, systems)
            ).get()
            input_ids, targets = p.apply_async(
                tokenize_conversations, (conversations, tokenizer)
            ).get()
            targets = p.apply_async(
                mask_targets, (conversations, targets, tokenizer, conv)
            ).get()
            p.close()
            p.join()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_llama3_inference(
    sources, tokenizer: transformers.PreTrainedTokenizer, **kwargs
) -> Dict:
    systems = None if not kwargs else kwargs.get("systems", None)

    conversations, conv = apply_prompt_template(sources, systems)
    input_ids, _ = tokenize_conversations(conversations, tokenizer)

    return dict(
        input_ids=input_ids,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )