from dataclasses import dataclass


@dataclass
class CLIPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
