import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from .meta_linear import MetaHeteroDictLinear, MetaHeteroLinear
from torch_geometric.typing import Adj, EdgeType, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index
from dataclasses import dataclass
import torch.nn as nn
from pathlib import Path
from transformers.configuration_utils import PretrainedConfig

wd = Path(__file__).resolve().parent
@dataclass
class MetaHGTConvCfg:
    in_channels: int 
    out_channels: int
    heads: int
    dynamic: bool = True

def check_nan(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

class MetaHGTConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dynamic: bool = False,
        text_cfg = None, 
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.config = PretrainedConfig()

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        

        self.kqv_lin = MetaHeteroDictLinear(text_cfg.width, self.in_channels,
                                        self.out_channels * 3, dynamic)

        self.out_lin = MetaHeteroDictLinear(text_cfg.width, self.out_channels, self.out_channels, dynamic)
        self.context_length = text_cfg.context_length

        dim = out_channels // heads

        self.k_rel = MetaHeteroLinear(text_cfg.width, dim, dim, dynamic)
        self.v_rel = MetaHeteroLinear(text_cfg.width, dim, dim, dynamic)

        self.skipTrans = nn.Linear(text_cfg.width, 1) # node aware, skip: 1

        self.p_relTrans = nn.Linear(text_cfg.width, heads) # edge aware, p_rel: 1, heads
        # self.ln_final = layernorm(text_cfg.width)

        self.norm = nn.LayerNorm(self.out_channels, eps=1e-6)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj], 
        edge_type_feas_dict: Dict[EdgeType, Tensor], 
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(edge_index_dict.keys())
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}

        edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(edge_index_dict.keys())
        }
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            # import pdb; pdb.set_trace()
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        edge_feas_dict = {edge_types_map[k]: v for k, v in edge_type_feas_dict.items()}

        k = self.k_rel(ks, type_vec, edge_feas_dict).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec, edge_feas_dict).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def _construct_p_rel(self, edge_type_feas_dict: Dict[EdgeType, Tensor]):
        p_rel = {k: self.p_relTrans(v).unsqueeze(0) for k, v in edge_type_feas_dict.items()}
        return p_rel
    def _construct_skip(self, node_type_feas_dict: Dict[EdgeType, Tensor]):
        skip = {k: self.skipTrans(v) for k, v in node_type_feas_dict.items()}
        return skip

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        data_type: str = 'dblp', 
        node_type_feas_dict: Dict[NodeType, Tensor] = None,
        edge_type_feas_dict: Dict[EdgeType, Tensor] = None,
    ) -> Dict[NodeType, Optional[Tensor]]:

        
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # import pdb; pdb.set_trace()
        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict, node_type_feas_dict)

        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            # check_nan(k_dict[key], "k_dict")
            q_dict[key] = q.view(-1, H, D)
            # check_nan(q_dict[key], "q_dict")
            v_dict[key] = v.view(-1, H, D)
            # check_nan(v_dict[key], "v_dict")

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict, edge_type_feas_dict)
        p_rel = self._construct_p_rel(edge_type_feas_dict)
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=p_rel)
        # check_nan(k, 'k before')
        # check_nan(q, 'q before')
        # check_nan(v, 'v before')
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)
        # check_nan(out, 'out after')

        dst_node_types = set([key[-1] for key in edge_index_dict.keys()])

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            # if node_type in dst_node_types:
            out_dict[node_type] = out[start_offset:end_offset]
        # for k, v in out_dict.items(): 
        #     check_nan(v, f'out_dict[{k}]')
        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            # torch.nn.functional.gelu(v) if v is not None else v
            v if v is not None else v
            for k, v in out_dict.items()
        }, node_type_feas_dict)

        skip = self._construct_skip(node_type_feas_dict)
        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]
            # check_nan(out, f'out_dict[{node_type}]')

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
                # check_nan(out, f'out')
            # check_nan(out, f'before norm')
            out = self.norm(out)
            # check_nan(out, f'after norm')
            out_dict[node_type] = out
        
        # import pdb; pdb.set_trace()
        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
