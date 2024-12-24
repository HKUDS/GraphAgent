from typing import Dict, List

import torch
from torch import Tensor
import torch.nn as nn

init = nn.init.xavier_uniform_

class ParameterGenerator(nn.Module):
    def __init__(self, memory_size, in_channels, out_channels, hidden_channels, dynamic):
        super(ParameterGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dynamic = dynamic

        if self.dynamic:
            print('Using DYNAMIC')
            self.weight_generator = nn.Sequential(*[
                nn.Linear(memory_size, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels * out_channels)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(memory_size, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            ])
        else:
            print('Using FC')
            self.weights = nn.Parameter(init(torch.empty(in_channels, out_channels)), requires_grad=True)
            self.biases = nn.Parameter(init(torch.empty(out_channels)), requires_grad=True)

    def forward(self, memory=None):
        if self.dynamic:
            weights = self.weight_generator(memory).view(self.in_channels, self.out_channels)
            biases = self.bias_generator(memory).view(self.out_channels)
        else:
            weights = self.weights
            biases = self.biases
        return weights, biases

class LinearCustom(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LinearCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inputs, parameters: List[Tensor]):
        weights, biases = parameters[0], parameters[1]
        assert weights.shape == torch.Size([self.in_channels, self.out_channels]) and biases.shape == torch.Size([self.out_channels])
        return torch.matmul(inputs, weights) + biases

class MetaHeteroLinear(torch.nn.Module):
    def __init__(
        self,
        memory_size: int,
        in_channels: int,
        out_channels: int,
        dynamic: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

        self.meta_lin = LinearCustom(self.in_channels, self.out_channels)
        self.lin_gen = ParameterGenerator(self.memory_size, self.in_channels, self.out_channels, self.memory_size //2, dynamic)

    def forward(self, x: Tensor, type_vec: Tensor, edge_feas_dict: Dict) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor): A vector that maps each entry to a type.
        """
        out = x.new_empty(x.size(0), self.out_channels)
        for i in edge_feas_dict.keys():
            mask = type_vec == i
            if mask.numel() == 0:
                continue
            params = self.lin_gen(edge_feas_dict[i])
            subset_out = self.meta_lin(x[mask], params)
            # The data type may have changed with mixed precision:
            out[mask] = subset_out.to(out.dtype)

        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_types={self.num_types}, '
                f'bias={self.kwargs.get("bias", True)})')


class MetaHeteroDictLinear(torch.nn.Module):
    def __init__(
        self,
        memory_size: int,
        in_channels: int,
        out_channels: int,
        dynamic: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

        self.lin_gen = ParameterGenerator(self.memory_size, self.in_channels, self.out_channels, self.memory_size //2, dynamic)
        
        self.meta_lin = LinearCustom(self.in_channels, self.out_channels)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        node_feas_dict: Dict, 
    ) -> Dict[str, Tensor]:
        r"""
        Args:
            x_dict (Dict[Any, torch.Tensor]): A dictionary holding input
                features for each individual type.
        """
        out_dict = {}

        for key, node_feas in node_feas_dict.items():
            if key in x_dict:
                params = self.lin_gen(node_feas)
                out_dict[key] = self.meta_lin(x_dict[key], params)

        return out_dict

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.kwargs.get("bias", True)})')


