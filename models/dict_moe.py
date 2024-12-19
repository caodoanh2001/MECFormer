from copy import deepcopy
from torch import Tensor, nn
from typing import List, Optional
import torch
import torch.nn.functional as F
    
class DictMoEGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers

        self.fc1 = nn.Linear(self.input_dim, hidden_size, bias=True)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, init_lambda)

        self.fc3 = nn.Linear(hidden_size, num_experts, bias=True)
        nn.init.normal_(self.fc3.weight, std=0.01)
        nn.init.constant_(self.fc3.bias, init_lambda)

        if num_hidden_layers == 0:
            self.weight = nn.Parameter(torch.ones(num_experts) * init_lambda, requires_grad=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = F.gelu(self.fc2(hidden_states))
        gate_weights = self.fc3(hidden_states)
        return gate_weights