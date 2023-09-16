from dataclasses import dataclass
from typing import List


@dataclass
class DinConfig:
    num_users: int
    num_materials: int
    num_categories: int
    embedding_dim: int
    lau_hidden_sizes: List[int]
    lau_activation_layer: str
    mlp_hidden_sizes: List[int]
    mlp_activation_layer: str