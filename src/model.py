import torch
import torch.nn.functional as F

from typing import List
from torch import nn
from src.config import DinConfig


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_sizes: List[int],
                 activation_layer: str):
        super().__init__()
        dimensions = [input_dim] + hidden_sizes

        # TODO: Implement the Dice activation
        Activation = nn.LeakyReLU if activation_layer == "LeakyReLU" else "Dice"

        layers = [nn.BatchNorm1d(input_dim)]
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(nn.BatchNorm1d(dimensions[i + 1]))
            layers.append(Activation(0.1))
        layers.append(nn.Linear(hidden_sizes[-1], 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LocalActivationUnit(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_sizes: List[int],
                 activation_layer: str):
        super().__init__()

        # TODO: Implement the Dice activation
        Activation = nn.Sigmoid if activation_layer == "Sigmoid" else "Dice"
        dimensions = [8 * embedding_dim] + hidden_sizes
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(Activation())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self,
                current_item: torch.FloatTensor,  # B x 2D
                historical_items: torch.FloatTensor,  # B x T x 2D
                mask: torch.FloatTensor):
        batch_size, seq_len, hidden_dim = historical_items.size()
        current_item = current_item.unsqueeze(1).expand(batch_size, seq_len, hidden_dim)  # B x T x 2D
        combinations = torch.cat([
            current_item,
            historical_items,
            current_item - historical_items,
            current_item * historical_items
        ], dim=2)  # B x T x 8D

        attn_weights = self.model(combinations).squeeze()  # B x T
        padded_value = torch.ones_like(attn_weights) * (-2 ** 31)
        masked_attn_weights = torch.where(mask == 1, attn_weights, padded_value)
        attn_scores = F.softmax(masked_attn_weights, dim=-1) * mask  # B x T
        return torch.matmul(attn_scores.unsqueeze(1), historical_items).squeeze()  # B x D


class DeepInterestNetwork(nn.Module):
    def __init__(self, config: DinConfig):
        super().__init__()
        self.user_embedding_layer = nn.Embedding(config.num_users, config.embedding_dim)
        self.material_embedding_layer = nn.Embedding(config.num_materials, config.embedding_dim)
        self.category_embedding_layer = nn.Embedding(config.num_categories, config.embedding_dim)

        self.lau_layer = LocalActivationUnit(embedding_dim=config.embedding_dim,
                                             hidden_sizes=config.lau_hidden_sizes,
                                             activation_layer=config.lau_activation_layer)

        self.mlp_layer = MLP(input_dim=7 * config.embedding_dim,
                             hidden_sizes=config.mlp_hidden_sizes,
                             activation_layer=config.mlp_activation_layer)

    def forward(self,
                uid: torch.FloatTensor,
                mid: torch.FloatTensor,
                cat: torch.FloatTensor,
                historical_mid: torch.FloatTensor,
                historical_cat: torch.FloatTensor,
                mask: torch.LongTensor):
        user_embedding = self.user_embedding_layer(uid)  # B x D
        material_embedding = self.material_embedding_layer(mid)  # B x D
        category_embedding = self.category_embedding_layer(cat)  # B x D

        historical_material_embedding = self.material_embedding_layer(historical_mid)  # B x T x D
        historical_category_embedding = self.category_embedding_layer(historical_cat)  # B x T x D

        current_item_embedding = torch.cat([material_embedding, category_embedding], dim=1)  # B x T x 2D
        historical_item_embedding = torch.cat([historical_material_embedding,
                                               historical_category_embedding], dim=2)  # B x T x 2D

        lau_embedding = self.lau_layer(current_item_embedding, historical_item_embedding, mask)

        historical_item_embedding_sum = (historical_item_embedding * mask.unsqueeze(2)).sum(dim=1)
        historical_item_embedding_mean = historical_item_embedding_sum / mask.sum(dim=1, keepdim=True)

        combinations = torch.cat([
            user_embedding,
            current_item_embedding,
            lau_embedding,
            historical_item_embedding_mean
        ], dim=-1)  # B x T x 7D

        output_logits = self.mlp_layer(combinations)
        return output_logits
