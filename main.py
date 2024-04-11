from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math


@dataclass
class Config:
    vocab_size: int = 5_000
    window: int = 512
    d_model: int = 768
    layers: int = 12
    p: float = 0.1
    heads: int = 12
    inner_state: int = 3072
    device: str = "cpu"
    n_class: int = 2

class DecoderLayer(nn.Module):
    def __init__(
        self,
        config
    ):
    
        super().__init__()
        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model
        )
        self.linear = nn.Linear(
            config.d_model,
            config.d_model
        )
        self.feed_forward_module = nn.Sequential(
            nn.Linear(config.d_model, config.inner_state),
            nn.GELU(),
            nn.Linear(config.inner_state, config.d_model),
            nn.Dropout(config.p)
        )
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)

        self.head_dim = config.d_model // config.heads
        self.heads = config.heads
        self.dropout = nn.Dropout(p=config.p)

        nn.init.normal_(self.feed_forward_module[0].weight, 0, 0.02)
        nn.init.normal_(self.feed_forward_module[1].weight, 0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        mask = torch.tril(torch.ones((L, L)))
        mask = mask.reshape(B, 1, L, L)
        qkv = self.qkv(x)
        q, k, v = torch.split(tensor=qkv, split_size_or_sections=D, dim=2)
        q = q.reshape(B, L, self.heads, self.head_dim)
        k = k.reshape(B, L, self.heads, self.head_dim)
        v = v.reshape(B, L, self.heads, self.head_dim)

        QKT = torch.einsum("bqhd, bkhd -> bhqk", [q, k]) / math.sqrt(D)
        QKT = QKT.masked_fill(mask == 0, float("-inf"))
        scores = self.dropout(F.softmax(QKT, dim=3))
        output = torch.einsum("bhqk, bvhd -> bqhd", [scores, v])
        output = output.reshape(B, L, D)
        linear = self.dropout(self.linear(output))
        added_norm1 = self.layer_norm1(x + linear)
        added_norm2 = self.layer_norm2(added_norm1 + self.feed_forward_module(added_norm1))
        return added_norm2 



