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

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.window, config.d_model)
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers)])
        self.dropout = nn.Dropout(p=config.p)
        
        nn.init.normal_(self.word_embedding.weight, 0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, L = x.shape
        positions = torch.arange(0, L).expand(B, L).to(self.config.device)
        output = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.decoder:
            output = layer(output)

        return output

class LMHead(nn.Module):
    def __init__(self, config, gpt):
        super().__init__()
        self.gpt = gpt
        self.prediction = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.prediction.weights = gpt.word_embedding.weight

    def forward(self, x):
        out = self.gpt(x)
        logits = self.prediction(out)
        return logits

class CLSHead(nn.Module):
    def __init__(
        self,
        config,
        gpt
        ):

        super().__init__()
        self.gpt = gpt
        self.prediction = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.prediction.weights = gpt.word_emb.weight
        self.classifier = nn.Linear(config.d_model, config.n_class)

        nn.init.normal_(self.classifier.weight, std=0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        dec_out = self.gpt(x)

        lm_logits = self.prediction(dec_out)
        cls_logits = self.classifier(dec_out)
        return lm_logits, cls_logits

if __name__ == "__main__":
    config = Config()
    gpt = GPT(config)
    lm_test = LMHead(config, gpt)
    cls_test = CLSHead(config, gpt)
    logits = lm_test(torch.randint(0, config.vocab_size, (1, config.window)))
    print(logits.shape)
    lm_logits, cls_logits = cls_test(torch.randint(0, config.vocab_size, (1, config.window)))
    print(lm_logits.shape, cls_logits.shape)
