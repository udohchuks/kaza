"""Audio input embeddings for Kaza (12 parallel GFSQ groups per stream)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NUM_CODEBOOKS = 12
VOCAB_SIZE = 2016
BOS_TOKEN_ID = VOCAB_SIZE  # 2016 (Moshi's initial_token_id)
ZERO_TOKEN_ID = -1  # Moshi's zero_idx: maps to exact 0 vector (no gradient)


def shift_agent_tokens(target_tokens: torch.Tensor) -> torch.Tensor:
    """Shift target frames [B, T, 12] right by 1, prepending BOS at t=0."""
    bos = torch.full(
        (target_tokens.shape[0], 1, target_tokens.shape[2]),
        BOS_TOKEN_ID,
        dtype=target_tokens.dtype,
        device=target_tokens.device,
    )
    return torch.cat([bos, target_tokens[:, :-1, :]], dim=1)


class StreamEmbedding(nn.Module):
    """12 separate embedding tables [2017, d_model] summed across the 12 GFSQ slots."""

    def __init__(
        self,
        d_model: int = 2048,
        num_codebooks: int = NUM_CODEBOOKS,
        vocab_size: int = VOCAB_SIZE,
        init_std: float = 0.02,
        zero_init: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.tables = nn.ModuleList(
            [nn.Embedding(vocab_size + 1, d_model) for _ in range(num_codebooks)]
        )
        std = 0.0 if zero_init else init_std / math.sqrt(num_codebooks)
        for table in self.tables:
            if zero_init:
                nn.init.zeros_(table.weight)
            else:
                nn.init.normal_(table.weight, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens [B, T, 12] in {-1, 0..2016} -> [B, T, d_model]."""
        out = torch.zeros(
            (*tokens.shape[:2], self.d_model),
            dtype=self.tables[0].weight.dtype,
            device=tokens.device,
        )
        for i, table in enumerate(self.tables):
            ids = tokens[:, :, i]
            is_zero = ids == ZERO_TOKEN_ID
            emb = table(ids.masked_fill(is_zero, 0))
            out = out + emb.masked_fill(is_zero.unsqueeze(-1), 0.0)
        return out


class KazaInputEmbedding(nn.Module):
    """Combines agent t-1 (and optional Phase 2 user t) into [B, T, d_model]."""

    def __init__(
        self,
        d_model: int = 2048,
        use_user_stream: bool = False,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.agent = StreamEmbedding(d_model=d_model)
        self.user = StreamEmbedding(d_model=d_model, zero_init=True) if use_user_stream else None

    def forward(
        self,
        agent_in: torch.Tensor,
        user_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training and self.drop_prob > 0.0:
            drop = (
                (torch.rand(agent_in.shape, device=agent_in.device) < self.drop_prob)
                & (agent_in != BOS_TOKEN_ID)
                & (agent_in != ZERO_TOKEN_ID)
            )
            agent_in = agent_in.masked_fill(drop, ZERO_TOKEN_ID)

        x = self.agent(agent_in)
        if self.user is not None and user_in is not None:
            x = x + self.user(user_in)
        return x
