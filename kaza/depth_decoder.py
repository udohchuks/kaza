"""MaskGIT Depth Decoder: 3 bidirectional layers predicting 12 parallel LiquidCodec tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaza.embeddings import NUM_CODEBOOKS, VOCAB_SIZE

MASK_TOKEN_ID = VOCAB_SIZE  # 2016 ([MASK] token for the 12 depth-decoder slots)


def sample_cosine_mask(
    shape: tuple[int, ...],
    device: torch.device,
    num_codebooks: int = NUM_CODEBOOKS,
    mask_all: bool = False,
) -> torch.Tensor:
    """Sample a boolean mask [*shape, K] with k = ceil(K * cos(pi/2 * u)) masked slots per frame."""
    if mask_all:
        return torch.ones((*shape, num_codebooks), dtype=torch.bool, device=device)

    u = torch.rand(shape, device=device)
    k = torch.ceil(num_codebooks * torch.cos(0.5 * math.pi * u)).long().clamp(1, num_codebooks)
    ranks = torch.rand((*shape, num_codebooks), device=device).argsort(dim=-1).argsort(dim=-1)
    return ranks < k.unsqueeze(-1)


class MaskGITDepthDecoder(nn.Module):
    """3-layer width-1024 bidirectional transformer predicting 12 parallel GFSQ tokens."""

    def __init__(
        self,
        d_backbone: int = 2048,
        d_model: int = 1024,
        n_layers: int = 3,
        n_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.0,
        num_codebooks: int = NUM_CODEBOOKS,
        vocab_size: int = VOCAB_SIZE,
        mtp_offsets: tuple[int, ...] = (),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.mtp_offsets = tuple(mtp_offsets)

        self.context_proj = nn.Linear(d_backbone, d_model)
        self.slot_embeds = nn.ModuleList(
            [nn.Embedding(vocab_size + 1, d_model) for _ in range(num_codebooks)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)])
        self.mtp_projs = nn.ModuleDict(
            {
                str(d): nn.Sequential(
                    nn.Linear(d_backbone, 512),
                    nn.GELU(),
                    nn.Linear(512, d_model),
                )
                for d in self.mtp_offsets
            }
        )

    def forward(self, context: torch.Tensor, masked_tokens: torch.Tensor) -> torch.Tensor:
        """Predict logits [..., 12, 2016] from backbone context [..., 2048] and tokens [..., 12]."""
        lead = context.shape[:-1]
        ctx = self.context_proj(context).reshape(-1, 1, self.d_model)  # [N, 1, d_model]
        tok = masked_tokens.reshape(-1, self.num_codebooks)  # [N, 12]

        slots = torch.stack([emb(tok[:, k]) for k, emb in enumerate(self.slot_embeds)], dim=1)
        h = self.norm(self.encoder(slots + ctx))  # [N, 12, d_model]
        logits = torch.stack([head(h[:, k, :]) for k, head in enumerate(self.heads)], dim=1)
        return logits.reshape(*lead, self.num_codebooks, self.vocab_size)

    def compute_loss(
        self,
        context: torch.Tensor,
        target_tokens: torch.Tensor,
        mask_all: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MaskGIT cross-entropy loss and accuracy on masked positions only."""
        mask = sample_cosine_mask(
            target_tokens.shape[:-1],
            device=target_tokens.device,
            num_codebooks=self.num_codebooks,
            mask_all=mask_all,
        )
        masked_tokens = target_tokens.masked_fill(mask, MASK_TOKEN_ID)
        logits = self.forward(context, masked_tokens)

        masked_logits = logits[mask].float()
        masked_targets = target_tokens[mask]
        loss = F.cross_entropy(masked_logits, masked_targets)
        acc = (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
        return loss, acc

    def compute_mtp_loss(
        self,
        context: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only Multi-Frame Prediction loss on future offsets (e.g. t+2, t+4)."""
        if not self.mtp_projs:
            return context.new_zeros(())

        T = target_tokens.shape[1]
        losses = []
        for d_str, proj in self.mtp_projs.items():
            d = int(d_str)
            if T <= d:
                continue
            h = self.norm(proj(context[:, :-d]))  # [B, T-d, d_model]
            tgt = target_tokens[:, d:]  # [B, T-d, 12]
            slot_loss = sum(
                F.cross_entropy(
                    head(h).reshape(-1, self.vocab_size).float(),
                    tgt[..., k].reshape(-1),
                )
                for k, head in enumerate(self.heads)
            ) / self.num_codebooks
            losses.append(slot_loss)

        if not losses:
            return context.new_zeros(())
        return sum(losses) / len(losses)

    @torch.no_grad()
    def decode(
        self,
        context: torch.Tensor,
        num_passes: int = 3,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Iteratively unmask all 12 tokens from context [..., 2048] -> tokens [..., 12]."""
        lead = context.shape[:-1]
        ctx_flat = context.reshape(-1, context.shape[-1])
        n = ctx_flat.shape[0]
        device = context.device

        tokens = torch.full((n, self.num_codebooks), MASK_TOKEN_ID, dtype=torch.long, device=device)
        unmasked = torch.zeros((n, self.num_codebooks), dtype=torch.bool, device=device)
        remaining = self.num_codebooks

        for step in range(num_passes):
            n_commit = remaining if step == num_passes - 1 else remaining // (num_passes - step)
            logits = self.forward(ctx_flat, tokens).float()  # [N, 12, 2016]

            if greedy or temperature <= 0.0:
                probs = F.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1)
                conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            else:
                scaled = logits / temperature
                probs = F.softmax(scaled, dim=-1)
                pred = torch.multinomial(probs.reshape(-1, self.vocab_size), 1).reshape(n, self.num_codebooks)
                log_p = torch.log(probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
                # Decaying Gumbel confidence noise (SoundStorm)
                noise_scale = 1.0 - step / max(num_passes, 1)
                u = torch.rand_like(log_p).clamp(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                conf = log_p + noise_scale * gumbel

            conf = conf.masked_fill(unmasked, float("-inf"))
            top_idx = conf.topk(n_commit, dim=-1).indices
            commit_mask = torch.zeros_like(unmasked).scatter_(1, top_idx, True)

            tokens = torch.where(commit_mask, pred, tokens)
            unmasked = unmasked | commit_mask
            remaining -= n_commit

        return tokens.reshape(*lead, self.num_codebooks)
