"""Full Kaza model wiring audio embeddings, expanded LFM3 backbone, and MaskGIT depth decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from kaza.backbone import MODEL_ID as BACKBONE_MODEL_ID, KazaBackbone
from kaza.depth_decoder import MaskGITDepthDecoder
from kaza.embeddings import BOS_TOKEN_ID, NUM_CODEBOOKS, KazaInputEmbedding, shift_agent_tokens


class KazaModel(nn.Module):
    """Audio-only speech-to-speech model (Phase 1: agent stream; Phase 2: + user stream)."""

    def __init__(
        self,
        backbone_id: str = BACKBONE_MODEL_ID,
        device: torch.device | str = "cuda",
        frozen_dtype: torch.dtype = torch.bfloat16,
        trainable_dtype: torch.dtype = torch.float32,
        autocast_dtype: torch.dtype = torch.bfloat16,
        use_user_stream: bool = False,
        drop_prob: float = 0.0,
        mtp_offsets: tuple[int, ...] = (),
        mtp_weight: float = 0.0,
        nope_new_layers: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.mtp_weight = mtp_weight
        self.last_losses: dict[str, float] = {}

        self.backbone = KazaBackbone(
            model_id=backbone_id,
            device=self.device,
            frozen_dtype=frozen_dtype,
            trainable_dtype=trainable_dtype,
            autocast_dtype=autocast_dtype,
            nope_new_layers=nope_new_layers,
        )
        d_model = self.backbone.hidden_size
        self.embeddings = KazaInputEmbedding(
            d_model=d_model,
            use_user_stream=use_user_stream,
            drop_prob=drop_prob,
        ).to(device=self.device, dtype=trainable_dtype)
        self.depth_decoder = MaskGITDepthDecoder(
            d_backbone=d_model,
            mtp_offsets=mtp_offsets,
        ).to(device=self.device, dtype=trainable_dtype)

    def set_warmup(self, enabled: bool) -> None:
        """During warmup, freeze the 6 new backbone layers so only embeddings + depth decoder train."""
        self.backbone.set_new_layers_trainable(not enabled)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only trainable parameters (embeddings + depth decoder + 6 new Global GQA layers)."""
        new_prefixes = tuple(f"backbone.model.layers.{i}." for i in self.backbone.new_layer_indices)
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if k.startswith(("embeddings.", "depth_decoder.")) or k.startswith(new_prefixes)
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys in trainable checkpoint: {unexpected[:5]}")

    def _autocast(self):
        use_amp = self.autocast_dtype != torch.float32 and self.device.type == "cuda"
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=use_amp,
        )

    def training_step(
        self,
        target_tokens: torch.Tensor,
        user_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass returning (total_loss, masked_accuracy)."""
        agent_in = shift_agent_tokens(target_tokens)
        with self._autocast():
            x = self.embeddings(agent_in, user_tokens)
            ctx = self.backbone(inputs_embeds=x, use_cache=False).last_hidden_state
            audio_loss, acc = self.depth_decoder.compute_loss(ctx, target_tokens, mask_all=False)
            if self.mtp_weight > 0.0 and self.depth_decoder.mtp_projs:
                mtp_loss = self.depth_decoder.compute_mtp_loss(ctx, target_tokens)
                total_loss = audio_loss + self.mtp_weight * mtp_loss
                self.last_losses = {"audio": audio_loss.item(), "mtp": mtp_loss.item()}
            else:
                total_loss = audio_loss
                self.last_losses = {"audio": audio_loss.item(), "mtp": 0.0}
            return total_loss, acc

    @torch.no_grad()
    def eval_all_masked(
        self,
        target_tokens: torch.Tensor,
        user_tokens: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Evaluate with all 12 slots masked (k=12), matching Inference Pass 1."""
        was_training = self.training
        self.eval()
        agent_in = shift_agent_tokens(target_tokens)
        with self._autocast():
            x = self.embeddings(agent_in, user_tokens)
            ctx = self.backbone(inputs_embeds=x, use_cache=False).last_hidden_state
            loss, acc = self.depth_decoder.compute_loss(ctx, target_tokens, mask_all=True)
        if was_training:
            self.train()
        return loss.item(), acc.item()

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        num_new_frames: int,
        num_passes: int = 3,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Free-running generation using the 30-layer KV/ShortConv cache.

        Args:
            prompt_tokens: [B, P, 12] ground-truth prefix (P >= 0).
            num_new_frames: number of 80 ms frames to generate autoregressively.
        Returns:
            [B, P + num_new_frames, 12] tokens (prompt followed by generated frames).
        """
        was_training = self.training
        self.eval()
        B, P, _ = prompt_tokens.shape

        with self._autocast():
            if P > 0:
                prompt_in = shift_agent_tokens(prompt_tokens)
                x = self.embeddings(prompt_in)
                pkv = self.backbone(inputs_embeds=x, use_cache=True).past_key_values
                last_tok = prompt_tokens[:, -1:, :]
            else:
                pkv = None
                last_tok = torch.full(
                    (B, 1, NUM_CODEBOOKS), BOS_TOKEN_ID, dtype=torch.long, device=self.device
                )

            generated = []
            for _ in range(num_new_frames):
                x = self.embeddings(last_tok)
                out = self.backbone(inputs_embeds=x, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                next_tok = self.depth_decoder.decode(
                    out.last_hidden_state,
                    num_passes=num_passes,
                    greedy=greedy,
                    temperature=temperature,
                )
                generated.append(next_tok)
                last_tok = next_tok

        if was_training:
            self.train()
        if not generated:
            return prompt_tokens
        return torch.cat([prompt_tokens, *generated], dim=1)
