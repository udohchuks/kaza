"""Expanded LFM3-1.7B-Base backbone with 6 zero-init trainable Global GQA layers."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_ID = "LiquidAI/LFM3-1.7B-Base"


def _zero_linear(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _zero_init_decoder_layer(layer: nn.Module) -> None:
    """Zero-init attention output and FFN down projections so the layer is an exact identity at step 0."""
    # Attention output projection (Lfm3Attention uses o_proj)
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise ValueError("Expected a Global GQA layer with `self_attn`.")
    out_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
    if out_proj is None:
        raise ValueError("Could not find attention output projection (`o_proj` / `out_proj`).")
    _zero_linear(out_proj)

    # FFN down projection (Lfm3MLP uses feed_forward.w2)
    ffn = getattr(layer, "feed_forward", None) or getattr(layer, "mlp", None)
    if ffn is None:
        raise ValueError("Could not find FFN module (`feed_forward` / `mlp`).")
    down_proj = getattr(ffn, "w2", None) or getattr(ffn, "down_proj", None)
    if down_proj is None:
        raise ValueError("Could not find FFN down projection (`w2` / `down_proj`).")
    _zero_linear(down_proj)


def _set_layer_idx(layer: nn.Module, idx: int) -> None:
    if hasattr(layer, "layer_idx"):
        layer.layer_idx = idx
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
        layer.self_attn.layer_idx = idx
    if hasattr(layer, "conv") and hasattr(layer.conv, "layer_idx"):
        layer.conv.layer_idx = idx


class KazaBackbone(nn.Module):
    """LFM3-1.7B-Base (24 frozen layers) + 6 trainable Global GQA copies (30 layers total)."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: torch.device | str = "cuda",
        frozen_dtype: torch.dtype = torch.bfloat16,
        trainable_dtype: torch.dtype = torch.float32,
        autocast_dtype: torch.dtype = torch.bfloat16,
        nope_new_layers: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.frozen_dtype = frozen_dtype
        self.autocast_dtype = autocast_dtype
        self.nope_new_layers = nope_new_layers
        self.nope_hook_calls = 0

        # Load base Lfm3Model directly (skips unused text lm_head)
        base = AutoModel.from_pretrained(model_id, dtype=frozen_dtype)
        # Drop unused text token embeddings
        if hasattr(base, "embed_tokens"):
            base.embed_tokens = None

        # Freeze all original parameters
        base.requires_grad_(False)

        orig_types = list(base.config.layer_types)
        new_layers: list[nn.Module] = []
        new_types: list[str] = []
        self.new_layer_indices: list[int] = []

        for orig_idx, (layer, ltype) in enumerate(zip(base.layers, orig_types)):
            new_layers.append(layer)
            new_types.append(ltype)

            # Insert a zero-init trainable copy right after each original Global GQA layer
            if ltype == "full_attention":
                cloned = copy.deepcopy(layer).to(dtype=trainable_dtype)
                _zero_init_decoder_layer(cloned)
                cloned.requires_grad_(True)
                self.new_layer_indices.append(len(new_layers))
                new_layers.append(cloned)
                new_types.append("full_attention")

        # Re-index all 30 layers so KV and ShortConv caches stay aligned
        for idx, layer in enumerate(new_layers):
            _set_layer_idx(layer, idx)

        base.layers = nn.ModuleList(new_layers)
        base.config.num_hidden_layers = len(new_layers)
        base.config.layer_types = new_types

        self.model = base.to(self.device)
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size

        if self.nope_new_layers:
            self._attach_nope_hooks()

    def _attach_nope_hooks(self) -> None:
        """Replace position_embeddings=(cos, sin) with (1, 0) on the 6 new Global GQA layers (NoPE)."""
        def _nope_pre_hook(_module, args, kwargs):
            pe = kwargs.get("position_embeddings")
            if pe is not None and isinstance(pe, tuple) and len(pe) == 2:
                cos, sin = pe
                kwargs = dict(kwargs)
                kwargs["position_embeddings"] = (torch.ones_like(cos), torch.zeros_like(sin))
                self.nope_hook_calls += 1
            return args, kwargs

        for idx in self.new_layer_indices:
            self.model.layers[idx].register_forward_pre_hook(_nope_pre_hook, with_kwargs=True)

    def set_new_layers_trainable(self, trainable: bool) -> None:
        """Toggle gradients on the 6 new Global GQA layers (off during brief warmup)."""
        for idx in self.new_layer_indices:
            self.model.layers[idx].requires_grad_(trainable)

    def summary(self) -> str:
        """Return a readable 30-layer table showing type, trainable status, dtype, and params."""
        new_set = set(self.new_layer_indices)
        lines = [
            f"KazaBackbone ({self.config.num_hidden_layers} layers, hidden={self.hidden_size})",
            f"{'Idx':>3}  {'Type':<18}  {'Status':<10}  {'Dtype':<14}  {'Params':>8}",
            "-" * 61,
        ]
        for idx, (layer, ltype) in enumerate(zip(self.model.layers, self.config.layer_types)):
            p0 = next(layer.parameters())
            n_m = sum(p.numel() for p in layer.parameters()) / 1e6
            tag = "NEW (train)" if idx in new_set else "frozen"
            lines.append(f"{idx:>3}  {ltype:<18}  {tag:<10}  {str(p0.dtype):<14}  {n_m:>6.1f}M")
        return "\n".join(lines)

    def frozen_checksum(self) -> float:
        """Fast scalar checksum of all original frozen parameters (independent of warmup state)."""
        new_param_ids = {
            id(p)
            for idx in self.new_layer_indices
            for p in self.model.layers[idx].parameters()
        }
        total = 0.0
        with torch.no_grad():
            for p in self.model.parameters():
                if id(p) not in new_param_ids:
                    total += p.float().abs().sum().item()
        return total

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        use_amp = self.autocast_dtype != torch.float32 and inputs_embeds.device.type == "cuda"
        with torch.amp.autocast(
            device_type=inputs_embeds.device.type,
            dtype=self.autocast_dtype,
            enabled=use_amp,
        ):
            return self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )


@torch.no_grad()
def verify_step0_identity(
    backbone: KazaBackbone,
    model_id: str = MODEL_ID,
    seq_len: int = 8,
) -> float:
    """Check #1: compare expanded 30-layer backbone output against original 24-layer LFM3 at step 0.

    The new layers output zero at step 0, so this check cannot detect a broken NoPE hook;
    `train_phase1` asserts `nope_hook_calls` on the first training step.
    """
    orig = AutoModel.from_pretrained(model_id, dtype=backbone.frozen_dtype).to(backbone.device).eval()
    was_training = backbone.training
    backbone.eval()

    x = torch.randn(1, seq_len, backbone.hidden_size, device=backbone.device, dtype=backbone.frozen_dtype)
    use_amp = backbone.autocast_dtype != torch.float32 and backbone.device.type == "cuda"
    with torch.amp.autocast(device_type=backbone.device.type, dtype=backbone.autocast_dtype, enabled=use_amp):
        y_orig = orig(inputs_embeds=x, use_cache=False).last_hidden_state.float()
    y_exp = backbone(inputs_embeds=x, use_cache=False).last_hidden_state.float()
    max_diff = (y_orig - y_exp).abs().max().item()

    del orig
    backbone.train(was_training)
    if backbone.device.type == "cuda":
        torch.cuda.empty_cache()
    return max_diff
