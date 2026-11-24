"""Frozen LiquidCodec wrapper and audio reconstruction metrics."""

from __future__ import annotations

from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoModel

MODEL_ID = "LiquidAI/Liquid-Codec"
SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = 1920


def to_mono_24k(wav: np.ndarray | torch.Tensor, sr: int, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Convert audio array to 1D float32 mono tensor at target_sr."""
    x = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    x = x.astype(np.float32, copy=False)
    if x.ndim == 2:
        # Handle both [C, T] and [T, C]
        x = x.mean(axis=0 if x.shape[0] <= x.shape[1] else 1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(x))


class KazaCodec:
    """Thin frozen wrapper around LiquidAI/Liquid-Codec.

    Uses [B, T, 12] token layout (time first, 12 parallel GFSQ groups per 80 ms frame).
    """

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = MODEL_ID,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, wav_24k: torch.Tensor) -> torch.Tensor:
        """Encode 24 kHz mono audio [T_samples] or [B, T_samples] -> tokens [B, T, 12] (long)."""
        x = wav_24k.unsqueeze(0) if wav_24k.ndim == 1 else wav_24k
        x = x.to(device=self.device, dtype=self.dtype).contiguous()
        codes = self.model.encode(x).audio_codes  # [B, 12, T]
        return codes.transpose(1, 2).contiguous().long()  # [B, T, 12]

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens [T, 12] or [B, T, 12] -> 24 kHz mono audio [B, T_samples] (float32)."""
        t = tokens.unsqueeze(0) if tokens.ndim == 2 else tokens
        codes = t.transpose(1, 2).contiguous().to(device=self.device)  # [B, 12, T]
        wav = self.model.decode(codes).audio_values
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        return wav.float().cpu()


def codec_metrics(ref_24k: torch.Tensor, rec_24k: torch.Tensor) -> dict[str, float]:
    """Compute PESQ (WB, 16 kHz) and STOI (16 kHz) between reference and reconstructed 24 kHz audio."""
    from pesq import pesq
    from pystoi import stoi

    ref = ref_24k.detach().cpu().numpy().reshape(-1).astype(np.float32)
    rec = rec_24k.detach().cpu().numpy().reshape(-1).astype(np.float32)
    n = min(len(ref), len(rec))
    ref, rec = ref[:n], rec[:n]

    ref_16k = resample_poly(ref, 2, 3).astype(np.float32)
    rec_16k = resample_poly(rec, 2, 3).astype(np.float32)

    return {
        "duration_s": round(n / SAMPLE_RATE, 2),
        "pesq_wb": round(float(pesq(16_000, ref_16k, rec_16k, "wb")), 3),
        "stoi": round(float(stoi(ref_16k, rec_16k, 16_000, extended=False)), 3),
    }
