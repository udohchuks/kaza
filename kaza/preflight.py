"""Standalone preflight verification for NoPE hooks, streaming speed, and peak 60s VRAM."""

from __future__ import annotations

import time

import torch

from kaza.codec import FRAME_RATE, SAMPLES_PER_FRAME, KazaCodec
from kaza.data import StreamBatcher, WaxalStreamingDataset
from kaza.model import KazaModel
from kaza.train import TrainConfig, _encode_exact, build_optimizer


def preflight_check(
    model: KazaModel,
    codec: KazaCodec,
    train_data: WaxalStreamingDataset,
    cfg: TrainConfig,
    timed_batches: int = 5,
) -> dict[str, float]:
    """Check NoPE hooks, steady-state streaming speed, and peak VRAM of one full max-length step.

    The VRAM figure includes AdamW state: one optimizer step runs with every LR and weight decay
    set to 0, so no weight changes.
    """
    max_frames = max(cfg.frame_lengths)
    batches = StreamBatcher(train_data, cfg.batch_size, prefetch=cfg.prefetch_batches)
    try:
        t0 = time.perf_counter()
        wav = next(batches)
        t_first = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(timed_batches):
            next(batches)
        t_stream = (time.perf_counter() - t1) / timed_batches
    finally:
        batches.close()

    on_cuda = model.device.type == "cuda"
    if on_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t2 = time.perf_counter()
    tokens = _encode_exact(codec, wav[:, : max_frames * SAMPLES_PER_FRAME], max_frames)
    if on_cuda:
        torch.cuda.synchronize()
    t_codec = time.perf_counter() - t2

    optimizer, _ = build_optimizer(model, cfg)
    for group in optimizer.param_groups:
        group["lr"], group["weight_decay"] = 0.0, 0.0

    model.backbone.nope_hook_calls = 0
    model.train()
    t3 = time.perf_counter()
    loss, _ = model.training_step(tokens)
    loss.backward()
    optimizer.step()
    if on_cuda:
        torch.cuda.synchronize()
    t_train = time.perf_counter() - t3
    hooks_fired = model.backbone.nope_hook_calls

    vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if on_cuda else 0.0
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    if on_cuda:
        torch.cuda.empty_cache()

    batch_audio_sec = cfg.batch_size * max_frames / FRAME_RATE
    stream_rt = batch_audio_sec / max(t_stream, 1e-6)
    gpu_rt = batch_audio_sec / max(t_codec + t_train, 1e-6)
    expected_hooks = len(model.backbone.new_layer_indices) if model.backbone.nope_new_layers else 0

    print(
        f"Preflight ({cfg.batch_size} x {max_frames} frames = {batch_audio_sec:.0f}s audio):\n"
        f"  NoPE hook calls: {hooks_fired} (expected {expected_hooks})\n"
        f"  Stream: first batch {t_first:.1f}s | steady {t_stream:.2f}s/batch = {stream_rt:.0f}x real time\n"
        f"  GPU: codec {t_codec * 1000:.0f}ms + fwd/bwd/step {t_train:.2f}s = {gpu_rt:.0f}x real time\n"
        f"  Peak VRAM incl. AdamW state: {vram_gb:.2f} GB"
    )
    if hooks_fired != expected_hooks:
        raise RuntimeError(f"NoPE hook fired {hooks_fired} times, expected {expected_hooks}.")
    if stream_rt < gpu_rt:
        print("  [WARN] Streaming is slower than the GPU step; training will wait on data.")
    return {"vram_gb": vram_gb, "codec_ms": t_codec * 1000, "stream_rt": stream_rt, "gpu_rt": gpu_rt}
