"""Phase 1 training loop: live HF streaming, 2-group LR + no-decay, Drive checkpoints, monitoring."""

from __future__ import annotations

import inspect
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from kaza.codec import FRAME_RATE, SAMPLE_RATE, SAMPLES_PER_FRAME, KazaCodec
from kaza.data import TokenChunkDataset
from kaza.model import KazaModel


@dataclass
class TrainConfig:
    out_dir: str = "/content/drive/MyDrive/kaza_phase1/runs/waxal_nope_v1"
    frame_lengths: tuple[int, ...] = (250, 375, 750)  # 20s, 30s, 60s (exact divisors of 750)
    val_seq_len: int = 375  # 30.0s fixed validation windows
    batch_size: int = 4  # A100-40GB: 4 (safe up to 60s) | A100-80GB: 8 | L4: 2
    grad_accum: int = 8  # effective batch = 32 clips (~21 min audio / step)
    num_workers: int = 2  # CPU workers prefetching HF audio + soundfile + resample_poly
    max_steps: int = 5000
    warmup_freeze_steps: int = 200  # embeddings + depth decoder only before unfreezing 6 new layers
    lr_warmup_steps: int = 100
    backbone_warmup_steps: int = 100  # gentle 0 -> backbone_lr ramp after unfreezing
    lr: float = 3e-4  # embeddings + depth decoder + MTP
    min_lr: float = 3e-5
    backbone_lr: float = 1e-4  # gentler LR for 6 new NoPE Global GQA layers
    min_backbone_lr: float = 1e-5
    weight_decay: float = 0.01  # applied to 2D+ Linear weights only (norms, biases, embeddings excluded)
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 200
    sample_every: int = 200
    eval_batches: int = 16
    prompt_frames: int = 25  # 2.0s prompt for audio continuation samples
    gen_frames: int = 75  # 6.0s generated continuation
    sample_temp: float = 0.8


def _cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step <= 0:
        return 0.0
    if step <= warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _lrs_for_step(step: int, cfg: TrainConfig) -> tuple[float, float]:
    """Return (head_lr, backbone_lr) with a separate gentle ramp for the 6 new Global GQA layers."""
    head_lr = _cosine_lr(step, cfg.lr_warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
    if step <= cfg.warmup_freeze_steps:
        bb_lr = 0.0
    else:
        bb_step = step - cfg.warmup_freeze_steps
        bb_total = max(1, cfg.max_steps - cfg.warmup_freeze_steps)
        bb_lr = _cosine_lr(
            bb_step, cfg.backbone_warmup_steps, bb_total, cfg.backbone_lr, cfg.min_backbone_lr
        )
    return head_lr, bb_lr


def _collect_decay_split(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split a module's parameters into (decay, no_decay), excluding norms, biases, and embeddings."""
    no_decay_ids: set[int] = set()
    for sub in module.modules():
        if isinstance(sub, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)) or "norm" in type(sub).__name__.lower():
            for p in sub.parameters(recurse=False):
                no_decay_ids.add(id(p))

    decay, no_decay = [], []
    seen: set[int] = set()
    for name, p in module.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in no_decay_ids or p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(model: KazaModel, cfg: TrainConfig) -> tuple[torch.optim.AdamW, list[nn.Parameter]]:
    """Build 4-group AdamW: (head vs 6 new backbone layers) x (decay vs no_decay)."""
    model.set_warmup(False)

    head_decay, head_no_decay = [], []
    for mod in (model.embeddings, model.depth_decoder):
        d, nd = _collect_decay_split(mod)
        head_decay.extend(d)
        head_no_decay.extend(nd)

    bb_decay, bb_no_decay = [], []
    for idx in model.backbone.new_layer_indices:
        d, nd = _collect_decay_split(model.backbone.model.layers[idx])
        bb_decay.extend(d)
        bb_no_decay.extend(nd)

    groups = [
        {"name": "head_decay", "params": head_decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"name": "head_no_decay", "params": head_no_decay, "lr": cfg.lr, "weight_decay": 0.0},
        {"name": "bb_decay", "params": bb_decay, "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
        {"name": "bb_no_decay", "params": bb_no_decay, "lr": cfg.backbone_lr, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    all_trainable = head_decay + head_no_decay + bb_decay + bb_no_decay
    return optimizer, all_trainable


def _save_checkpoint(
    path: Path,
    model: KazaModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    cfg: TrainConfig,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "best_val_loss": best_val_loss,
            "config": asdict(cfg),
            "model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


@torch.no_grad()
def evaluate_val(
    model: KazaModel,
    val_loader: DataLoader,
    max_batches: int = 16,
) -> tuple[float, float]:
    """Held-out k=12 (all-masked) loss and accuracy."""
    losses, accs = [], []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch.to(model.device, non_blocking=True)
        loss, acc = model.eval_all_masked(batch)
        losses.append(loss)
        accs.append(acc)
    if not losses:
        return float("nan"), float("nan")
    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def sample_continuation(
    model: KazaModel,
    codec: KazaCodec,
    val_tokens: torch.Tensor,
    out_dir: Path,
    step: int,
    prompt_frames: int = 25,
    gen_frames: int = 75,
    temperature: float = 0.8,
    display_inline: bool = True,
) -> Path:
    """Generate a free-running continuation from a held-out validation prompt."""
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    ref_tokens = val_tokens[: prompt_frames + gen_frames].unsqueeze(0).to(model.device).long()
    prompt = ref_tokens[:, :prompt_frames]

    ref_path = samples_dir / "val_reference.wav"
    if not ref_path.exists():
        ref_wav = codec.decode(ref_tokens)[0].numpy()
        sf.write(ref_path, ref_wav, SAMPLE_RATE)

    gen_tokens = model.generate(
        prompt,
        num_new_frames=gen_frames,
        num_passes=3,
        greedy=False,
        temperature=temperature,
    )
    gen_wav = codec.decode(gen_tokens)[0].numpy()
    out_path = samples_dir / f"step_{step:06d}.wav"
    sf.write(out_path, gen_wav, SAMPLE_RATE)

    if display_inline:
        try:
            from IPython.display import Audio, display

            print(f"[Sample step {step}] 2s held-out prompt + 6s free-running continuation (temp={temperature}):")
            display(Audio(gen_wav, rate=SAMPLE_RATE))
        except Exception:
            pass
    return out_path


def preflight_check(
    model: KazaModel,
    codec: KazaCodec,
    train_data: IterableDataset,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Verify NoPE hook firing, 60s max-length micro-batch VRAM, and streaming throughput."""
    # 1. Verify NoPE hook signature & firing on all 6 new layers
    idx0 = model.backbone.new_layer_indices[0]
    sig = inspect.signature(model.backbone.model.layers[idx0].forward)
    has_pe = "position_embeddings" in sig.parameters
    print(f"Layer {idx0} forward has `position_embeddings`: {has_pe}")

    # 2. Measure 1 full max-length (e.g. 750-frame = 60s) micro-batch: stream + encode + fwd/bwd
    max_frames = max(cfg.frame_lengths)
    loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    wav_batch = next(iter(loader))[:, : max_frames * SAMPLES_PER_FRAME]
    t_stream = time.perf_counter() - t0

    t1 = time.perf_counter()
    tokens = codec.encode(wav_batch)[:, :max_frames]
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    t_codec = time.perf_counter() - t1

    model.backbone.nope_hook_calls = 0
    model.train()
    model.set_warmup(False)
    t2 = time.perf_counter()
    loss, _ = model.training_step(tokens)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    t_train = time.perf_counter() - t2

    hooks_fired = model.backbone.nope_hook_calls
    vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if model.device.type == "cuda" else 0.0
    audio_sec = cfg.batch_size * max_frames / FRAME_RATE
    audio_per_sec = audio_sec / max(t_stream + t_codec + t_train, 1e-6)

    print(
        f"Preflight ({cfg.batch_size} x {max_frames} frames = {audio_sec:.0f}s audio):\n"
        f"  NoPE hook calls: {hooks_fired} (expected {len(model.backbone.new_layer_indices) if model.backbone.nope_new_layers else 0})\n"
        f"  Timing: stream={t_stream:.2f}s | codec={t_codec*1000:.1f}ms | fwd+bwd={t_train:.2f}s "
        f"| {audio_per_sec:.1f} audio-sec/sec\n"
        f"  Peak VRAM (max length {max_frames} frames): {vram_gb:.2f} GB"
    )
    if model.backbone.nope_new_layers and hooks_fired != len(model.backbone.new_layer_indices):
        raise RuntimeError(
            f"NoPE hook fired {hooks_fired} times, expected {len(model.backbone.new_layer_indices)}!"
        )
    return {"vram_gb": vram_gb, "codec_ms": t_codec * 1000, "audio_per_sec": audio_per_sec}


def train_phase1(
    model: KazaModel,
    train_data: IterableDataset | torch.Tensor,
    val_tokens: torch.Tensor,
    cfg: TrainConfig,
    codec: KazaCodec | None = None,
    resume: bool = True,
) -> list[dict]:
    """Run Scaled Phase 1 training with exact frame-length sampling per step & Drive resume."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    latest_ckpt = out_dir / "latest.pt"
    best_ckpt = out_dir / "best.pt"

    is_streaming = isinstance(train_data, IterableDataset)
    if is_streaming and codec is None:
        raise ValueError("`codec` is required when `train_data` is a streaming waveform dataset.")

    optimizer, all_trainable = build_optimizer(model, cfg)

    start_step = 0
    best_val_loss = float("inf")
    if resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"Resumed from {latest_ckpt.name} at step {start_step} (best_val_k12_loss={best_val_loss:.4f})")

    if start_step >= cfg.max_steps:
        print(f"Already reached max_steps ({cfg.max_steps}).")
        return []

    # Reseed streaming dataset from start_step so resume never replays from clip 0
    if is_streaming and hasattr(train_data, "seed"):
        train_data.seed = cfg.seed + start_step * 31

    max_frames = max(cfg.frame_lengths)
    if is_streaming:
        loader_kwargs = (
            {"num_workers": cfg.num_workers, "prefetch_factor": 2, "persistent_workers": True}
            if cfg.num_workers > 0
            else {"num_workers": 0}
        )
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, drop_last=True, **loader_kwargs)
    else:
        train_ds = TokenChunkDataset(
            train_data,
            seq_len=max_frames,
            random_offset=True,
            steps_per_epoch=cfg.max_steps * cfg.batch_size * cfg.grad_accum,
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    val_ds = TokenChunkDataset(val_tokens, seq_len=cfg.val_seq_len, random_offset=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    train_iter = iter(train_loader)

    # Queue of sliced micro-batches so splitting 750-frame batches into 250/375 wastes zero audio
    pending: dict[int, deque[torch.Tensor]] = {fl: deque() for fl in cfg.frame_lengths}
    step_rng = random.Random(cfg.seed + start_step)

    def _next_microbatch(step_frames: int) -> torch.Tensor:
        nonlocal train_iter
        q = pending[step_frames]
        if not q:
            try:
                raw = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                raw = next(train_iter)
            unit = step_frames * SAMPLES_PER_FRAME if raw.ndim == 2 else step_frames
            n_splits = max(1, raw.shape[1] // unit)
            for s in range(n_splits):
                q.append(raw[:, s * unit : (s + 1) * unit])
        return q.popleft()

    model.train()
    model.set_warmup(start_step < cfg.warmup_freeze_steps)
    if start_step < cfg.warmup_freeze_steps:
        print(
            f"Warmup active: embeddings + depth decoder frozen-backbone until step {cfg.warmup_freeze_steps} "
            f"(then 6 new NoPE Global GQA layers ramp 0 -> {cfg.backbone_lr:.0e} over {cfg.backbone_warmup_steps} steps)"
        )

    history: list[dict] = []
    t_log = time.perf_counter()
    frames_since_log = 0

    for step in range(start_step + 1, cfg.max_steps + 1):
        if step == cfg.warmup_freeze_steps + 1:
            model.set_warmup(False)
            print(f"--- Step {step}: unfroze 6 new NoPE Global GQA layers (ramping 0 -> {cfg.backbone_lr:.0e}) ---")

        head_lr, bb_lr = _lrs_for_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = head_lr if group["name"].startswith("head_") else bb_lr

        # Pick ONE exact frame length for the entire optimizer step (equal grad_accum weights)
        step_frames = step_rng.choice(cfg.frame_lengths)

        optimizer.zero_grad(set_to_none=True)
        loss_audio_sum, loss_mtp_sum, acc_sum = 0.0, 0.0, 0.0

        for _ in range(cfg.grad_accum):
            mb = _next_microbatch(step_frames)
            if mb.ndim == 2:
                tokens = codec.encode(mb)[:, :step_frames]
            else:
                tokens = mb.to(model.device, non_blocking=True)

            loss, acc = model.training_step(tokens)
            (loss / cfg.grad_accum).backward()

            loss_audio_sum += model.last_losses.get("audio", loss.item()) / cfg.grad_accum
            loss_mtp_sum += model.last_losses.get("mtp", 0.0) / cfg.grad_accum
            acc_sum += acc.item() / cfg.grad_accum
            frames_since_log += tokens.shape[0] * tokens.shape[1]

        grad_norm = float(torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip).item())
        optimizer.step()

        record: dict = {
            "step": step,
            "frames": step_frames,
            "lr": head_lr,
            "bb_lr": bb_lr,
            "loss_audio": round(loss_audio_sum, 4),
            "loss_mtp": round(loss_mtp_sum, 4),
            "acc": round(acc_sum, 4),
            "grad_norm": round(grad_norm, 3),
        }

        do_eval = (step % cfg.eval_every == 0) or (step == cfg.max_steps)
        if do_eval:
            val_loss, val_acc = evaluate_val(model, val_loader, max_batches=cfg.eval_batches)
            record["val_k12_loss"] = round(val_loss, 4)
            record["val_k12_acc"] = round(val_acc, 4)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(best_ckpt, model, optimizer, step, best_val_loss, cfg)

        if step % cfg.log_every == 0 or step == start_step + 1 or do_eval:
            dt = max(time.perf_counter() - t_log, 1e-6)
            fps = frames_since_log / dt
            vram = torch.cuda.max_memory_allocated() / (1024**3) if model.device.type == "cuda" else 0.0
            record["frames_per_sec"] = round(fps, 1)
            record["audio_sec_per_sec"] = round(fps / FRAME_RATE, 1)
            record["vram_gb"] = round(vram, 2)

            val_str = (
                f" | val_k12={record['val_k12_loss']:.4f} ({record['val_k12_acc']*100:.1f}%)"
                if "val_k12_loss" in record
                else ""
            )
            print(
                f"step {step:4d}/{cfg.max_steps} ({step_frames}f) | audio={loss_audio_sum:.4f} "
                f"mtp={loss_mtp_sum:.4f} acc={acc_sum*100:5.1f}%{val_str} "
                f"| lr={head_lr:.1e}/{bb_lr:.1e} gnorm={grad_norm:.2f} "
                f"| {fps/FRAME_RATE:.1f}x RT | {vram:.1f} GB"
            )
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            history.append(record)
            t_log = time.perf_counter()
            frames_since_log = 0

        if step % cfg.save_every == 0 or step == cfg.max_steps:
            _save_checkpoint(latest_ckpt, model, optimizer, step, best_val_loss, cfg)

        if codec is not None and (step % cfg.sample_every == 0 or step == cfg.max_steps):
            sample_continuation(
                model,
                codec,
                val_tokens,
                out_dir=out_dir,
                step=step,
                prompt_frames=cfg.prompt_frames,
                gen_frames=cfg.gen_frames,
                temperature=cfg.sample_temp,
            )

    return history


def plot_metrics(out_dir: str | Path) -> None:
    """Plot train audio/MTP loss and held-out k=12 validation curves from metrics.jsonl."""
    import matplotlib.pyplot as plt

    path = Path(out_dir) / "metrics.jsonl"
    if not path.exists():
        print(f"No metrics file found at {path}")
        return

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [r["step"] for r in rows]
    val_rows = [r for r in rows if "val_k12_loss" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, [r["loss_audio"] for r in rows], label="Train audio loss")
    if any(r.get("loss_mtp", 0) > 0 for r in rows):
        ax1.plot(steps, [r["loss_mtp"] for r in rows], label="Train MTP (t+2, t+4)", alpha=0.7)
    if val_rows:
        ax1.plot(
            [r["step"] for r in val_rows],
            [r["val_k12_loss"] for r in val_rows],
            "o-",
            label="Val k=12 loss",
        )
    ax1.set_title("Loss")
    ax1.set_xlabel("Step")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, [r["acc"] * 100 for r in rows], label="Train masked acc (%)")
    if val_rows:
        ax2.plot(
            [r["step"] for r in val_rows],
            [r["val_k12_acc"] * 100 for r in val_rows],
            "o-",
            label="Val k=12 acc (%)",
        )
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
