"""Phase 1 training loop, Drive checkpointing, held-out k=12 eval, and monitoring."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader

from kaza.codec import SAMPLE_RATE, KazaCodec
from kaza.data import TokenChunkDataset
from kaza.model import KazaModel


@dataclass
class TrainConfig:
    out_dir: str = "/content/drive/MyDrive/kaza_runs/twi_phase1"
    seq_len: int = 375  # 30.0 s at 12.5 Hz
    batch_size: int = 2
    grad_accum: int = 8  # effective batch = 16 clips (8 min audio / step)
    max_steps: int = 2000
    warmup_freeze_steps: int = 200  # embeddings + depth decoder only before unfreezing 6 new layers
    lr_warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 200
    sample_every: int = 200
    eval_batches: int = 16
    prompt_frames: int = 25  # 2.0 s prompt for audio continuation samples
    gen_frames: int = 75  # 6.0 s generated continuation
    sample_temp: float = 0.8


def _lr_for_step(step: int, cfg: TrainConfig) -> float:
    if step <= cfg.lr_warmup_steps:
        return cfg.lr * step / max(1, cfg.lr_warmup_steps)
    progress = (step - cfg.lr_warmup_steps) / max(1, cfg.max_steps - cfg.lr_warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


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
    """Generate a free-running Twi continuation from a held-out validation prompt."""
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


def train_phase1(
    model: KazaModel,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    cfg: TrainConfig,
    codec: KazaCodec | None = None,
    resume: bool = True,
) -> list[dict]:
    """Run Scaled Phase 1 training with checkpointing, validation, and audio sampling."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    latest_ckpt = out_dir / "latest.pt"
    best_ckpt = out_dir / "best.pt"

    train_ds = TokenChunkDataset(
        train_tokens,
        seq_len=cfg.seq_len,
        random_offset=True,
        steps_per_epoch=cfg.max_steps * cfg.batch_size * cfg.grad_accum,
    )
    val_ds = TokenChunkDataset(val_tokens, seq_len=cfg.seq_len, random_offset=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    train_iter = iter(train_loader)

    # Ensure all trainable params (including 6 new layers) are registered in AdamW
    model.set_warmup(False)
    all_trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

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

    model.train()
    model.set_warmup(start_step < cfg.warmup_freeze_steps)
    if start_step < cfg.warmup_freeze_steps:
        print(f"Warmup active: training embeddings + depth decoder only until step {cfg.warmup_freeze_steps}")

    history: list[dict] = []
    t_log = time.perf_counter()
    frames_since_log = 0

    for step in range(start_step + 1, cfg.max_steps + 1):
        if step == cfg.warmup_freeze_steps + 1:
            model.set_warmup(False)
            print(f"--- Step {step}: unfroze 6 new Global GQA layers ---")

        lr = _lr_for_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_audio_sum, loss_mtp_sum, acc_sum = 0.0, 0.0, 0.0

        for _ in range(cfg.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(model.device, non_blocking=True)
            loss, acc = model.training_step(batch)
            (loss / cfg.grad_accum).backward()

            loss_audio_sum += model.last_losses.get("audio", loss.item()) / cfg.grad_accum
            loss_mtp_sum += model.last_losses.get("mtp", 0.0) / cfg.grad_accum
            acc_sum += acc.item() / cfg.grad_accum
            frames_since_log += batch.shape[0] * batch.shape[1]

        grad_norm = float(torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip).item())
        optimizer.step()

        record: dict = {
            "step": step,
            "lr": lr,
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
            record["vram_gb"] = round(vram, 2)

            val_str = (
                f" | val_k12={record['val_k12_loss']:.4f} ({record['val_k12_acc']*100:.1f}%)"
                if "val_k12_loss" in record
                else ""
            )
            print(
                f"step {step:4d}/{cfg.max_steps} | audio={loss_audio_sum:.4f} mtp={loss_mtp_sum:.4f} "
                f"acc={acc_sum*100:5.1f}%{val_str} | lr={lr:.1e} gnorm={grad_norm:.2f} "
                f"| {fps:.0f} fr/s | {vram:.1f} GB"
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
