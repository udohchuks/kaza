"""Phase 1 training: live WAXAL streaming, 4-group AdamW (head/backbone x decay/no-decay), Drive checkpoints."""

from __future__ import annotations

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
from torch.utils.data import DataLoader

from kaza.codec import FRAME_RATE, SAMPLE_RATE, SAMPLES_PER_FRAME, KazaCodec
from kaza.data import StreamBatcher, TokenChunkDataset, WaxalStreamingDataset
from kaza.model import KazaModel

ADAM_BETAS = (0.9, 0.95)
DECAY_GROUPS = ("head_decay", "bb_decay")

# Config fields that must match when resuming a run
RESUME_CRITICAL_FIELDS = (
    "frame_lengths",
    "batch_size",
    "grad_accum",
    "max_steps",
    "warmup_freeze_steps",
    "lr_warmup_steps",
    "backbone_warmup_steps",
    "lr",
    "min_lr",
    "backbone_lr",
    "min_backbone_lr",
    "weight_decay",
)


@dataclass
class TrainConfig:
    out_dir: str = "/content/drive/MyDrive/kaza_phase1/runs/waxal_nope_v1"
    frame_lengths: tuple[int, ...] = (250, 375, 750)  # 20s, 30s, 60s (each divides 750)
    val_seq_len: int = 375  # 30.0s fixed validation windows
    batch_size: int = 4  # A100-40GB: 4 (safe up to 60s) | A100-80GB: 8 | L4: 2
    grad_accum: int = 8  # effective batch = 32 clips
    prefetch_batches: int = 4  # streamed batches kept ready by the background thread
    max_steps: int = 5000
    warmup_freeze_steps: int = 200  # embeddings + depth decoder only before unfreezing 6 new layers
    lr_warmup_steps: int = 100
    backbone_warmup_steps: int = 100  # 0 -> backbone_lr ramp after unfreezing
    lr: float = 3e-4  # embeddings + depth decoder + MTP
    min_lr: float = 3e-5
    backbone_lr: float = 1e-4  # gentler LR for the 6 new NoPE Global GQA layers
    min_backbone_lr: float = 1e-5
    weight_decay: float = 0.01  # matrix weights only (norms, biases, embeddings excluded)
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


# ----- schedule & optimizer --------------------------------------------------------------------


def _cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step <= 0:
        return 0.0
    if step <= warmup:
        return max_lr * step / max(1, warmup)
    progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _lrs_for_step(step: int, cfg: TrainConfig) -> tuple[float, float]:
    """Return (head_lr, backbone_lr). The new layers stay at 0 until `warmup_freeze_steps`."""
    head_lr = _cosine_lr(step, cfg.lr_warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
    if step <= cfg.warmup_freeze_steps:
        return head_lr, 0.0
    bb_lr = _cosine_lr(
        step - cfg.warmup_freeze_steps,
        cfg.backbone_warmup_steps,
        max(1, cfg.max_steps - cfg.warmup_freeze_steps),
        cfg.backbone_lr,
        cfg.min_backbone_lr,
    )
    return head_lr, bb_lr


def _collect_decay_split(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split a module's parameters into (decay, no_decay): norms, biases and embeddings get no decay."""
    no_decay_ids: set[int] = set()
    for sub in module.modules():
        if isinstance(sub, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)) or "norm" in type(sub).__name__.lower():
            no_decay_ids.update(id(p) for p in sub.parameters(recurse=False))

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
    optimizer = torch.optim.AdamW(groups, betas=ADAM_BETAS)
    return optimizer, head_decay + head_no_decay + bb_decay + bb_no_decay


def _apply_config_hparams(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """Re-apply config hyperparameters after `optimizer.load_state_dict` restored the saved ones."""
    for group in optimizer.param_groups:
        group["weight_decay"] = cfg.weight_decay if group["name"] in DECAY_GROUPS else 0.0
        group["betas"] = ADAM_BETAS


# ----- checkpoints -----------------------------------------------------------------------------


def _model_settings(model: KazaModel) -> dict:
    return {
        "drop_prob": model.embeddings.drop_prob,
        "mtp_offsets": list(model.depth_decoder.mtp_offsets),
        "mtp_weight": model.mtp_weight,
        "nope_new_layers": model.backbone.nope_new_layers,
    }


def _save_checkpoint(
    path: Path,
    model: KazaModel,
    step: int,
    best_val_loss: float,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Atomically save trainable weights (and optimizer state when given) to `path`."""
    payload = {
        "step": step,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
        "model_settings": _model_settings(model),
        "model": model.trainable_state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _check_resume_compat(ckpt: dict, model: KazaModel, cfg: TrainConfig) -> None:
    """Raise if the checkpoint's model settings or schedule differ from the current run."""
    problems = []
    saved_model = ckpt.get("model_settings")
    if saved_model is not None and saved_model != _model_settings(model):
        problems.append(f"model settings: checkpoint={saved_model} current={_model_settings(model)}")
    saved_cfg = ckpt.get("config", {})
    current_cfg = asdict(cfg)
    for key in RESUME_CRITICAL_FIELDS:
        if key not in saved_cfg:
            continue
        saved = tuple(saved_cfg[key]) if isinstance(saved_cfg[key], list) else saved_cfg[key]
        if saved != current_cfg[key]:
            problems.append(f"{key}: checkpoint={saved} current={current_cfg[key]}")
    if problems:
        raise ValueError(
            "Checkpoint does not match the current run. Use a new RUN_NAME or restore these settings:\n  "
            + "\n  ".join(problems)
        )


# ----- evaluation & samples --------------------------------------------------------------------


@torch.no_grad()
def evaluate_val(model: KazaModel, val_loader: DataLoader, max_batches: int = 16) -> tuple[float, float]:
    """Held-out k=12 (all-masked) loss and accuracy."""
    losses, accs = [], []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        loss, acc = model.eval_all_masked(batch.to(model.device, non_blocking=True))
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
    name: str,
    prompt_frames: int = 25,
    gen_frames: int = 75,
    temperature: float = 0.8,
) -> Path:
    """Continue a held-out prompt free-running; save `samples/{name}.wav` and play it in notebooks."""
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    ref_tokens = val_tokens[: prompt_frames + gen_frames].unsqueeze(0).to(model.device).long()
    ref_path = samples_dir / f"val_reference_{prompt_frames}+{gen_frames}f.wav"
    if not ref_path.exists():
        sf.write(ref_path, codec.decode(ref_tokens)[0].numpy(), SAMPLE_RATE)

    gen_tokens = model.generate(
        ref_tokens[:, :prompt_frames],
        num_new_frames=gen_frames,
        num_passes=3,
        greedy=False,
        temperature=temperature,
    )
    gen_wav = codec.decode(gen_tokens)[0].numpy()
    out_path = samples_dir / f"{name}.wav"
    sf.write(out_path, gen_wav, SAMPLE_RATE)

    try:
        from IPython.display import Audio, display
    except ImportError:
        return out_path
    print(
        f"[Sample {name}] {prompt_frames / FRAME_RATE:.0f}s held-out prompt + "
        f"{gen_frames / FRAME_RATE:.0f}s free-running continuation (temp={temperature}):"
    )
    display(Audio(gen_wav, rate=SAMPLE_RATE))
    return out_path


# ----- streaming helpers -----------------------------------------------------------------------


def _encode_exact(codec: KazaCodec, wav: torch.Tensor, frames: int) -> torch.Tensor:
    """Encode `[B, frames * 1920]` audio and return exactly `frames` token frames."""
    tokens = codec.encode(wav)
    if tokens.shape[1] < frames:
        raise RuntimeError(f"Codec returned {tokens.shape[1]} frames for {frames}-frame audio.")
    return tokens[:, :frames]


class _MicroBatchSlicer:
    """Slices streamed max-length waveform batches into shorter micro-batches with no wasted audio."""

    def __init__(self, batches: StreamBatcher, frame_lengths: tuple[int, ...]):
        self.batches = batches
        self.pending: dict[int, deque[torch.Tensor]] = {fl: deque() for fl in frame_lengths}

    def next(self, frames: int) -> torch.Tensor:
        q = self.pending[frames]
        if not q:
            try:
                raw = next(self.batches)
            except StopIteration:
                raise RuntimeError("Training stream ended: every source was dropped (see [ALERT] lines).") from None
            unit = frames * SAMPLES_PER_FRAME
            q.extend(raw[:, s * unit : (s + 1) * unit] for s in range(raw.shape[1] // unit))
        return q.popleft()


# ----- training --------------------------------------------------------------------------------


def _open_tb_writer(log_dir: Path):
    """Open TensorBoard SummaryWriter if available (preinstalled on Colab)."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        return None


def _log_tb(writer, record: dict, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("loss/audio", record["loss_audio"], step)
    if record.get("loss_mtp", 0.0) > 0.0:
        writer.add_scalar("loss/mtp", record["loss_mtp"], step)
    writer.add_scalar("acc/train_masked", record["acc"] * 100.0, step)
    writer.add_scalar("lr/head", record["lr"], step)
    writer.add_scalar("lr/backbone", record["bb_lr"], step)
    writer.add_scalar("health/grad_norm", record["grad_norm"], step)
    writer.add_scalar("perf/audio_x_rt", record["audio_sec_per_sec"], step)
    writer.add_scalar("perf/vram_gb", record["vram_gb"], step)
    if "val_k12_loss" in record:
        writer.add_scalar("loss/val_k12", record["val_k12_loss"], step)
        writer.add_scalar("acc/val_k12", record["val_k12_acc"] * 100.0, step)
        writer.add_scalar("health/active_sources", record["active_sources"], step)
    writer.flush()


def train_phase1(
    model: KazaModel,
    train_data: WaxalStreamingDataset,
    val_tokens: torch.Tensor,
    cfg: TrainConfig,
    codec: KazaCodec,
    resume: bool = True,
) -> list[dict]:
    """Run Phase 1 training with one frame length per optimizer step, Drive checkpoints and resume."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    latest_ckpt = out_dir / "latest.pt"
    best_ckpt = out_dir / "best.pt"
    on_cuda = model.device.type == "cuda"

    optimizer, all_trainable = build_optimizer(model, cfg)

    start_step = 0
    best_val_loss = float("inf")
    if resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        _check_resume_compat(ckpt, model, cfg)
        model.load_trainable_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        _apply_config_hparams(optimizer, cfg)
        start_step = int(ckpt["step"])
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"Resumed from {latest_ckpt.name} at step {start_step} (best_val_k12_loss={best_val_loss:.4f})")

    if start_step >= cfg.max_steps:
        print(f"Already reached max_steps ({cfg.max_steps}).")
        return []

    # New randomness per resume point: stream order, masks and slot drops
    torch.manual_seed(cfg.seed + start_step)
    train_data.seed = cfg.seed + start_step * 31
    step_rng = random.Random(cfg.seed + start_step)

    val_loader = DataLoader(
        TokenChunkDataset(val_tokens, seq_len=cfg.val_seq_len), batch_size=cfg.batch_size, shuffle=False
    )
    batches = StreamBatcher(train_data, cfg.batch_size, prefetch=cfg.prefetch_batches)
    slicer = _MicroBatchSlicer(batches, cfg.frame_lengths)

    model.train()
    model.set_warmup(start_step < cfg.warmup_freeze_steps)
    if start_step < cfg.warmup_freeze_steps:
        print(
            f"Warmup: only embeddings + depth decoder train until step {cfg.warmup_freeze_steps}, "
            f"then the 6 new NoPE layers ramp 0 -> {cfg.backbone_lr:.0e} over {cfg.backbone_warmup_steps} steps."
        )

    history: list[dict] = []
    tb_writer = _open_tb_writer(out_dir / "tb")
    t_log = time.perf_counter()
    frames_since_log = 0
    completed_step = start_step
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    try:
        for step in range(start_step + 1, cfg.max_steps + 1):
            if step == cfg.warmup_freeze_steps + 1:
                model.set_warmup(False)
                print(f"--- Step {step}: unfroze 6 new NoPE Global GQA layers ---")

            head_lr, bb_lr = _lrs_for_step(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = head_lr if group["name"].startswith("head_") else bb_lr

            step_frames = step_rng.choice(cfg.frame_lengths)  # one length per optimizer step
            if step == start_step + 1:
                model.backbone.nope_hook_calls = 0
            optimizer.zero_grad(set_to_none=True)
            loss_audio_sum, loss_mtp_sum, acc_sum = 0.0, 0.0, 0.0

            for _ in range(cfg.grad_accum):
                tokens = _encode_exact(codec, slicer.next(step_frames), step_frames)
                loss, acc = model.training_step(tokens)
                (loss / cfg.grad_accum).backward()

                loss_audio_sum += model.last_losses["audio"] / cfg.grad_accum
                loss_mtp_sum += model.last_losses["mtp"] / cfg.grad_accum
                acc_sum += acc.item() / cfg.grad_accum
                frames_since_log += tokens.shape[0] * tokens.shape[1]
                del tokens, loss, acc

            if step == start_step + 1 and model.backbone.nope_new_layers:
                expected_hooks = len(model.backbone.new_layer_indices) * cfg.grad_accum
                if model.backbone.nope_hook_calls != expected_hooks:
                    raise RuntimeError(
                        f"NoPE hook fired {model.backbone.nope_hook_calls} times on step {step}, "
                        f"expected {expected_hooks}."
                    )

            grad_norm = float(torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip).item())
            optimizer.step()
            completed_step = step

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

            do_eval = step % cfg.eval_every == 0 or step == cfg.max_steps
            if do_eval:
                val_loss, val_acc = evaluate_val(model, val_loader, max_batches=cfg.eval_batches)
                record["val_k12_loss"] = round(val_loss, 4)
                record["val_k12_acc"] = round(val_acc, 4)
                record["active_sources"] = train_data.active_sources
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(best_ckpt, model, step, best_val_loss, cfg)  # weights only

            if step % cfg.log_every == 0 or step == start_step + 1 or do_eval:
                dt = max(time.perf_counter() - t_log, 1e-6)
                rt = frames_since_log / FRAME_RATE / dt
                vram = torch.cuda.max_memory_allocated() / (1024**3) if on_cuda else 0.0
                record["audio_sec_per_sec"] = round(rt, 1)
                record["vram_gb"] = round(vram, 2)

                val_str = (
                    f" | val_k12={record['val_k12_loss']:.4f} ({record['val_k12_acc'] * 100:.1f}%)"
                    f" | sources={record['active_sources']}"
                    if do_eval
                    else ""
                )
                print(
                    f"step {step:4d}/{cfg.max_steps} ({step_frames}f) | audio={loss_audio_sum:.4f} "
                    f"mtp={loss_mtp_sum:.4f} acc={acc_sum * 100:5.1f}%{val_str} "
                    f"| lr={head_lr:.1e}/{bb_lr:.1e} gnorm={grad_norm:.2f} | {rt:.1f}x RT | {vram:.1f} GB"
                )
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                _log_tb(tb_writer, record, step)
                history.append(record)
                t_log = time.perf_counter()
                frames_since_log = 0
                if on_cuda:
                    torch.cuda.reset_peak_memory_stats()

            if step % cfg.save_every == 0 or step == cfg.max_steps:
                _save_checkpoint(latest_ckpt, model, step, best_val_loss, cfg, optimizer=optimizer)

            if step % cfg.sample_every == 0 or step == cfg.max_steps:
                sample_continuation(
                    model,
                    codec,
                    val_tokens,
                    out_dir=out_dir,
                    name=f"step_{step:06d}",
                    prompt_frames=cfg.prompt_frames,
                    gen_frames=cfg.gen_frames,
                    temperature=cfg.sample_temp,
                )
    except BaseException:
        # On a crash or manual interrupt, keep the last completed step unless it was just saved
        if completed_step > start_step and completed_step % cfg.save_every != 0:
            try:
                _save_checkpoint(latest_ckpt, model, completed_step, best_val_loss, cfg, optimizer=optimizer)
                print(f"Saved {latest_ckpt.name} at step {completed_step} before stopping.")
            except Exception as save_err:
                print(f"[WARN] Could not save {latest_ckpt.name}: {save_err}")
        raise
    finally:
        batches.close()
        if tb_writer is not None:
            tb_writer.close()

    return history


def plot_metrics(out_dir: str | Path) -> None:
    """Plot train audio/MTP loss and held-out k=12 validation curves from metrics.jsonl."""
    import matplotlib.pyplot as plt

    path = Path(out_dir) / "metrics.jsonl"
    if not path.exists():
        print(f"No metrics file found at {path}")
        return

    # Keep the last record per step: a resume can log the same step twice
    by_step: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[r["step"]] = r
    rows = [by_step[s] for s in sorted(by_step)]
    steps = [r["step"] for r in rows]
    val_rows = [r for r in rows if "val_k12_loss" in r]
    val_steps = [r["step"] for r in val_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, [r["loss_audio"] for r in rows], label="Train audio loss")
    if any(r.get("loss_mtp", 0) > 0 for r in rows):
        ax1.plot(steps, [r["loss_mtp"] for r in rows], label="Train MTP (t+2, t+4)", alpha=0.7)
    if val_rows:
        ax1.plot(val_steps, [r["val_k12_loss"] for r in val_rows], "o-", label="Val k=12 loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Step")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, [r["acc"] * 100 for r in rows], label="Train masked acc (%)")
    if val_rows:
        ax2.plot(val_steps, [r["val_k12_acc"] * 100 for r in val_rows], "o-", label="Val k=12 acc (%)")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
