"""Dataset acquisition, LiquidCodec pre-tokenization, and chunked token Dataset."""

from __future__ import annotations

import io
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from kaza.codec import FRAME_RATE, SAMPLE_RATE, KazaCodec, to_mono_24k

TWI_TRAIN_SOURCES: list[tuple[str, str, str]] = [
    ("google/WaxalNLP", "aka_asr", "train"),
    ("google/WaxalNLP", "aka_asr", "unlabeled"),
]
TWI_VAL_SOURCES: list[tuple[str, str, str]] = [
    ("google/WaxalNLP", "aka_asr", "validation"),
]
IGBO_TRAIN_SOURCES: list[tuple[str, str, str]] = [
    ("google/WaxalNLP", "ibo_tts", "train"),
]


def _decode_row_audio(row: dict) -> torch.Tensor | None:
    audio = row.get("audio")
    if not audio:
        return None
    if audio.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(audio["bytes"]))
    elif audio.get("path") is not None:
        wav, sr = sf.read(audio["path"])
    elif audio.get("array") is not None:
        wav, sr = audio["array"], audio["sampling_rate"]
    else:
        return None
    if len(wav) < int(0.4 * sr):  # skip ultra-short <0.4s artifacts
        return None
    return to_mono_24k(wav, int(sr))


def build_token_cache(
    sources: list[tuple[str, str, str]],
    codec: KazaCodec,
    out_path: str | Path,
    max_hours: float = 10.0,
    encode_chunk_sec: float = 30.0,
    gap_sec: float = 0.3,
    overwrite: bool = False,
) -> dict:
    """Stream HF audio datasets, encode with frozen LiquidCodec, and cache as uint16 tokens.

    1 hour of 12.5 Hz x 12 tokens in uint16 is only ~1.08 MB on disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        data = torch.load(out_path, map_location="cpu", weights_only=True)
        print(
            f"Loaded cached tokens: {out_path.name} | "
            f"{data['hours']:.2f}h ({data['tokens'].shape[0]:,} frames, {data['num_clips']:,} clips)"
        )
        return data

    from datasets import Audio, load_dataset

    max_samples = int(max_hours * 3600 * SAMPLE_RATE)
    chunk_samples = int(encode_chunk_sec * SAMPLE_RATE)
    gap_samples = int(gap_sec * SAMPLE_RATE)

    token_chunks: list[torch.Tensor] = []
    wav_buf: list[torch.Tensor] = []
    buf_samples = 0
    total_samples = 0
    num_clips = 0
    skipped = 0
    t0 = time.perf_counter()

    def _flush() -> None:
        nonlocal wav_buf, buf_samples
        if not wav_buf:
            return
        cat = torch.cat(wav_buf)
        tok = codec.encode(cat)[0].cpu().to(torch.uint16)  # [T, 12]
        token_chunks.append(tok)
        wav_buf = []
        buf_samples = 0

    for dataset_id, config, split in sources:
        if total_samples >= max_samples:
            break
        print(f"Streaming {dataset_id} ({config}/{split})...")
        try:
            ds = load_dataset(dataset_id, config, split=split, streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"  [WARN] Could not open {dataset_id} ({config}/{split}): {e}")
            continue

        for row in ds:
            try:
                w = _decode_row_audio(row)
            except Exception:
                skipped += 1
                continue
            if w is None:
                skipped += 1
                continue

            if wav_buf and gap_samples > 0:
                wav_buf.append(torch.randn(gap_samples) * 1e-4)
                buf_samples += gap_samples
            wav_buf.append(w)
            buf_samples += len(w)
            total_samples += len(w)
            num_clips += 1

            if buf_samples >= chunk_samples:
                _flush()
                if num_clips % 200 == 0:
                    hrs = total_samples / (3600 * SAMPLE_RATE)
                    elapsed = time.perf_counter() - t0
                    print(f"  {hrs:.2f}h encoded ({num_clips:,} clips, {skipped} skipped, {elapsed:.0f}s)")

            if total_samples >= max_samples:
                break

    _flush()
    if not token_chunks:
        raise RuntimeError(f"No audio encoded from sources: {sources}")

    all_tokens = torch.cat(token_chunks, dim=0)  # [N_frames, 12] uint16
    hours = all_tokens.shape[0] / (FRAME_RATE * 3600)
    payload = {
        "tokens": all_tokens,
        "hours": round(hours, 4),
        "num_clips": num_clips,
        "skipped_clips": skipped,
        "sources": sources,
    }

    tmp_path = out_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / (1024**2)
    print(
        f"Saved {out_path.name}: {hours:.2f}h | {all_tokens.shape[0]:,} frames | "
        f"{num_clips:,} clips ({skipped} skipped) | {size_mb:.2f} MB in {time.perf_counter() - t0:.1f}s"
    )
    return payload


class TokenChunkDataset(Dataset):
    """Slices a contiguous [N_frames, 12] uint16 token stream into fixed seq_len windows."""

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int = 375,
        random_offset: bool = True,
        steps_per_epoch: int | None = None,
    ):
        if tokens.shape[0] < seq_len:
            raise ValueError(f"Token stream ({tokens.shape[0]} frames) shorter than seq_len ({seq_len}).")
        self.tokens = tokens
        self.seq_len = seq_len
        self.random_offset = random_offset
        self.max_start = tokens.shape[0] - seq_len
        self.num_chunks = max(1, tokens.shape[0] // seq_len)
        self._length = steps_per_epoch if (random_offset and steps_per_epoch is not None) else self.num_chunks

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.random_offset:
            start = int(torch.randint(0, self.max_start + 1, (1,)).item())
        else:
            start = (idx % self.num_chunks) * self.seq_len
        return self.tokens[start : start + self.seq_len].long()
