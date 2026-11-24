"""WAXAL live HF streaming dataset (language-weighted, RAM-bounded, same-speaker packing) & val cache."""

from __future__ import annotations

import io
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset, IterableDataset

from kaza.codec import FRAME_RATE, SAMPLE_RATE, SAMPLES_PER_FRAME, KazaCodec, to_mono_24k

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


def compute_language_weights(
    sources: list[tuple[str, str, str]],
    row_counts: list[int],
    lang_alpha: float = 0.5,
) -> list[float]:
    """Weight each language by (total_rows ** lang_alpha), then split proportionally within language.

    Prevents languages with more splits from getting extra weight and stops small TTS splits
    from repeating at the same rate as large unlabeled ASR splits.
    """
    lang_totals: dict[str, int] = defaultdict(int)
    langs: list[str] = []
    safe_counts = [max(1, int(c)) for c in row_counts]

    for (_, cfg, _), count in zip(sources, safe_counts):
        lang = cfg.split("_")[0]
        langs.append(lang)
        lang_totals[lang] += count

    raw_weights = [
        (lang_totals[lang] ** lang_alpha) * (count / lang_totals[lang])
        for lang, count in zip(langs, safe_counts)
    ]
    total_w = sum(raw_weights)
    return [w / total_w for w in raw_weights]


def discover_waxal_sources(
    dataset_id: str = "google/WaxalNLP",
    include_tts: bool = True,
    lang_alpha: float = 0.5,
) -> tuple[list[tuple[str, str, str]], list[float], list[tuple[str, str, str]]]:
    """Auto-discover all non-duplicate WAXAL configs (`*_asr` + `*_tts`, skipping `*_v2`).

    Returns `(train_sources, train_weights, val_sources)` using metadata-only row counts.
    """
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder

    configs = sorted(get_dataset_config_names(dataset_id))
    train_sources: list[tuple[str, str, str]] = []
    train_counts: list[int] = []
    val_sources: list[tuple[str, str, str]] = []

    for cfg in configs:
        if cfg.endswith("_v2"):
            continue
        if not include_tts and cfg.endswith("_tts"):
            continue

        split_counts: dict[str, int] = {}
        try:
            builder = load_dataset_builder(dataset_id, cfg)
            if builder.info.splits:
                for sname, sinfo in builder.info.splits.items():
                    split_counts[sname] = max(1, int(sinfo.num_examples or 1))
        except Exception:
            pass

        if not split_counts:
            try:
                for sname in get_dataset_split_names(dataset_id, cfg):
                    split_counts[sname] = 1
            except Exception as e:
                print(f"  [WARN] Could not list splits for {cfg}: {e}")
                continue

        for sname in ("train", "unlabeled"):
            if sname in split_counts:
                train_sources.append((dataset_id, cfg, sname))
                train_counts.append(split_counts[sname])
        if "validation" in split_counts:
            val_sources.append((dataset_id, cfg, "validation"))

    train_weights = compute_language_weights(train_sources, train_counts, lang_alpha=lang_alpha)
    n_langs = len({cfg.split("_")[0] for _, cfg, _ in train_sources})
    print(
        f"Discovered {dataset_id}: {len(train_sources)} train splits ({n_langs} languages, "
        f"alpha={lang_alpha}), {len(val_sources)} validation splits."
    )
    return train_sources, train_weights, val_sources


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


def _row_speaker_id(row: dict, config: str) -> str | None:
    spk = row.get("speaker_id")
    if spk is not None:
        s = str(spk).strip()
        if s and s.lower() not in ("none", "nan", "unknown"):
            return s
    if config.endswith("_tts"):
        return f"tts_{config}"
    return None


class WaxalStreamingDataset(IterableDataset):
    """Live HF streaming dataset packing same-language & same-speaker clips into exact frames.

    - Exact frame counts: `clip_samples = clip_frames * SAMPLES_PER_FRAME (1920)`.
    - Language-weighted sampling (`source_weights` from `rows^0.5` per language, proportional per split).
    - Hard per-worker RAM cap (`max_worker_mb`, default 256 MB) + `max_speakers=8` per source.
    """

    def __init__(
        self,
        sources: list[tuple[str, str, str]],
        source_weights: list[float] | None = None,
        clip_frames: int = 750,  # 60.0s; exact multiple of 250 (20s), 375 (30s), 750 (60s)
        gap_sec: float = 0.3,
        max_speakers: int = 8,
        max_worker_mb: float = 256.0,
        seed: int = 42,
        max_retries: int = 5,
    ):
        super().__init__()
        if not sources:
            raise ValueError("WaxalStreamingDataset requires at least one source.")
        self.sources = list(sources)
        if source_weights is not None:
            if len(source_weights) != len(sources):
                raise ValueError("source_weights must match length of sources.")
            self.source_weights = list(source_weights)
        else:
            self.source_weights = [1.0 / len(sources)] * len(sources)
        self.clip_frames = clip_frames
        self.clip_samples = clip_frames * SAMPLES_PER_FRAME
        self.gap_samples = int(round(gap_sec * SAMPLE_RATE))
        self.max_speakers = max_speakers
        self.max_worker_samples = int((max_worker_mb * 1024 * 1024) // 4)  # float32 = 4 bytes
        self.seed = seed
        self.max_retries = max_retries

    def _open_source_iter(self, dataset_id: str, config: str, split: str, epoch_seed: int):
        from datasets import Audio, load_dataset

        ds = load_dataset(dataset_id, config, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.shuffle(seed=epoch_seed, buffer_size=200)
        return iter(ds)

    def _append_to_bucket(self, bucket: list[torch.Tensor], w: torch.Tensor) -> int:
        added = 0
        if bucket and self.gap_samples > 0:
            bucket.append(torch.randn(self.gap_samples) * 1e-4)
            added += self.gap_samples
        bucket.append(w)
        added += len(w)
        return added

    def _pop_clip_from_bucket(self, bucket: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        cat = torch.cat(bucket)
        clip = cat[: self.clip_samples].contiguous()
        rem = cat[self.clip_samples :]
        if len(rem) > 0:
            return clip, [rem], len(rem)
        return clip, [], 0

    def _enforce_worker_ram_cap(self, states: list[dict]) -> None:
        """Evict oldest speaker buckets across all sources if worker buffer exceeds max_worker_mb."""
        total = sum(s["spk_samples"] + s["mixed_len"] for s in states)
        while total > self.max_worker_samples:
            # Pick the source holding the most speaker-bucket samples
            candidates = [s for s in states if s["spk_bufs"]]
            if not candidates:
                break
            victim = max(candidates, key=lambda s: s["spk_samples"])
            _, (old_buf, old_len) = victim["spk_bufs"].popitem(last=False)
            victim["spk_samples"] -= old_len
            if old_buf:
                added = self._append_to_bucket(victim["mixed_buf"], torch.cat(old_buf))
                victim["mixed_len"] += added
                total = total - old_len + added
            else:
                total -= old_len
            # If moving to mixed didn't reduce total (just moved bytes), drop oldest mixed tail if still over cap
            if total > self.max_worker_samples and victim["mixed_len"] > self.clip_samples * 2:
                cat = torch.cat(victim["mixed_buf"])
                keep = cat[-self.clip_samples :]
                dropped = len(cat) - len(keep)
                victim["mixed_buf"] = [keep]
                victim["mixed_len"] = len(keep)
                total -= dropped

    def _next_clip_from_source(self, state: dict, all_states: list[dict]) -> torch.Tensor | None:
        """Pull utterances from one language stream until a speaker (or fallback) bucket fills."""
        spk_bufs: OrderedDict[str, tuple[list[torch.Tensor], int]] = state["spk_bufs"]
        retries = 0

        while True:
            if state["mixed_len"] >= self.clip_samples:
                clip, mbuf, mlen = self._pop_clip_from_bucket(state["mixed_buf"])
                state["mixed_buf"], state["mixed_len"] = mbuf, mlen
                return clip

            if state["iter"] is None:
                try:
                    state["epoch"] += 1
                    state["iter"] = self._open_source_iter(
                        state["dataset_id"],
                        state["config"],
                        state["split"],
                        self.seed + state["epoch"] * 1009,
                    )
                except Exception as e:
                    retries += 1
                    if retries > self.max_retries:
                        print(f"  [WARN] Open failed for {state['config']}/{state['split']}: {e}")
                        return None
                    time.sleep(min(2**retries, 16))
                    continue

            try:
                row = next(state["iter"])
                retries = 0
            except StopIteration:
                state["iter"] = None
                continue
            except Exception as e:
                retries += 1
                state["iter"] = None
                if retries > self.max_retries:
                    print(f"  [WARN] Stream error on {state['config']}/{state['split']}: {e}")
                    return None
                time.sleep(min(2**retries, 16))
                continue

            try:
                w = _decode_row_audio(row)
            except Exception:
                continue
            if w is None:
                continue

            spk = _row_speaker_id(row, state["config"])
            if spk is None:
                state["mixed_len"] += self._append_to_bucket(state["mixed_buf"], w)
                if state["mixed_len"] >= self.clip_samples:
                    clip, mbuf, mlen = self._pop_clip_from_bucket(state["mixed_buf"])
                    state["mixed_buf"], state["mixed_len"] = mbuf, mlen
                    return clip
            else:
                if spk in spk_bufs:
                    buf, blen = spk_bufs.pop(spk)
                    state["spk_samples"] -= blen
                else:
                    buf, blen = [], 0
                blen += self._append_to_bucket(buf, w)

                if blen >= self.clip_samples:
                    clip, rem_buf, rem_len = self._pop_clip_from_bucket(buf)
                    if rem_len > 0:
                        spk_bufs[spk] = (rem_buf, rem_len)
                        state["spk_samples"] += rem_len
                    return clip

                spk_bufs[spk] = (buf, blen)
                state["spk_samples"] += blen
                while len(spk_bufs) > self.max_speakers:
                    _, (old_buf, old_len) = spk_bufs.popitem(last=False)
                    state["spk_samples"] -= old_len
                    if old_buf:
                        state["mixed_len"] += self._append_to_bucket(
                            state["mixed_buf"], torch.cat(old_buf)
                        )
                self._enforce_worker_ram_cap(all_states)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id * 9973)

        paired = list(zip(self.sources, self.source_weights))
        if worker_info is not None and worker_info.num_workers > 1:
            my_paired = paired[worker_id :: worker_info.num_workers] or paired
        else:
            my_paired = paired

        initial_count = len(my_paired)
        states = [
            {
                "dataset_id": d,
                "config": c,
                "split": s,
                "weight": w,
                "iter": None,
                "epoch": worker_id,
                "spk_bufs": OrderedDict(),
                "spk_samples": 0,
                "mixed_buf": [],
                "mixed_len": 0,
            }
            for (d, c, s), w in my_paired
        ]

        while states:
            weights = [s["weight"] for s in states]
            idx = rng.choices(range(len(states)), weights=weights, k=1)[0]
            clip = self._next_clip_from_source(states[idx], states)
            if clip is None:
                dropped = states.pop(idx)
                print(
                    f"[ALERT] Worker {worker_id} dropped {dropped['config']}/{dropped['split']} "
                    f"— {len(states)}/{initial_count} active sources remaining."
                )
                continue
            yield clip


def build_token_cache(
    sources: list[tuple[str, str, str]],
    codec: KazaCodec,
    out_path: str | Path,
    max_hours: float = 0.5,
    encode_chunk_frames: int = 375,
    gap_sec: float = 0.3,
    overwrite: bool = False,
) -> dict:
    """Stream HF audio datasets into a compact uint16 `.pt` cache balanced across all sources."""
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
    chunk_samples = encode_chunk_frames * SAMPLES_PER_FRAME
    per_source_cap = max(chunk_samples, max_samples // max(1, len(sources)))
    gap_samples = int(round(gap_sec * SAMPLE_RATE))

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
        tok = codec.encode(cat)[0].cpu().to(torch.uint16)
        token_chunks.append(tok)
        wav_buf = []
        buf_samples = 0

    for dataset_id, config, split in sources:
        if total_samples >= max_samples:
            break
        src_samples = 0
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
            src_samples += len(w)
            total_samples += len(w)
            num_clips += 1

            if buf_samples >= chunk_samples:
                _flush()
            if src_samples >= per_source_cap or total_samples >= max_samples:
                break

        _flush()

    if not token_chunks:
        raise RuntimeError(f"No audio encoded from sources: {sources}")

    all_tokens = torch.cat(token_chunks, dim=0)
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
        f"Saved {out_path.name}: {hours:.2f}h across {len(sources)} sources | "
        f"{all_tokens.shape[0]:,} frames | {num_clips:,} clips ({skipped} skipped) | "
        f"{size_mb:.2f} MB in {time.perf_counter() - t0:.1f}s"
    )
    return payload


class TokenChunkDataset(Dataset):
    """Slices a contiguous [N_frames, 12] uint16 token tensor into fixed seq_len windows."""

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
