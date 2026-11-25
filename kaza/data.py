"""WAXAL live HF streaming (language-weighted, RAM-bounded packing) and the balanced validation cache."""

from __future__ import annotations

import io
import queue
import random
import statistics
import threading
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset, IterableDataset

from kaza.codec import FRAME_RATE, SAMPLE_RATE, SAMPLES_PER_FRAME, KazaCodec, to_mono_24k

Source = tuple[str, str, str]  # (dataset_id, config, split)

MIN_UTTERANCE_SEC = 0.4  # skip ultra-short artifacts
GAP_NOISE_STD = 1e-4  # low-level room tone between packed utterances
SHUFFLE_BUFFER_ROWS = 16 #200  # HF streaming shuffle buffer per open source
MAX_BACKOFF_SEC = 16
EPOCH_SEED_STRIDE = 1009
MAX_CONSECUTIVE_DECODE_FAILURES = 500
BYTES_PER_SAMPLE = 4  # float32


def _language_of(config: str) -> str:
    return config.split("_")[0]


def compute_language_weights(
    sources: list[Source],
    row_counts: list[int],
    lang_alpha: float = 0.5,
) -> list[float]:
    """Weight each language by (total_rows ** lang_alpha), then split proportionally within language.

    Prevents languages with more splits from getting extra weight and stops small TTS splits
    from repeating at the same rate as large unlabeled ASR splits.
    """
    lang_totals: dict[str, int] = defaultdict(int)
    safe_counts = [max(1, int(c)) for c in row_counts]
    langs = [_language_of(cfg) for _, cfg, _ in sources]
    for lang, count in zip(langs, safe_counts):
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
) -> tuple[list[Source], list[float], list[Source]]:
    """Auto-discover WAXAL `*_asr` + `*_tts` configs and their split sizes.

    A `*_v2` config is skipped only when its base config also exists. Row counts come from
    dataset metadata; a split with unknown size gets the median known count (with a warning)
    so a metadata failure cannot silently drop a language.

    Returns `(train_sources, train_weights, val_sources)`.
    """
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder

    configs = sorted(get_dataset_config_names(dataset_id))
    config_set = set(configs)
    train_sources: list[Source] = []
    train_counts: list[int | None] = []
    val_sources: list[Source] = []

    for cfg in configs:
        if cfg.endswith("_v2") and cfg[: -len("_v2")] in config_set:
            continue
        if not include_tts and "_tts" in cfg:
            continue

        split_counts: dict[str, int | None] = {}
        try:
            builder = load_dataset_builder(dataset_id, cfg)
            for sname, sinfo in (builder.info.splits or {}).items():
                split_counts[sname] = int(sinfo.num_examples) if sinfo.num_examples else None
        except Exception as e:
            print(f"  [WARN] No split metadata for {cfg}: {e}")

        if not split_counts:
            try:
                split_counts = {sname: None for sname in get_dataset_split_names(dataset_id, cfg)}
            except Exception as e:
                print(f"  [WARN] Skipping {cfg}: could not list splits ({e})")
                continue

        for sname in ("train", "unlabeled"):
            if sname in split_counts:
                train_sources.append((dataset_id, cfg, sname))
                train_counts.append(split_counts[sname])
        if "validation" in split_counts:
            val_sources.append((dataset_id, cfg, "validation"))

    known = [c for c in train_counts if c]
    fallback = int(statistics.median(known)) if known else 1
    unknown = [f"{c}/{s}" for (_, c, s), n in zip(train_sources, train_counts) if not n]
    if unknown:
        print(f"  [WARN] Unknown row counts for {unknown}; using median {fallback:,} rows.")
    counts = [c if c else fallback for c in train_counts]

    train_weights = compute_language_weights(train_sources, counts, lang_alpha=lang_alpha)
    n_langs = len({_language_of(cfg) for _, cfg, _ in train_sources})
    print(
        f"Discovered {dataset_id}: {len(train_sources)} train splits ({n_langs} languages, "
        f"alpha={lang_alpha}), {len(val_sources)} validation splits."
    )
    return train_sources, train_weights, val_sources


def _decode_row_audio(row: dict) -> torch.Tensor | None:
    """Decode an `Audio(decode=False)` row to 24 kHz mono float32, or None if unusable."""
    audio = row.get("audio")
    if not audio:
        return None
    if audio.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(audio["bytes"]))
    elif audio.get("path") is not None:
        wav, sr = sf.read(audio["path"])
    else:
        return None
    if len(wav) < int(MIN_UTTERANCE_SEC * sr):
        return None
    return to_mono_24k(wav, int(sr))


def _row_speaker_id(row: dict, config: str) -> str | None:
    spk = row.get("speaker_id")
    if spk is not None:
        s = str(spk).strip()
        if s and s.lower() not in ("none", "nan", "unknown"):
            return s
    # Assumes a TTS config without a speaker column holds a single voice
    if "_tts" in config:
        return f"tts_{config}"
    return None


class WaxalStreamingDataset(IterableDataset):
    """Live HF streaming of language-weighted WAXAL sources, packed into exact-length clips.

    - Every clip is exactly `clip_frames * SAMPLES_PER_FRAME` samples.
    - Each clip holds one language. Same-speaker packing is best effort: up to `max_speakers`
      partial speaker buckets per source; evicted buckets fall back to a same-language bucket.
    - `max_buffer_mb` bounds the packing buffers only (not HF's internal shuffle buffers).
    - A source is dropped (with an alert) after repeated network errors, an epoch with no
      usable rows, or `MAX_CONSECUTIVE_DECODE_FAILURES` decode failures in a row.

    Iterate in the main process (e.g. through `StreamBatcher`). HF `datasets` splits streams
    across DataLoader workers on its own, which would lose rows and distort the weights.
    """

    def __init__(
        self,
        sources: list[Source],
        source_weights: list[float] | None = None,
        clip_frames: int = 750,
        gap_sec: float = 0.3,
        max_speakers: int = 8, #16,
        max_buffer_mb: float = 256.0, #1024.0,
        seed: int = 42,
        max_retries: int = 5,
    ):
        super().__init__()
        if not sources:
            raise ValueError("WaxalStreamingDataset requires at least one source.")
        if source_weights is not None and len(source_weights) != len(sources):
            raise ValueError("source_weights must match length of sources.")
        self.sources = list(sources)
        self.source_weights = (
            list(source_weights) if source_weights is not None else [1.0 / len(sources)] * len(sources)
        )
        self.clip_samples = clip_frames * SAMPLES_PER_FRAME
        self.gap_samples = int(round(gap_sec * SAMPLE_RATE))
        self.max_speakers = max_speakers
        self.max_buffer_samples = int(max_buffer_mb * 1024 * 1024) // BYTES_PER_SAMPLE
        self.seed = seed
        self.max_retries = max_retries
        self.active_sources = len(self.sources)

    # ----- buffers -------------------------------------------------------------------------

    def _append_to_bucket(self, bucket: list[torch.Tensor], w: torch.Tensor) -> int:
        added = 0
        if bucket and self.gap_samples > 0:
            bucket.append(torch.randn(self.gap_samples) * GAP_NOISE_STD)
            added += self.gap_samples
        bucket.append(w)
        return added + len(w)

    def _pop_clip(self, bucket: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        """Split one clip off a bucket. Clones so no slice keeps the concatenated parent alive."""
        cat = torch.cat(bucket)
        clip = cat[: self.clip_samples].clone()
        rem = cat[self.clip_samples :].clone()
        return (clip, [rem], len(rem)) if len(rem) else (clip, [], 0)

    @staticmethod
    def _buffered(states: list[dict]) -> int:
        return sum(s["spk_samples"] + s["mixed_len"] for s in states)

    def _evict_oldest_speaker(self, state: dict) -> None:
        _, (old_buf, old_len) = state["spk_bufs"].popitem(last=False)
        state["spk_samples"] -= old_len
        if old_buf:
            state["mixed_len"] += self._append_to_bucket(state["mixed_buf"], torch.cat(old_buf))

    def _enforce_buffer_cap(self, states: list[dict]) -> None:
        """Keep packing buffers under `max_buffer_mb`.

        First moves the oldest speaker buckets into same-language buckets. When no speaker
        buckets remain, trims the largest same-language bucket (oldest audio is dropped).
        """
        while self._buffered(states) > self.max_buffer_samples:
            with_spk = [s for s in states if s["spk_bufs"]]
            if with_spk:
                self._evict_oldest_speaker(max(with_spk, key=lambda s: s["spk_samples"]))
                continue
            victim = max(states, key=lambda s: s["mixed_len"])
            if victim["mixed_len"] == 0:
                return
            if victim["mixed_len"] > self.clip_samples:
                keep = torch.cat(victim["mixed_buf"])[-self.clip_samples :].clone()
                victim["mixed_buf"], victim["mixed_len"] = [keep], len(keep)
            else:
                victim["mixed_buf"], victim["mixed_len"] = [], 0

    # ----- per-source streaming ------------------------------------------------------------

    def _open_source_iter(self, dataset_id: str, config: str, split: str, epoch_seed: int):
        from datasets import Audio, load_dataset

        ds = load_dataset(dataset_id, config, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.shuffle(seed=epoch_seed, buffer_size=SHUFFLE_BUFFER_ROWS)
        return iter(ds)

    def _new_state(self, source: Source, weight: float) -> dict:
        dataset_id, config, split = source
        return {
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "weight": weight,
            "iter": None,
            "epoch": 0,
            "usable_in_epoch": 0,
            "decode_fail_streak": 0,
            "decode_fail_total": 0,
            "spk_bufs": OrderedDict(),
            "spk_samples": 0,
            "mixed_buf": [],
            "mixed_len": 0,
        }

    def _next_clip_from_source(self, state: dict, all_states: list[dict]) -> torch.Tensor | None:
        """Pull utterances from one source until a speaker or same-language bucket fills a clip.

        Returns None when the source must be dropped.
        """
        name = f"{state['config']}/{state['split']}"
        spk_bufs: OrderedDict[str, tuple[list[torch.Tensor], int]] = state["spk_bufs"]
        retries = 0

        while True:
            if state["mixed_len"] >= self.clip_samples:
                clip, state["mixed_buf"], state["mixed_len"] = self._pop_clip(state["mixed_buf"])
                return clip

            if state["iter"] is None:
                try:
                    state["epoch"] += 1
                    state["usable_in_epoch"] = 0
                    state["iter"] = self._open_source_iter(
                        state["dataset_id"],
                        state["config"],
                        state["split"],
                        self.seed + state["epoch"] * EPOCH_SEED_STRIDE,
                    )
                except Exception as e:
                    retries += 1
                    if retries > self.max_retries:
                        print(f"  [WARN] Open failed for {name}: {e}")
                        return None
                    time.sleep(min(2**retries, MAX_BACKOFF_SEC))
                    continue

            try:
                row = next(state["iter"])
                retries = 0
            except StopIteration:
                state["iter"] = None
                if state["usable_in_epoch"] == 0:
                    print(f"  [WARN] {name}: a full pass yielded no usable audio.")
                    return None
                continue
            except Exception as e:
                retries += 1
                state["iter"] = None
                if retries > self.max_retries:
                    print(f"  [WARN] Stream error on {name}: {e}")
                    return None
                time.sleep(min(2**retries, MAX_BACKOFF_SEC))
                continue

            try:
                w = _decode_row_audio(row)
            except Exception:
                w = None
                state["decode_fail_streak"] += 1
                state["decode_fail_total"] += 1
                if state["decode_fail_streak"] >= MAX_CONSECUTIVE_DECODE_FAILURES:
                    print(f"  [WARN] {name}: {state['decode_fail_streak']} decode failures in a row.")
                    return None
            if w is None:
                continue
            state["decode_fail_streak"] = 0
            state["usable_in_epoch"] += 1

            spk = _row_speaker_id(row, state["config"])
            if spk is None:
                state["mixed_len"] += self._append_to_bucket(state["mixed_buf"], w)
                self._enforce_buffer_cap(all_states)
                continue

            buf, blen = spk_bufs.pop(spk, ([], 0))
            state["spk_samples"] -= blen
            blen += self._append_to_bucket(buf, w)

            if blen >= self.clip_samples:
                clip, rem_buf, rem_len = self._pop_clip(buf)
                if rem_len:
                    spk_bufs[spk] = (rem_buf, rem_len)
                    state["spk_samples"] += rem_len
                self._enforce_buffer_cap(all_states)
                return clip

            spk_bufs[spk] = (buf, blen)
            state["spk_samples"] += blen
            while len(spk_bufs) > self.max_speakers:
                self._evict_oldest_speaker(state)
            self._enforce_buffer_cap(all_states)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "WaxalStreamingDataset must be iterated in the main process "
                "(HF datasets shards streams per DataLoader worker on its own). Use StreamBatcher."
            )

        rng = random.Random(self.seed)
        states = [self._new_state(src, w) for src, w in zip(self.sources, self.source_weights)]
        initial_count = len(states)
        self.active_sources = initial_count

        while states:
            idx = rng.choices(range(len(states)), weights=[s["weight"] for s in states], k=1)[0]
            clip = self._next_clip_from_source(states[idx], states)
            if clip is None:
                dropped = states.pop(idx)
                self.active_sources = len(states)
                print(
                    f"[ALERT] Dropped {dropped['config']}/{dropped['split']} "
                    f"({dropped['decode_fail_total']} decode failures) "
                    f"- {len(states)}/{initial_count} sources remain."
                )
                continue
            yield clip


class StreamBatcher:
    """Collects clips from a streaming dataset into `[B, samples]` batches on a background thread.

    Keeps up to `prefetch` batches ready so HF download, decoding and resampling overlap with
    GPU work. Raises StopIteration when the dataset is exhausted and re-raises producer errors.
    """

    _DONE = object()

    def __init__(self, dataset: IterableDataset, batch_size: int, prefetch: int = 4):
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._produce, args=(dataset, batch_size), daemon=True, name="StreamBatcher"
        )
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self, dataset: IterableDataset, batch_size: int) -> None:
        try:
            clips: list[torch.Tensor] = []
            for clip in dataset:
                clips.append(clip)
                if len(clips) == batch_size:
                    if not self._put(torch.stack(clips)):
                        return
                    clips = []
            self._put(self._DONE)
        except BaseException as e:  # forward any producer failure to the consumer
            self._put(e)

    def __iter__(self) -> StreamBatcher:
        return self

    def __next__(self) -> torch.Tensor:
        item = self._queue.get()
        if item is self._DONE:
            raise StopIteration
        if isinstance(item, BaseException):
            raise RuntimeError("Streaming producer failed.") from item
        return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def build_token_cache(
    sources: list[Source],
    codec: KazaCodec,
    out_path: str | Path,
    max_hours: float = 0.5,
    encode_chunk_frames: int = 375,
    gap_sec: float = 0.3,
    overwrite: bool = False,
) -> dict:
    """Stream HF audio into a compact uint16 `.pt` token cache with an equal audio budget per source.

    An existing cache is reused only if it was built from the same sources.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        data = torch.load(out_path, map_location="cpu", weights_only=True)
        cached = [tuple(s) for s in data.get("sources", [])]
        if cached != [tuple(s) for s in sources]:
            raise ValueError(
                f"{out_path.name} was built from different sources. "
                "Pass overwrite=True (or a new out_path) to rebuild it."
            )
        print(
            f"Loaded cached tokens: {out_path.name} | "
            f"{data['hours']:.2f}h ({data['tokens'].shape[0]:,} frames, {data['num_clips']:,} utterances)"
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
    num_utterances = 0
    skipped = 0
    t0 = time.perf_counter()

    def _flush() -> None:
        nonlocal wav_buf, buf_samples
        if not wav_buf:
            return
        tok = codec.encode(torch.cat(wav_buf))[0].cpu().to(torch.uint16)
        token_chunks.append(tok)
        wav_buf, buf_samples = [], 0

    for dataset_id, config, split in sources:
        if total_samples >= max_samples:
            break
        src_samples = 0
        try:
            ds = load_dataset(dataset_id, config, split=split, streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            for row in ds:
                try:
                    w = _decode_row_audio(row)
                except Exception:
                    w = None
                if w is None:
                    skipped += 1
                    continue

                if wav_buf and gap_samples > 0:
                    wav_buf.append(torch.randn(gap_samples) * GAP_NOISE_STD)
                    buf_samples += gap_samples
                wav_buf.append(w)
                buf_samples += len(w)
                src_samples += len(w)
                total_samples += len(w)
                num_utterances += 1

                if buf_samples >= chunk_samples:
                    _flush()
                if src_samples >= per_source_cap or total_samples >= max_samples:
                    break
        except Exception as e:
            print(f"  [WARN] {config}/{split} stopped early after {src_samples / SAMPLE_RATE:.0f}s: {e}")
        if src_samples < per_source_cap:
            print(f"  [WARN] {config}/{split} gave {src_samples / SAMPLE_RATE:.0f}s of a {per_source_cap / SAMPLE_RATE:.0f}s budget.")
        _flush()

    if not token_chunks:
        raise RuntimeError(f"No audio encoded from sources: {sources}")

    all_tokens = torch.cat(token_chunks, dim=0)
    hours = all_tokens.shape[0] / (FRAME_RATE * 3600)
    payload = {
        "tokens": all_tokens,
        "hours": round(hours, 4),
        "num_clips": num_utterances,
        "skipped_clips": skipped,
        "sources": sources,
    }

    tmp_path = out_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / (1024**2)
    print(
        f"Saved {out_path.name}: {hours:.2f}h across {len(sources)} sources | "
        f"{all_tokens.shape[0]:,} frames | {num_utterances:,} utterances ({skipped} skipped) | "
        f"{size_mb:.2f} MB in {time.perf_counter() - t0:.1f}s"
    )
    return payload


class TokenChunkDataset(Dataset):
    """Splits a contiguous [N_frames, 12] uint16 token tensor into consecutive seq_len windows."""

    def __init__(self, tokens: torch.Tensor, seq_len: int = 375):
        if tokens.shape[0] < seq_len:
            raise ValueError(f"Token stream ({tokens.shape[0]} frames) shorter than seq_len ({seq_len}).")
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_chunks = tokens.shape[0] // seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        return self.tokens[start : start + self.seq_len].long()
