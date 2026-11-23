# Kaza Phase 1: speech continuation

Phase 1 teaches the model to continue speech in African languages. It uses one stream only, the agent stream: each clip is one long piece of audio with every speaker in it. Simulated two streams (Phase 2), real full-duplex conversation and instruction tuning (Phase 3) are out of scope here. The model architecture is in `project.md`.

## Order of work

1. Colab setup.
2. Codec test.
3. Overfit test on one clip.
4. Scaled Phase 1 training.

## Fixed choices

- Depth decoder: MaskGIT (see `project.md`).
- Optimizer: AdamW for all trainable parameters. Muon is an ablation for the scaled run, not used in the overfit.
- User slot in Phase 1: left out (adds 0 to the backbone input; mathematically equivalent to a fixed silence vector and avoids ~0.8 GB of unused user-channel embedding/AdamW memory until Phase 2, when the user stream first appears).
- No text anywhere, including evaluation: no transcripts and no ASR-based measures.

## Environment

| Work | Platform | GPU | Precision |
|---|---|---|---|
| Overfit test & Scaled Phase 1 | Colab Pro | L4 (22.5 GB) or A100 (40/80 GB) | Native BF16 autocast; frozen backbone in BF16, trainable weights + AdamW states in FP32 |

- Colab does not guarantee the GPU type, so the first notebook cell prints the GPU name, memory and compute capability (BF16 hardware needs 8.0 or higher; L4 is 8.9, A100 is 8.0).
- Fallback if a BF16 operation fails: FP32. Do not use FP16: LFM3 ships in BF16, and BF16-trained models can overflow in FP16.
- Trainable parameters always keep FP32 master weights (so small AdamW updates on zero-init layers do not underflow in BF16).
- Data and output folders on Colab live under `/content` (`kaza_data` and `kaza_out`).
- Code: written in this repository, pushed to GitHub, cloned in the notebook.
- Code is written one part at a time. Each part needs approval before it is written.
- Software: transformers 5.24.0. It needs Python 3.11 or later and PyTorch 2.6 or later. The first notebook cell prints the Python and PyTorch versions.
- No extra GPU kernels are needed. LFM3's Triton conv kernel runs only with gradients off. Training uses the PyTorch path.

## Models

| Model | Hugging Face repo | Size | How it loads | Licence |
|---|---|---|---|---|
| Codec | `LiquidAI/Liquid-Codec` | 147 MB | `trust_remote_code=True` (not built into transformers 5.24.0) | LFM Open License v1.0 |
| Backbone | `LiquidAI/LFM3-1.7B-Base` | 3.46 GB | Built into transformers 5.24.0 (`lfm3`) | LFM Open License v1.0 |

- Both repos are public. No token or access request is needed.
- Use the Base checkpoint, not `LFM3-1.7B`. The model card recommends Base for heavy fine-tuning.
- The codec weights are byte-identical to the codec inside `LiquidAI/LFM3-Audio-2B` (all 203 tensors match).
- Licence: commercial use is allowed only below $10M annual revenue. Research use is allowed.

## Data preparation

- Audio: 24 kHz, mono, samples in [-1, 1].
- Encode each recording in **one** codec call, start to end. Do not cut a recording before encoding: a cut makes the tokens near the cut differ from continuous encoding.
- Store 12 tokens per 80 ms frame on disk. Training reads tokens only. The codec does not run during training.
- Training example length: 30 to 45 s.

| Example length | Frames (steps) | Tokens per example |
|---|---|---|
| 30 s | 375 | 4,500 |
| 45 s | 563 | 6,756 |

- Candidate datasets:
  - WAXAL (`google/WaxalNLP`): CC-BY-4.0, open download. Mostly short single-speaker utterances (good for the codec test and short clips).
  - NaijaVoices: CC BY-NC-SA 4.0 (research / non-commercial only, ShareAlike), gated (register at naijavoices.com). About 600 h each of Hausa, Igbo and Yoruba (84 GB compressed). Mostly short single-speaker utterances.
  - Long-form / multi-speaker African audio (podcasts, talk radio, conversational corpora): needed for continuous 30–45 s Phase 1 clips without artificial concatenation and required for Phase 2 diarized two-stream simulation.

## Memory budget (estimate, to be measured in the overfit test)

- Frozen backbone in BF16: about 3.5 GB (doubles to about 7 GB if T4 falls back to FP32).
- 6 new Global GQA layers (about 360M parameters) with AdamW: about 5.8 to 6 GB.
- Trainable agent-channel embeddings (12 tables × 2,016 × 2,048 ≈ 50M params) with AdamW: about 0.8 GB.
- Depth decoder (~75M params, including 12 output heads) with AdamW: about 1.2 GB.
- Activations: roughly 1 to 1.5 GB per 45 s example.
- Total for one 30 s example in BF16: about 12.5 to 13 GB (fits on a 15 GB T4). If T4 falls back to FP32, enable gradient checkpointing right away.
- If memory runs out: gradient checkpointing.

## Attention reach at 30 to 45 s

- 5 of LFM3's 24 original layers use a sliding window of 256 steps (about 20 s).
- 12 layers use global attention and see the whole example (6 frozen original + 6 new trainable Global GQA layers).
- Anything older than about 20 s reaches the model only through the global layers.

## Codec test

Goal: decide whether LiquidCodec reconstructs African-language speech well enough, or needs fine-tuning.

Method: encode each clip to tokens, decode back to audio, compare the reconstruction with the original.

| Measure | What it shows | Tool |
|---|---|---|
| Listening | Whether it sounds right | Play original and reconstruction side by side |
| Mel distance (lower is better) | How far apart the two spectrograms are | PyTorch |
| PESQ, wideband (about 1 to 4.5) | Perceived quality | `torchmetrics`, audio resampled to 16 kHz |
| STOI (0 to 1) | Intelligibility | `torchmetrics` |
| Speaker similarity (cosine, about 0 to 1) | Same voice before and after | ECAPA speaker-embedding model |

Steps:

1. Control: one English LibriSpeech test-clean clip. Our numbers should be near Liquid AI's English numbers (PESQ 2.640, mel distance 0.120, speaker similarity 0.785). If they are not, our script is wrong.
2. African clips: Hausa, Igbo, Yoruba and Swahili, same script.

Pass: African-language scores fall inside the range Liquid AI reports for its non-English test sets:

| Measure | LiquidCodec range (Liquid AI's card) |
|---|---|
| PESQ (WB) | 2.233 to 2.637 |
| Mel distance | 0.117 to 0.127 |
| Speaker similarity (ECAPA2) | 0.766 to 0.825 |

Clearly worse scores put codec fine-tuning back on the table. Liquid AI's card does not report STOI, so STOI is compared between languages only.

## Overfit test

Data: one clip of about 30 s. This is longer than the 20 s sliding window, so it tests the global layers.

Run the checks in order. A check counts only if all checks before it pass.

| # | Check | Pass when |
|---|---|---|
| 1 | Block expansion is an identity at step 0 | Backbone output with the 6 new layers equals plain LFM3 output exactly, before any training |
| 2 | Loss falls (teacher forcing) | Audio loss reaches about 0 and token accuracy reaches 100% (including at `k = 12`, all tokens masked) |
| 3 | Frozen weights stay frozen | Checksums of the 24 original layers and the codec are unchanged after training |
| 4 | Free-running playback | Given the first 1 s, the model generates the rest (greedy MaskGIT, Gumbel noise off) and the tokens match the clip; decoded audio sounds the same |
| 5 | Step time | One generation step takes under 80 ms on the L4/A100 (measured on Colab Pro; use `torch.compile` / CUDA graphs if needed) |

Status: confirmed.

## Not needed in Phase 1

- Streaming codec encoder. It is needed only for live use (see `project.md`).
- Two-channel conversation data.

## Open decisions

1. Scaled-run evaluation: how to measure Phase 1 quality after scaling. Not yet defined.
2. Batch size for 30 to 45 s examples on the L4 (set after measuring memory in the overfit test).
