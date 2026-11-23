# Kaza Phase 1: speech continuation

Phase 1 teaches the model to continue speech in African languages from single-speaker audio. Conversation (full-duplex, two speakers) and instruction tuning (SFT or RL) are Phase 2 and are out of scope here. The model architecture is in `project.md`.

## Order of work

1. Colab setup.
2. Codec test.
3. Overfit test on one clip.
4. Scaled Phase 1 training.

## Fixed choices

- Depth decoder: MaskGIT (see `project.md`).
- Optimizer: AdamW for all trainable parameters. Muon is an ablation for the scaled run, not used in the overfit.
- No text anywhere, including evaluation: no transcripts and no ASR-based measures.

## Environment

| Work | Platform | GPU | Precision |
|---|---|---|---|
| Overfit test | Free Colab or Kaggle | T4, about 15 to 16 GB (Kaggle gives 2 × T4; we use one) | BF16, emulated in software on the T4 (no BF16 hardware); slower, same numbers as the L4 |
| Scaled Phase 1 training | Colab Pro | L4, 22.5 GB usable | BF16, native |

- Colab does not guarantee the GPU type, so the first notebook cell prints the GPU name, memory and compute capability (BF16 hardware needs 8.0 or higher; the T4 is 7.5).
- Fallback if a BF16 operation fails on the T4: FP32. Do not use FP16: LFM3 ships in BF16, and BF16-trained models can overflow in FP16.
- The same code runs on all three platforms. Only the data and output folders differ (Colab: `/content`, Kaggle: `/kaggle/working`).
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
  - WAXAL (`google/WaxalNLP`): CC-BY-4.0, open download.
  - NaijaVoices: CC BY-NC-SA 4.0, research use only, gated (register at naijavoices.com). About 600 h each of Hausa, Igbo and Yoruba. Compressed copy is 84 GB.

## Memory budget (estimate, to be measured in the overfit test)

- Frozen backbone in BF16: about 3.5 GB.
- 6 new layers (about 370M parameters) with Adam: about 6 GB.
- Depth decoder and summary module: small.
- Activations: roughly 1 to 1.5 GB per 45 s example.
- Total for one 30 s example: about 11 GB, so the overfit fits on a T4.
- If memory runs out: gradient checkpointing.

## Attention reach at 30 to 45 s

- 5 of LFM3's 24 original layers use a sliding window of 256 steps (about 20 s).
- 6 layers use global attention and see the whole example.
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
| 2 | Loss falls (teacher forcing) | Audio loss and prior loss reach about 0; token accuracy and summary-code accuracy reach 100% |
| 3 | Frozen weights stay frozen | Checksums of the 24 original layers and the codec are unchanged after training |
| 4 | Free-running playback | Given the first 1 s, the model generates the rest (summary head + MaskGIT) and the tokens match the clip; decoded audio sounds the same |
| 5 | Step time | One generation step takes under 80 ms on the L4 (measured on Colab Pro, not on the T4) |

Status: proposed, not yet confirmed.

## Not needed in Phase 1

- Streaming codec encoder. It is needed only for live use (see `project.md`).
- Two-channel conversation data.

## Open decisions

1. User slot: each step takes a user frame, but Phase 1 audio has one speaker. Options: a fixed silence token, or leave the user input out.
2. Overfit checks: confirm the 5 checks above.
3. Scaled-run evaluation: how to measure Phase 1 quality after scaling. Not yet defined.
4. Batch size for 30 to 45 s examples on the L4 (set after measuring memory in the overfit test).
