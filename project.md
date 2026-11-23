# Kaza

Full-duplex speech-to-speech model for African languages. Trained on audio only, no text.

Figures: `assets/` (PNG and PDF).

- `model.png`: one step of the model, layer by layer.
- `pipeline.png`: training and inference side by side.
- `summary.png`: the summary module and FSQ 8 × 8.
- `encoder.png`, `quantizer.png`, `decoder.png`: the codec.
- `model`, `pipeline` and `summary` are drawn from `assets/src/*.html`. To redraw them on Windows: `powershell -File assets/src/render.ps1 model pipeline summary` (uses Edge or Chrome, headless).

## Phases

1. Speech continuation on single-speaker audio. Details: `phase1.md`.
2. Conversation: full-duplex on two-channel data, then instruction tuning (SFT or RL). Data source not yet chosen.

## Components

| Part | Spec | Trained? | Figure |
|---|---|---|---|
| Encoder | LiquidCodec. 24 kHz audio → one 256-number vector per 80 ms. Causal. | Frozen | `assets/encoder.png` |
| Quantizer | GFSQ. 256 → 48 → 12 groups of 4 → 12 tokens (0–2,015) per frame. Levels 8, 7, 6, 6. | No weights | `assets/quantizer.png` |
| Backbone | LFM3-1.7B-Base with block expansion: 24 original layers + 6 new ones (one after every 4th), 30 total. Width 2,048. | New layers only | `assets/model.png` |
| Summary head | Linear 2,048 → 64. Predicts the summary code. | Yes | `assets/model.png` |
| Depth decoder | 3 bidirectional layers, width 1,024. Context vector + summary code → 12 codec tokens, decoded with MaskGIT. | Yes | `assets/model.png` |
| Summary module | 2-layer bidirectional Transformer, width 512, 8 heads, CLS token → Linear 512 → 2 → FSQ 8 × 8 (64 codes). | Yes, training only | `assets/summary.png` |
| Decoder | LiquidCodec. 12 tokens → 80 ms of audio (1,920 samples). | Frozen | `assets/decoder.png` |

## One step (80 ms)

- Input: embeddings of the user's frame (12 tokens) + our last frame (12 tokens) + our last summary code, summed to 2,048.
- Output: summary code, then 12 codec tokens, then 80 ms of audio.
- Time budget: under 80 ms per step.

## Summary module

- Input: our own codec tokens for the current frame and the next 4 frames (400 ms).
- Output: one code out of 64.
- Used in training only. At inference LFM3's summary head supplies the code.

## Summary code into the depth decoder

- The depth decoder receives the code as its 2 FSQ values (each rounded to one of 8 levels), through a linear layer 2 → 1,024. It does not receive a looked-up index embedding.
- Reason: the audio loss must reach the summary module through the straight-through FSQ. An index lookup blocks that gradient.
- Training: the 2 values come from the summary module.
- Inference: the summary head's chosen index is converted to its 2 values (code = a + 8b, so 18 → a=2, b=2) and passed through the same linear layer.
- FSQ levels: 8 × 8 = 64 codes (6 bits). Chosen because the FSQ paper's heuristic is at least 5 levels per dimension.

## New backbone layers

- Each is a copy of the layer below it.
- Output weights initialised to zero.
- The 24 original layers stay frozen.

## Depth decoder: MaskGIT

- The 12 tokens are side-by-side groups of one vector, not a residual stack, so no strict order between them is needed.
- Training: hide a random share of the 12 tokens each step (cosine schedule) and predict the hidden ones.
- Inference default: 3 passes × 4 tokens, most confident tokens kept first.
- One trained model covers all speeds: 1 pass = all 12 in parallel, 12 passes = one token at a time.
- Precedent: SoundStorm.

## Losses

- Audio loss: cross-entropy on the 12 codec tokens. Trains the depth decoder, the new backbone layers, the embeddings and the summary module (straight-through FSQ).
- Prior loss: cross-entropy of the summary head against the summary module's code. The code is stop-gradient.
- No usage loss. The FSQ paper (2309.15505) states that FSQ needs no auxiliary losses and reaches about 100% codebook usage (tested on about 1,000 codes and up, not 64).
- Code usage is measured in every run: a histogram of how often each summary code is used. A usage loss is added only if usage drops.

## Training schedule

1. Audio loss only. Backbone fully frozen.
2. Add the prior loss, weight ramped from 0. New backbone layers unfrozen.

## Inference

- Summary module off.
- Summary head picks the code, depth decoder writes 12 tokens, codec decoder plays 80 ms.

## Live use: codec streaming

- The released codec code encodes a whole recording in one call. It has no streaming encoder.
- Live use needs a streaming encoder that keeps three things between 80 ms pieces:
  1. Sound buffer: the last 480 samples (20 ms) of the previous piece.
  2. State of the kernel-3 conv (`frontend.conv`, weight 256 × 256 × 3): its last 2 input strokes (2 × 256). Today the code pads with zeros (`padding=2`) instead.
  3. KV cache and conv states of the encoder's 8 LFM3 layers.
- Test: streaming tokens must equal whole-recording tokens exactly.
- The codec decoder already streams (`LiquidCodecStreamingCache`).
- Delays added by the codec: encoder waits 20 ms for the next piece; decoder holds back 40 ms (960 samples) for overlap-add. Theoretical latency: 80 + 20 + 40 = 140 ms, plus compute.
- Not needed for training: training tokens are made offline from whole recordings.

## Open

- End-of-speech token: none yet (2,016 codes).
- Fine-tune the codec on African-language audio if its reconstruction is poor.

## Ablations

- Frames ahead, number of codes, codes per second.
- Summary FSQ levels: 8 × 8 (default), 4 × 4 × 4, 5 × 5 × 5 (125 codes).
- No summary code; fixed k-means units in its place.
- Block expansion vs full fine-tuning at a low learning rate.
- MaskGIT passes at inference: 1, 2, 3, 4, 12.
- Continuous head: predict the 48 numbers with flow matching, then round to the GFSQ grid.

## Reference

- LiquidCodec: STFT 1,280, hop 320; 8 encoder + 8 decoder layers, width 512; decoder attention window 30 steps (400 ms); ~73.6M parameters.
- LFM3-1.7B-Base: FFN 7,680; 13 ShortConv, 5 SWA GQA (window 256), 6 global GQA; 16 Q / 4 K/V heads.
- Papers: Moshi (arXiv 2410.00037), FLAIR (2603.17837), LLaMA Pro (2401.02415), Genie (2402.15391), FSQ (2309.15505), MaskGIT (2202.04200), SoundStorm (2305.09636).
