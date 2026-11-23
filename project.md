# Kaza

Full-duplex speech-to-speech model for African languages. Trained on audio only, no text.

Figures: `assets/` (PNG and PDF).

- `model.png`: one step of the model, layer by layer.
- `pipeline.png`: training and inference side by side.
- `encoder.png`, `quantizer.png`, `decoder.png`: the codec.
- `model` and `pipeline` are drawn from `assets/src/*.html`. To redraw them on Windows: `powershell -File assets/src/render.ps1 model pipeline` (uses Edge or Chrome, headless).

## Phases

1. Speech continuation. One stream only, the agent stream: each clip is one long piece of audio, with every speaker in it. The user slot is left out. Details: `phase1.md`.
2. Simulated two streams (Moshi's post-training method). Diarize multi-speaker African mono audio (e.g. podcasts, talk radio, conversational recordings; Moshi used PyAnnote), pick one speaker at random as the agent, and put everything else in the user stream. Encode the two waveforms separately (with low-level noise augmentation on diarized digital-silence spans so they match live mic room tone). Zero-initialise the new user-channel embedding tables at step 0 so Phase 2 starts identically to Phase 1.
3. Conversation: full-duplex on real two-channel data, then instruction tuning (SFT or RL). Data source not yet chosen.

Terms: **agent** = the voice Kaza produces; **user** = the other side.

## Components

| Part | Spec | Trained? | Figure |
|---|---|---|---|
| Encoder | LiquidCodec. 24 kHz audio → one 256-number vector per 80 ms. Causal. | Frozen | `assets/encoder.png` |
| Quantizer | GFSQ. 256 → 48 → 12 groups of 4 → 12 tokens (0–2,015) per frame. Levels 8, 7, 6, 6. | No weights | `assets/quantizer.png` |
| Backbone | LFM3-1.7B-Base with block expansion: 24 original layers + 6 new Global GQA layers (one after each Global GQA layer), 30 total. Width 2,048. | New layers only | `assets/model.png` |
| Depth decoder | 3 bidirectional layers, width 1,024. Context vector → 12 codec tokens, decoded with MaskGIT. | Yes | `assets/model.png` |
| Decoder | LiquidCodec. 12 tokens → 80 ms of audio (1,920 samples). | Frozen | `assets/decoder.png` |

## One step (80 ms)

- Input: embeddings of the user's frame (12 tokens) + the agent's last frame (12 tokens), summed to 2,048. LiquidCodec is used as-is; Kaza uses 12 separate embedding tables per audio channel (2,016 entries each) so the 12 GFSQ groups do not collide when summed. At `t = 0` (before the agent has spoken), `agent t−1` uses a learned `BOS` embedding.
- Output: the depth decoder writes 12 codec tokens from the backbone's context vector, then the codec decoder plays 80 ms of audio.
- Time budget: under 80 ms per step (including streaming codec encode/decode; use `torch.compile` / CUDA graphs as needed on L4/A100).

## New backbone layers

- 6 new Global GQA + FFN layers, inserted right after each of the 6 original Global GQA + FFN layers (layers 2, 6, 10, 14, 18, 22; still spaced 4 original layers apart).
- Each is a copy of the Global GQA + FFN layer below it.
- Output weights (`o_proj` and FFN `down_proj`, plus any biases) initialised to zero so the expanded backbone is an exact identity at step 0.
- HF `LFM3` wiring: update `config.num_hidden_layers` (24 → 30), insert `"full_attention"` into `config.layer_types` at the 6 new positions, re-index `layer_idx` on all 30 attention/ShortConv modules so KV/conv caches stay aligned, and drop unused text `embed_tokens` / `lm_head`.
- The 24 original layers stay frozen (stored in BF16). All trainable parameters and AdamW states stay in FP32 master weights, with forward/backward under `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.

## Depth decoder: MaskGIT

- The 12 tokens are side-by-side groups of one vector, not a residual stack, so no strict order between them is needed.
- Conditioning: project the backbone's `2,048` context vector via `Linear(2,048 → 1,024)` and add it to every token slot.
- Embeddings & heads: 12 separate slot embedding tables (2,016 tokens + 1 `[MASK]` token = 2,017 entries; slot identity is carried by the per-slot tables) and 12 separate linear output heads (2,016 classes each).
- Training: hide a random share of the 12 tokens each step using `ceil(12 * cos(pi/2 * u))` (so `k = 12` all-masked receives ~26% of steps, matching inference pass 1, and at least 1 token is always masked) and compute cross-entropy only on the masked positions.
- Inference default: 3 passes × 4 tokens, most confident tokens kept first (with Gumbel noise / decaying temperature on confidence scores, as in SoundStorm; disabled for greedy overfit checks).
- One trained model covers all speeds: 1 pass = all 12 in parallel, 12 passes = one token at a time.
- Precedent: SoundStorm.

## Losses & pretraining consistency

- Audio loss: cross-entropy on the masked codec tokens of frame `t` (every step). Trains the depth decoder, the 6 new backbone layers, and the embeddings.
- Scaled-run consistency regularizers (off during the overfit test, turned on for scaled Phase 1):
  - **Multi-frame prediction (MTP, training only):** auxiliary prediction of future frames (e.g. `t+2` and `t+4`, 160 ms and 320 ms ahead) from `context_t` via lightweight offset-specific projections (`Linear(2,048 → 512) → Linear(512 → 1,024)` feeding the shared depth decoder's output heads, keeping MTP overhead tiny) with a small loss weight (e.g. `0.2`), forcing `context_t` to plan syllables/words ahead with zero cost at inference.
  - **Input token corruption:** randomly zero out 5–10% of the 12 slot embeddings in the agent's last frame (`agent t−1`) during training (no extra `[MASK]` entry needed in the 2,016-vocab backbone tables) so the backbone relies on long-range Global GQA context instead of blindly copying the last frame.
- Long-range conversational coherence and instruction following are further strengthened via RL in Phase 3.

## Training schedule

1. Brief warmup: embeddings and depth decoder only, backbone fully frozen.
2. Unfreeze the 6 new Global GQA backbone layers and train end-to-end on the audio loss (plus MTP and light input corruption in scaled runs).

## Inference

- Every 80 ms step: backbone updates its KV/conv cache from `user t` and `agent t−1`, depth decoder writes 12 tokens (3 passes × 4 tokens), codec decoder plays 80 ms.

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

- Finite-clip termination: full-duplex turn-taking uses silence tokens rather than EOS; an EOS token is only needed if we later want explicit termination on finite offline clips.
- Fine-tune the codec on African-language audio if its reconstruction is poor.

## Ablations

- Multi-frame prediction offsets (`t+2`, `t+4`) and loss weight; input corruption rate (0% vs 5–10%).
- Block expansion (Global GQA vs ShortConv) vs full fine-tuning at a low learning rate.
- MaskGIT passes at inference: 1, 2, 3, 4, 12.
- Continuous head: predict the 48 numbers with flow matching, then round to the GFSQ grid.
- Adding a frozen multilingual SSL semantic stream (e.g. mHuBERT-147 at 12.5 Hz) if acoustic-only pretraining drifts.

## Reference

- LiquidCodec: STFT 1,280, hop 320; 8 encoder + 8 decoder layers, width 512; decoder attention window 30 steps (400 ms); ~73.6M parameters.
- LFM3-1.7B-Base: FFN 7,680; 13 ShortConv, 5 SWA GQA (window 256), 6 global GQA; 16 Q / 4 K/V heads.
- Papers: Moshi (arXiv 2410.00037), LLaMA Pro (2401.02415), MaskGIT (2202.04200), SoundStorm (2305.09636), Multi-Token Prediction (2404.19737).
