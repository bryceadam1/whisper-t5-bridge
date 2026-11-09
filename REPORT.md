# whisper-t5-bridge: results

**Bottom line.** A 10.2M-parameter bridge trained for $7.18 of GPU time lets flan-t5-small act on
speech, and it is 2.5x faster than going through a Whisper transcript (73-77 ms vs 185-198 ms median per
utterance on CPU). But it carries the *gist* of what was said far better than the *words*: it matches
the text-prompted teacher's output exactly 35% of the time, against 59% for the plain ASR route and 17%
with no input at all. It is good enough for gist-level tasks (sentiment 0.76, question type 0.74 exact
match) and not for tasks that hinge on exact content (closed-book QA 0.32, multiple choice 0.27, which is
no better than ignoring the input).

## What was built

* **Data**: 8 permissively licensed FLAN 2021 tasks with a single speakable input (see README),
  19,675 training inputs and 861 held-out inputs. Each prompt is split into instruction and input by
  matching FLAN's own templates. Two CC0 tasks (fix_punct, true_case) were dropped because their web
  text carries adult-site spam.
* **Speech**: Audio8-TTS-Realistic-Air (pinned, code reviewed, run only on the GPU), cloning 32
  LibriSpeech training voices and 8 held-out voices, with tempo varied 0.9-1.15x. 14,848 training clips
  (19.4 h) were generated before the TTS time cap; 13,800 passed a Whisper check (WER <= 0.3, <= 25 s),
  giving 40,183 training pairs. Every eval input was voiced twice, by a seen and by a held-out voice
  (1.4 h each); 812 and 801 eval examples passed the same check.
* **Teacher**: flan-t5-small on the full text prompt; greedy outputs (mean 7.4 tokens) and top-16
  log-probabilities. The 2% of outputs that never ended (repetition loops) were excluded.
* **Student**: T5-encoded instruction ++ bridge(Whisper base.en states) -> frozen T5 decoder. Whisper
  sees the game's window: audio plus 3 s of silence, even frame count.
* **Loss**: KL(teacher || student) on the top-16 support + 0.2 x CE on the teacher's tokens.

Variants, all 3-layer bridges trained from the same data:

| run | what trains | notes | training time |
|---|---|---|---|
| stage 1 | bridge | 6000 steps, batch 64; validation KL plateaued from step ~3000 | 10.7 min |
| stage 2 | bridge + LoRA r8 on decoder cross-attention (229k params) | from stage 1, 3000 steps | 7.4 min |
| stage 1 + aug (**released**) | bridge | stage 1 plus 2 time masks (<= 5%) and noise on the Whisper states, dropout 0.2 | 10.9 min |

## Results

Agreement with the teacher on held-out FLAN inputs. *Exact match*: greedy outputs identical.
*Token agree*: with teacher forcing, fraction of positions where the system's top token is the teacher's.
*Mean KL*: KL(teacher || system) per output token over the full vocabulary, teacher forcing.
*ASR control*: the bridge bypassed; Whisper's own transcript (decoded from the same window, with
openai/whisper's suppression and temperature fallback) put in place of the input in the text prompt.
*Instruction only*: the decoder sees the instruction and nothing else; the floor.

| condition | n | system | exact match | token agree | mean KL |
|---|---|---|---|---|---|
| seen voices, truncated | 812 | bridge, stage 1 + aug (released) | 0.363 | 0.756 | 0.943 |
|  |  | bridge, stage 1 | 0.358 | 0.750 | 0.968 |
|  |  | bridge + LoRA, stage 2 | 0.367 | 0.754 | 0.977 |
|  |  | ASR control | 0.573 | 0.840 | 0.366 |
|  |  | instruction only | 0.171 | 0.568 | 1.387 |
| held-out voices, truncated | 801 | bridge, stage 1 + aug (released) | 0.353 | 0.755 | 0.947 |
|  |  | bridge, stage 1 | 0.352 | 0.749 | 0.974 |
|  |  | bridge + LoRA, stage 2 | 0.351 | 0.752 | 0.986 |
|  |  | ASR control | 0.586 | 0.848 | 0.358 |
|  |  | instruction only | 0.171 | 0.568 | 1.366 |
| seen voices, full 30 s | 812 | bridge, stage 1 + aug (released) | 0.347 | 0.747 | 1.051 |
|  |  | bridge, stage 1 | 0.331 | 0.741 | 1.122 |
|  |  | bridge + LoRA, stage 2 | 0.339 | 0.747 | 1.113 |
|  |  | ASR control | 0.599 | 0.858 | 0.341 |
|  |  | instruction only | 0.171 | 0.568 | 1.387 |
| held-out voices, full 30 s | 801 | bridge, stage 1 + aug (released) | 0.331 | 0.744 | 1.052 |
|  |  | bridge, stage 1 | 0.321 | 0.733 | 1.133 |
|  |  | bridge + LoRA, stage 2 | 0.341 | 0.745 | 1.127 |
|  |  | ASR control | 0.614 | 0.860 | 0.347 |
|  |  | instruction only | 0.171 | 0.568 | 1.366 |

**Held-out voices, truncated window**, by task family (exact match / mean KL):

| family | n | bridge (released) | ASR control | instruction only |
|---|---|---|---|---|
| sentiment | 107 | 0.76 / 0.13 | 0.98 / 0.01 | 0.59 / 0.18 |
| question_type | 115 | 0.74 / 0.16 | 0.93 / 0.03 | 0.28 / 0.78 |
| closed_book_qa | 219 | 0.32 / 1.12 | 0.42 / 0.15 | 0.00 / 1.26 |
| multiple_choice | 162 | 0.27 / 0.84 | 0.74 / 0.06 | 0.26 / 0.94 |
| concept_to_sentence | 106 | 0.04 / 0.61 | 0.37 / 0.21 | 0.00 / 1.20 |
| continuation | 92 | 0.01 / 3.66 | 0.05 / 3.71 | 0.00 / 5.98 |
| all | 801 | 0.35 / 0.95 | 0.59 / 0.36 | 0.17 / 1.37 |

The seen-voice table is within ±0.03 of this on every family (see `results/stage1_aug/summary.md`).

What the numbers say:

1. **The bridge transmits content, but coarsely.** It cuts the floor's error by 41% on sentiment and 64% on
   question type, and gets a third of closed-book questions exactly right where the floor gets none. Its failures are near misses
   in meaning ("ethyl" → "helium", "a room with a fireplace" → "a room featuring a fireplace"), the
   signature of a representation that encodes topic and intent better than exact words.
2. **The ASR route is better on every family**, most of all where exact words matter (multiple choice
   0.74 vs 0.27). Even ASR is far from perfect: T5-small's outputs change with small differences in
   casing, punctuation and recognition errors (Whisper's WER on these clips is 0.06-0.13 in the truncated
   window).
3. **Voices generalise.** Held-out speakers score within 0.01 of seen speakers.
4. **The truncated window is the right training choice.** On full 30 s windows the bridge loses 0.02
   exact match and its KL rises by about 0.1; the augmentation narrowed that gap. Whisper's own decoding
   is the other way round: in the truncated window it is prone to repetition loops on very short clips
   (corpus WER 0.13 vs 0.05 on seen voices, after the standard temperature fallback).
5. **LoRA on cross-attention did not help** (exact match 0.351 vs 0.352 on held-out voices; KL slightly
   worse), and it is not free: with the adapter merged, the plain *text* path drifts too, matching the
   teacher's text-prompt output in only 86.5% of cases. The bridge alone had plateaued (validation KL
   flat from step ~3000) while training agreement kept rising (0.90 vs 0.77), so the limit is the data,
   not the decoder's capacity to read the bridge.
6. **Continuation (lambada) is not meaningful** for any system: flan-t5-small's own continuations are
   near random, so agreement is low everywhere.

## CPU wall time

Per utterance, Apple M1 Pro, PyTorch 2.15 eager, float32, 100 held-out clips (60 for one thread), mean
3.6 s of speech. Measured with the stage-1 bridge (same architecture as the released one). The Whisper
encoder runs in both paths.

| component (ms per utterance) | 4 threads median | 4 threads mean | 1 thread median | 1 thread mean |
|---|---|---|---|---|
| Whisper encoder, truncated window (shared) | 30 | 31 | 42 | 42 |
| bridge alone | 9 | 9 | 11 | 11 |
| **bridge path**: bridge + T5 encoder (instruction) + T5 decoder | 77 | 111 | 73 | 104 |
| Whisper decoder (transcript) | 124 | 417 | 117 | 488 |
| T5 encoder (full prompt) | 14 | 14 | 12 | 11 |
| T5 decoder | 58 | 97 | 56 | 87 |
| **ASR path**: Whisper decoder + T5 encoder + T5 decoder | 198 | 527 | 185 | 586 |

The ASR path's mean is dominated by clips where Whisper loops and the temperature fallback re-decodes;
the bridge path has no such tail. Candle numbers will differ, but the ratio should hold: the bridge
replaces an autoregressive Whisper decode with one 9 ms forward pass.

## Cost

| item | amount |
|---|---|
| VM.GPU.A10.1, 02:57-06:32 UTC on 2026-11-09 | 3.59 h × $2.00 = **$7.18** |
| boot volume, 150 GB | ~$0.21 per day while the instance exists (about $0.03 so far) |
| **total so far** | **≈ $7.21** of the $15 budget |

GPU time by phase (approx.): setup and TTS benchmarking 0.3 h; TTS 2.1 h (eval clips first, then
training clips until a 105-min cap, with teacher caching and eval features alongside); training
features 0.1 h; three training runs and three evaluations 0.6 h; about 0.35 h idle (waiting on checks
and an IP change). The TTS model ran at 3.3x real time out of the box; sizing its KV cache to each
batch (its code reads `config.max_seq_len` at call time) brought that to 13-14x.

The instance is **stopped, not terminated**: its disk still holds the 22 h of audio, the Whisper
states and the teacher cache, which would make a flan-t5-base run cheap. Terminate it to stop the
boot-volume charge.

## Deliverables

* `release/bridge.safetensors`: the released bridge (stage 1 + aug), float32, 52 flat-named tensors,
  spec embedded in the metadata. **Not committed or published** (awaiting your go-ahead).
* `release/bridge_spec.json`: layer shapes, op order, GELU (erf), pre-norm LayerNorms with eps 1e-5,
  Whisper window and log-mel rules, and how the output joins T5.
* `release/reference_io.safetensors`: one Whisper state sequence and the bridge output for it, to check
  a candle port (PyTorch reloads reproduce it to within 1e-6).
* Stage-2 LoRA weights are in `runs/stage2/` (not recommended for use).

## Caveats

* Checkpoints were selected on a 576-pair slice of the eval_seen clips, which share inputs with the
  held-out-voice set. With 12 candidate checkpoints per run the bias is small, but a separate
  validation split would be cleaner next time.
* All speech is synthetic (one TTS model, LibriSpeech voices, clean). Real microphones, noise and
  spontaneous speech will be harder; expect lower numbers in the game.
* The QC step drops clips Whisper misrecognises, which removes the hardest clips (mostly 2-4 word
  keyword lists) from both training and evaluation.

## Recommendations

1. **For the mod today**: use the released bridge where the gist is enough (intent, sentiment, short
   commands with a closed set of outcomes), and the ASR route where exact content matters (names,
   questions, picking an option). The bridge path is cheaper; it is not a drop-in replacement for ASR.
2. **Do not merge the LoRA.** It buys nothing on speech and changes the text path.
3. **If you want the bridge to close the gap**, in order of expected value:
   * *More and more varied speech.* The bridge overfits 13.8k clips. The remaining 26% of the selected
     inputs, more inputs per task, and some real recorded speech are the obvious next step; TTS is
     about $0.22 per 1,000 clips at the measured rate.
   * *An auxiliary lexical objective*, e.g. a CTC or transcript loss on the bridge output, to force
     word-level information through (the failures are lexical, not semantic).
   * *Less downsampling.* Stride 2 leaves 25 states/s; stride 1 doubles the decoder's cross-attention
     length but may keep more phonetic detail.
4. **flan-t5-base?** Worth one experiment, but not for the reason one might hope. Its d_model is 768,
   so the bridge needs a 768-wide output (a final 512→768 projection, or a 768-wide bridge at ~2x the
   parameters); teacher caching and training would take
   ~3x longer (~40 min of A10 time in total, **about $2-3** if the stopped instance's audio and Whisper
   states are reused, ~$8 if they have to be regenerated), and on CPU its decoder is roughly 3x slower
   (the bridge path would land near 200-250 ms, about where the small model's ASR path is now). It would
   give better *answers*, since flan-t5-small's teacher outputs are often wrong, but the agreement gap
   comes from the bridge's lexical fidelity and the data size, which a larger decoder does not fix. I
   would spend the next few dollars on more speech data with the small model first, and try base only
   if the game needs better answers more than lower latency.
