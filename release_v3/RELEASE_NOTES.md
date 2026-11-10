# whisper-t5-bridge v3: speech in, text instruction, text out with flan-t5-small

A 10.2M-parameter adapter ("bridge") that lets `google/flan-t5-small` act on **speech**: Whisper
base.en encodes the audio, the bridge maps it into flan-t5-small's encoder space, and flan-t5-small's
unmodified decoder answers a text instruction about it. Trained by distillation from flan-t5-small
reading the text (round 3, run "R3"; see [REPORT.md](https://github.com/bryceadam1/whisper-t5-bridge/blob/main/REPORT.md)).

## How well it does

Task accuracy against the datasets' own answers, on held-out FLAN 2021 items spoken by held-out voices,
averaged over the task families flan-t5-small itself can do (sentiment, question type, concept coverage,
continuation):

| system | task score | share of flan-t5-small | CPU per utterance (median, M1 Pro) |
|---|---|---|---|
| flan-t5-small reading the text | 0.823 | 100% | – |
| Whisper base.en transcript → flan-t5-small | 0.731 | 89% | 195 ms |
| **this bridge** | **0.546** | **66%** | **75 ms** |

It keeps ~75-85% of flan-t5-small's ability where the gist decides the answer (sentiment 0.74 vs 0.86,
question type 0.64 vs 0.82, concept coverage 0.67 vs 0.90) and is weak where the exact words of a long
input matter (predicting a passage's last word: 0.13 vs 0.72). Use it for intent, sentiment, short
classification and simple commands; use the transcript route where exact words matter.

## How to use it

1. **Audio window.** 16 kHz mono audio, followed by 3 s of silence, zero-padded to a multiple of 320
   samples. Compute openai/whisper's log-mel (n_fft 400, hop 160, 80 mel bins) over exactly that window.
2. **Whisper base.en encoder** on that window, using the first T_mel/2 positional embeddings
   (whisper.cpp's audio_ctx idea) → states (T, 512).
3. **Bridge** (this file) → (ceil(T/2), 512).
4. **flan-t5-small**: encoder_hidden_states = concat(T5-encoder(instruction text), bridge output) along
   the sequence axis (order does not matter), attention mask all ones; decode greedily.

`bridge_spec.json` gives every tensor shape, the op order, the activation (exact-erf GELU), pre-norm
LayerNorm placement and epsilon (1e-5). `reference_io.safetensors` holds one Whisper state sequence and
the bridge output for it, to check a reimplementation (e.g. in candle) numerically.

## Files

| file | sha256 |
|---|---|
| `bridge.safetensors` (float32, 52 tensors) | `97ed3abf22678b8cab2e8cf665a99cfbabf5785a8434c9466afbd1698398aeb2` |
| `bridge_spec.json` | `e744fd2b62b494d50e4572b6f4e189b11b35948690978eaed4e3b4fa69ec036b` |
| `reference_io.safetensors` | `044d160c35432c49173c3c90b40dd4ee29818a65f94e4ff4adc85b0395e55d7b` |

## Training data and licences

- **Teacher:** google/flan-t5-small (Apache-2.0); student encoder: openai/whisper-base.en (MIT).
- **Synthetic speech:** FLAN 2021 inputs from tasks with permissive licences, as audited by the Data
  Provenance Initiative: sst2, trec (CC0); trivia_qa, openbookqa (Apache-2.0); common_gen (MIT);
  lambada (CC BY 4.0); natural_questions (CC BY-SA 3.0); arc (CC BY-SA 4.0). Voiced with
  Edge0/Audio8-TTS-Realistic-Air (Apache-2.0), cloning LibriSpeech speakers.
- **Real speech:** LibriSpeech train-clean-100 and part of train-clean-360 (CC BY 4.0; Panayotov et al.,
  2015, from public-domain LibriVox recordings), with prompts from FLAN templates and tasks written by
  Qwen2.5-1.5B-Instruct (Apache-2.0).

**Licence of the weights:** Apache-2.0, like the code, with attribution to the sources above. Two of
them (natural_questions, arc) are CC BY-SA; whether share-alike terms reach weights trained on such data
is legally unsettled.

## Limitations

- Trained and evaluated on clean speech (synthetic and audiobook); expect lower accuracy on noisy,
  far-field or spontaneous speech.
- English only. Inherits flan-t5-small's knowledge limits: it cannot answer trivia that flan-t5-small
  cannot answer from text.
- Tuned for the truncated window above; on standard 30 s windows its task score drops to ~48% of
  flan-t5-small's.
