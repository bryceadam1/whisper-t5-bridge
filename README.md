# whisper-t5-bridge

A small trained adapter (the "bridge", ~10M parameters) that lets `google/flan-t5-small` take speech as
input. Whisper base.en's encoder states go through the bridge and are concatenated with the T5-encoded
instruction text; the result is handed to flan-t5-small's frozen decoder as its encoder states.

The bridge is trained by behaviour-alignment distillation, after BLSP
([arXiv 2309.00916](https://arxiv.org/abs/2309.00916)) and BLSP-KD
([arXiv 2405.19041](https://arxiv.org/abs/2405.19041)): flan-t5-small reading the full *text* prompt is
the teacher, and the student, hearing the input instead of reading it, learns to match the teacher's
next-token distributions.

Status: pipeline built and smoke-tested locally; GPU run pending. Results go in [REPORT.md](REPORT.md).

## Method

```
instruction text ──► T5 encoder (frozen) ─────────────┐
                                                       ├─ concat (seq axis) ─► T5 decoder (frozen) ─► output
spoken input ──► Whisper base.en encoder (frozen) ─► bridge ─┘
```

* **Teacher**: flan-t5-small on the full prompt (instruction plus input); greedy output and top-16
  log-probabilities per output token are cached.
* **Student**: as above. The order of the two parts along the sequence axis does not matter: T5's
  cross-attention has no position term.
* **Loss**: KL(teacher ‖ student) on the teacher's top-k support along the teacher's output, plus 0.2 ×
  cross-entropy on the teacher's greedy tokens.
* **What trains**: stage 1, only the bridge. Stage 2 (reported separately), LoRA (rank 8) on the
  decoder's cross-attention q, k, v, o, starting from the stage-1 bridge.

## Data

**FLAN 2021 subset** from [`DataProvenanceInitiative/flan2021_submix_original`](https://huggingface.co/datasets/DataProvenanceInitiative/flan2021_submix_original)
(revision `416c59f5`), zero-shot rows only. The collection's code is Apache-2.0, but each underlying
dataset keeps its own licence, so tasks were chosen with the Data Provenance Initiative's per-dataset
licence audit, keeping only permissive licences and prompts with a single speakable input:

| task | licence (per DPI) | family | input field |
|---|---|---|---|
| sst2 | CC0 1.0 | sentiment | `sentence` |
| trec | CC0 1.0 | question_type | `text` |
| trivia_qa | Apache 2.0 | closed_book_qa | `question` |
| natural_questions | CC BY-SA 3.0 | closed_book_qa | `question` |
| common_gen | MIT | concept_to_sentence | `concepts` |
| arc (easy, challenge) | CC BY-SA 4.0 | multiple_choice | `question` (options stay in text) |
| openbookqa | Apache 2.0 | multiple_choice | `question` |
| lambada | CC BY 4.0 | continuation | `sentence` |

Left out: anything NC, ND, "Custom" or "Unspecified"; translation (non-English); gigaword and
cnn_dailymail (their listed MIT licence covers loader code, not the text); and **fix_punct / true_case**
(CC0, but their web-scraped text carries adult-site spam that keyword filtering could not clean).

Each row is matched against FLAN's own templates ([google-research/FLAN](https://github.com/google-research/FLAN),
Apache-2.0, commit `e9e4ec6`, parsed with `ast.literal_eval`, never executed) to recover the input
field exactly. The **instruction** is the prompt with the input removed. Inputs are kept if they are
2-50 words of plain English text with few digits and no explicit or spam terms. Each training input is
rendered with up to 3 different templates, so one recording serves several prompts.

**Voices**: 40 LibriSpeech speakers (CC BY 4.0; readings of public-domain LibriVox recordings), 32 for
training (dev-clean) and 8 held out (test-clean), balanced by sex. Each speaker contributes a slow,
medium and fast 6-12 s reference clip with its exact transcript.

**Speech**: [`Edge0/Audio8-TTS-Realistic-Air`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air)
(Apache-2.0), revision `23c305cc`, run only on the cloud GPU. Its custom code needs
`trust_remote_code=True`; it was read before use (no network, subprocess or exec; `codec.pth` is
loaded with `torch.load(weights_only=True)`). Speech is cloned from a reference clip, then given a WSOLA
tempo change in [0.9, 1.15], resampled to 16 kHz mono, and checked with Whisper: clips with WER > 0.3 or
longer than 25 s are dropped.

## Whisper windows

The planned consumer (a game mod on candle) runs Whisper's encoder on the audio plus 3 s (300 mel
frames) of silence, rounded up to an even frame count, not on the standard 30 s window (cf.
whisper.cpp's `audio_ctx`). Training and the main evaluation use exactly that truncated window; results
on full 30 s windows are reported too. See [`whisper_t5_bridge/audio.py`](whisper_t5_bridge/audio.py);
its log-mel and encoder pass match transformers' reference implementation bit for bit on 30 s windows.

## Bridge layout

`Conv1d(512→512, k=3, stride 2, pad 1) + GELU`, then 3 pre-norm transformer layers (8 heads, FFN 2048,
GELU), then `LayerNorm`; all LayerNorms have weight and bias with eps 1e-5. Weights are saved as
safetensors with flat names; the full spec (shapes, op order, activation, norm placement, epsilon) is
embedded in the file's metadata and written to `bridge_spec.json`. See
[`whisper_t5_bridge/bridge.py`](whisper_t5_bridge/bridge.py).

## Pipeline

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt

# local (free)
python -m scripts.prepare_flan --parquet data/raw/flan2021/train-00000.parquet
python -m scripts.select_voices

# cloud GPU
python -m scripts.tts_generate --benchmark
python -m scripts.tts_generate --split train --limit N
python -m scripts.tts_generate --split eval_seen && python -m scripts.tts_generate --split eval_heldout
python -m scripts.teacher_cache --split train && python -m scripts.teacher_cache --split eval
python -m scripts.whisper_features --split train      # and eval_seen, eval_heldout
python -m scripts.train --out runs/stage1
python -m scripts.train --out runs/stage2 --init-bridge runs/stage1/bridge.safetensors --lora-rank 8
python -m scripts.evaluate --bridge runs/stage1/bridge.safetensors --out results/stage1

# local CPU timing
python -m scripts.cpu_timing --bridge runs/stage1/bridge.safetensors --out results/cpu_timing.json
```

`scripts/smoke_say_tts.py` voices a few items with macOS `say` to exercise the pipeline on a laptop; that
audio is never used for training or results.

## Compute

One Oracle Cloud `VM.GPU.A10.1` ($2.00/h) in a dedicated compartment, managed with
[`ops/oci.sh`](ops/oci.sh); every start and stop is logged in [`ops/usage_log.csv`](ops/usage_log.csv)
for a running total. Budget: $15 in total.
