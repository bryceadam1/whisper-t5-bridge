# whisper-t5-bridge: results

## Round 3 (2026-11-09/10): task accuracy, Whisper-decoder bridges, FLAN-style tasks on real speech

**Bottom line.** Measured against the datasets' own answers (not agreement with flan-t5-small), the best
bridge, **R3**, reaches **66% of flan-t5-small's task score** on the tasks flan-t5-small itself can do,
against 89% for the Whisper-transcript route. It keeps most of flan-t5-small's ability on gist tasks
(sentiment 0.74 vs 0.86, question type 0.64 vs 0.82, concept coverage 0.67 vs 0.90) and fails where the
exact words of a long passage matter (lambada continuation 0.13 vs 0.72). It is about 2.5x faster than the
ASR route on CPU (75 vs 195 ms median per utterance). R3 is C2 continued on FLAN-style tasks over real
speech; it is in `release_v3/` (same architecture and layout as before; local only, not published).

### A better yardstick: task accuracy

Gold answers were recovered for all 861 eval items from the FLAN rows they came from, and each system is
scored per task (label accuracy for sentiment and question type, option accuracy for multiple choice,
next-word accuracy for lambada, concept coverage for common_gen, lenient match for trivia;
`scripts/task_accuracy.py`). Two families cannot separate systems because flan-t5-small itself fails them
even on text (closed-book trivia 0.01; multiple choice 0.16, level with no input at all), so the summary
("informative macro") averages the four families where the teacher beats the instruction-only floor by
at least 0.2. openbookqa (45 items) is not scored: its eval prompts are open-ended.

Held-out voices, the game's truncated window:

| system | closed_book_qa | concept_to_sentence | continuation | multiple_choice | question_type | sentiment | macro | **informative macro** | % of teacher |
|---|---|---|---|---|---|---|---|---|---|
| teacher (text) | 0.01 | 0.90 | 0.72 | 0.16 | 0.82 | 0.86 | 0.578 | **0.823** | 100% |
| asr_control | 0.01 | 0.87 | 0.40 | 0.16 | 0.80 | 0.85 | 0.516 | **0.731** | 89% |
| **R3 (released v3)** | 0.00 | 0.67 | 0.13 | 0.16 | 0.64 | 0.74 | 0.392 | **0.546** | 66% |
| C2 | 0.00 | 0.67 | 0.08 | 0.18 | 0.63 | 0.73 | 0.382 | **0.528** | 64% |
| R4: Q-Former 25/s | 0.00 | 0.52 | 0.00 | 0.18 | 0.60 | 0.55 | 0.308 | **0.417** | 51% |
| R1: Q-Former 6/s + alignment | 0.00 | 0.62 | 0.10 | 0.17 | 0.63 | 0.58 | 0.351 | **0.484** | 59% |
| R1b: Q-Former 6/s, no alignment | 0.00 | 0.61 | 0.05 | 0.15 | 0.68 | 0.65 | 0.359 | **0.499** | 61% |
| baseline | 0.01 | 0.60 | 0.00 | 0.16 | 0.65 | 0.67 | 0.349 | **0.480** | 58% |
| instr_only | 0.00 | 0.00 | 0.00 | 0.17 | 0.23 | 0.50 | 0.150 | **0.183** | 22% |

informative families (teacher beats instruction-only by >= 0.2): concept_to_sentence, continuation, question_type, sentiment

Agreement with the teacher, for comparison:

| run | held-out voices, truncated: EM / agree / KL | seen voices, truncated: EM / agree / KL | held-out voices, full 30 s: EM / agree / KL | seen voices, full 30 s: EM / agree / KL |
|---|---|---|---|---|
| **R3 (released v3)** | 0.358 / 0.770 / 0.699 | 0.368 / 0.774 / 0.701 | 0.286 / 0.681 / 0.843 | 0.289 / 0.684 / 0.855 |
| C2 | 0.373 / 0.772 / 0.721 | 0.371 / 0.776 / 0.722 | 0.290 / 0.675 / 0.939 | 0.288 / 0.676 / 0.954 |
| R4: Q-Former 25/s | 0.293 / 0.717 / 0.829 | 0.310 / 0.722 / 0.834 | 0.285 / 0.704 / 0.913 | 0.288 / 0.705 / 0.920 |
| R1: Q-Former 6/s + alignment | 0.335 / 0.736 / 0.782 | 0.328 / 0.734 / 0.787 | 0.292 / 0.705 / 0.853 | 0.292 / 0.704 / 0.853 |
| R1b: Q-Former 6/s, no alignment | 0.348 / 0.737 / 0.813 | 0.328 / 0.734 / 0.813 | 0.298 / 0.708 / 0.950 | 0.298 / 0.712 / 0.960 |
| baseline | 0.328 / 0.741 / 0.916 | 0.334 / 0.742 / 0.909 | 0.328 / 0.741 / 0.979 | 0.329 / 0.741 / 0.972 |
| *ASR control* | 0.586 / 0.848 / 0.358 | 0.573 / 0.840 / 0.366 | 0.614 / 0.860 / 0.347 | 0.599 / 0.858 / 0.341 |
| *instruction only* | 0.171 / 0.568 / 1.366 | 0.171 / 0.568 / 1.387 | 0.171 / 0.568 / 1.366 | 0.171 / 0.568 / 1.387 |

### What was tried

| run | change | outcome |
|---|---|---|
| R1 | bridge = learned queries through Whisper base.en's 6 decoder layers (initialised from it, after DiVA), 6 queries/s, + alignment loss (query *i* to T5's encoding of transcript token *i*) | underfit (train KL 0.98 vs 0.42 for the conv bridge); 59% of teacher |
| R1b | R1 without the alignment loss, full learning rate on the Whisper layers | still underfit; 61% |
| R4 | R1b at 25 queries/s (testing whether 6/s compressed too much) | worse; 51% |
| R2 | C2 + 29k Qwen-written tasks | crashed (data bug, fixed); folded into R3 |
| **R3** | C2 continued 10k steps adding (a) FLAN's own sentence templates (sentiment, keywords, continuation) on all 52k LibriSpeech utterances, ~100k prompts, and (b) ~59k QA / multiple-choice / true-false tasks written by Qwen2.5-1.5B-Instruct for 24k utterances (no Claude usage) | best on task score (66%) and KL; exact match slightly below C2 |

Findings:

1. **FLAN-style tasks on real speech help modestly** (task score 64% → 66%, KL 0.721 → 0.699, and
   clearly better on full 30 s windows: 43% → 48%). Exact-match agreement fell slightly (0.373 → 0.358):
   R3 agrees with the teacher's exact wording less often while getting tasks right a little more often.
2. **The Whisper-decoder Q-Former did not work in three configurations.** Its failure mode is
   underfitting, not overfitting, and neither removing the alignment loss nor raising the query rate
   fixed it. A likely cause is that Whisper's decoder layers were trained on token embeddings as inputs and
   are not easily re-purposed to read out constant queries; DiVA also had 3.5k h of speech and a 1.5B
   encoder. Not recommended.
3. **The remaining gap is lexical.** Every bridge does well where the gist decides the answer and badly
   where exact words of a long input matter (lambada). This is consistent across all three rounds.

### Cost

Round 3 used 4.54 GPU-h, $9.08, including ~0.7 h lost to connectivity: the network blocked outbound SSH
intermittently (SSH on port 443 was added as a fallback), and the last session ran ~0.5 h past its planned
stop while the local watchdog was suspended (the Mac slept) and the VM unreachable. **Total, all rounds:
12.00 GPU-h, $24.00 compute + ~$0.18 boot volume ≈ $24.2 of the $25 budget.** The instance is stopped,
not terminated (its disk holds all generated audio, features and teacher caches; ~$0.21/day).

### Recommendations

1. **Use `release_v3` (R3) for the game's speech path** for gist-level tasks: intent, sentiment, short
   classification, simple commands. It is fast and keeps ~75-85% of flan-t5-small's task ability there.
2. **Keep the ASR route for tasks that depend on exact words** (open questions, names, copying or
   completing text): it keeps 89% of flan-t5-small's ability, at ~2.5x the CPU cost.
3. **If you want to push the bridge further,** the evidence points to more and more varied *real* speech
   (Common Voice, CC0, many speakers and microphones) with FLAN-style prompts like R3's, rather than new
   architectures or losses. Each data step has helped; no recipe change has.

---

## Round 2 (2026-11-09): what moved the needle

**Bottom line.** Real recorded speech did; changing the objective or the injection point did not. Adding
184 h of LibriSpeech to the FLAN TTS data took the bridge from 0.328 to **0.373 exact match** with the
text-prompted teacher (held-out voices, the game's truncated window), token agreement from 0.741 to
0.772, and mean KL down 21% (0.916 → 0.721), all against a baseline retrained under the same protocol.
It is still well short of the ASR route (0.586), and on full 30 s windows it is slightly worse than the
baseline. The improved bridge (C2) has the same architecture and tensor layout as round 1's, so it is a
drop-in replacement; it is in `release_v2/` on your Mac and not published.

### What was tried

All runs: 3-layer bridge, 6000-12000 steps, time masking + state noise + dropout 0.2, checkpoints chosen
on a 3% slice of the *training* clips (round 1 chose them on eval_seen clips, which inflated its exact
match by about 0.025: the released round-1 bridge scores 0.353 under that bias, the same recipe 0.328
without it).

| run | change | why |
|---|---|---|
| baseline | round-1 recipe, retrained under the clean protocol | fair reference |
| A | + "write down exactly what was said" task (target: the true transcript) + CTC head on the bridge output (T5 vocabulary, weight 0.2) | force word-level content through |
| B | bridge output as soft tokens in T5's *input-embedding* space, placed where the input sat in the prompt, run through T5's frozen encoder with the instruction | the standard speech-LLM design; lets T5's encoder contextualise the audio |
| A + B | both | |
| C | baseline recipe + 100 h of LibriSpeech train-clean-100 (251 speakers), each utterance with 2 teacher prompts from generic templates (translation, summary, topic, question generation, continuation) | more and real speech, with long, word-dependent teacher outputs (mean 49 tokens vs 7) |
| C2 | C continued for 8000 steps with a further 83 h of train-clean-360 (209 more speakers) | more data |

### Results

Agreement with the teacher on held-out FLAN inputs (metrics as in round 1 below).

| run | held-out voices, truncated: EM / agree / KL | seen voices, truncated: EM / agree / KL | held-out voices, full 30 s: EM / agree / KL | seen voices, full 30 s: EM / agree / KL |
|---|---|---|---|---|
| **C2: + 184 h LibriSpeech (released v2)** | 0.373 / 0.772 / 0.721 | 0.371 / 0.776 / 0.722 | 0.290 / 0.675 / 0.939 | 0.288 / 0.676 / 0.954 |
| C: + 100 h LibriSpeech | 0.365 / 0.766 / 0.772 | 0.373 / 0.773 / 0.764 | 0.282 / 0.693 / 0.917 | 0.288 / 0.695 / 0.924 |
| baseline (same protocol) | 0.328 / 0.741 / 0.916 | 0.334 / 0.742 / 0.909 | 0.328 / 0.741 / 0.979 | 0.329 / 0.741 / 0.972 |
| A: lexical objectives | 0.305 / 0.716 / 0.867 | 0.296 / 0.713 / 0.855 | 0.277 / 0.681 / 1.015 | 0.268 / 0.679 / 1.011 |
| B: encoder-input injection | 0.316 / 0.711 / 0.870 | 0.320 / 0.707 / 0.887 | 0.223 / 0.665 / 1.002 | 0.223 / 0.661 / 1.016 |
| A + B | 0.311 / 0.719 / 0.809 | 0.318 / 0.719 / 0.810 | 0.200 / 0.637 / 1.149 | 0.195 / 0.630 / 1.184 |
| *ASR control* | 0.586 / 0.848 / 0.358 | 0.573 / 0.840 / 0.366 | 0.614 / 0.860 / 0.347 | 0.599 / 0.858 / 0.341 |
| *instruction only* | 0.171 / 0.568 / 1.366 | 0.171 / 0.568 / 1.387 | 0.171 / 0.568 / 1.366 | 0.171 / 0.568 / 1.387 |

**held-out voices, truncated**, exact match / mean KL by family

| family | n | **C2: + 184 h LibriSpeech (released v2)** | C: + 100 h LibriSpeech | baseline (same protocol) | A: lexical objectives | B: encoder-input injection | A + B | ASR control |
|---|---|---|---|---|---|---|---|---|
| closed_book_qa | 219 | 0.32 / 0.92 | 0.30 / 0.96 | 0.25 / 1.03 | 0.22 / 0.83 | 0.23 / 0.78 | 0.23 / 0.72 | 0.42 / 0.15 |
| concept_to_sentence | 106 | 0.05 / 0.54 | 0.06 / 0.58 | 0.04 / 0.63 | 0.05 / 0.72 | 0.00 / 0.84 | 0.01 / 0.65 | 0.37 / 0.21 |
| continuation | 92 | 0.12 / 2.12 | 0.10 / 2.33 | 0.00 / 3.59 | 0.01 / 3.47 | 0.00 / 3.33 | 0.01 / 3.60 | 0.05 / 3.71 |
| multiple_choice | 162 | 0.28 / 0.64 | 0.30 / 0.70 | 0.27 / 0.80 | 0.25 / 0.76 | 0.28 / 0.74 | 0.30 / 0.70 | 0.74 / 0.06 |
| question_type | 115 | 0.72 / 0.17 | 0.71 / 0.18 | 0.73 / 0.19 | 0.65 / 0.19 | 0.76 / 0.14 | 0.67 / 0.18 | 0.93 / 0.03 |
| sentiment | 107 | 0.79 / 0.11 | 0.76 / 0.11 | 0.72 / 0.15 | 0.70 / 0.16 | 0.65 / 0.17 | 0.67 / 0.14 | 0.98 / 0.01 |
| ALL | 801 | 0.37 / 0.72 | 0.36 / 0.77 | 0.33 / 0.92 | 0.30 / 0.87 | 0.32 / 0.87 | 0.31 / 0.81 | 0.59 / 0.36 |

### What it means

1. **Data is the lever.** C and C2 improve every metric on the game's window and nearly every task
   family (closed-book QA 0.25 → 0.32, sentiment 0.72 → 0.79, continuation 0.00 → 0.12). The second 83 h
   added less than the first 100 h (exact match +0.008, KL −0.05), so returns are diminishing *for this
   kind of data*.
2. **The lexical objectives and the encoder-input design lower KL but cost exact match** and token
   agreement, and they hurt full-window robustness. They make the student's distributions broader and
   closer to the teacher's on average, while it picks the teacher's top token less often. With 13k TTS
   clips neither paid off; B was also about 2x slower to train (T5's encoder runs on ~200-400 soft tokens).
   A run of B with LibriSpeech was started and stopped early: at that speed it would have taken ~2.5 h.
3. **Full 30 s windows got worse with LibriSpeech** (exact match 0.328 → 0.290). LibriSpeech utterances
   are long (mean ~13 s), which shifts what the bridge sees; for the game's truncated window this does
   not matter, but a bridge for full windows would need full-window training data.
4. **Multiple choice remains the weak spot** (0.28, no better than ignoring the input): the bridge
   still does not carry the exact question well enough to pick among options; the ASR route does (0.74).

### Cost

Round 2 used 3.87 h (07:10-11:02 UTC), **$7.74**: three failed launches on GPU memory (~0.15 h), the
A/B/A+B chain and a same-protocol baseline (~1.6 h), LibriSpeech preparation (~0.5 h, overlapped with
training), C and C2 with evaluations (~1.4 h), and a slow openslr.org download abandoned for the Hugging
Face mirror. CPU timing is unchanged from round 1 (same architecture).

### Recommendations after round 2

1. **Use `release_v2` for the game's speech path** where the gist is enough; the ASR route is still
   clearly better where exact words matter (questions, names, picking an option).
2. **Keep adding real speech, but more varied rather than more of the same.** The second LibriSpeech
   dose helped less than the first. Read audiobooks are clean and slow; Common Voice (CC0, many accents
   and microphones) and noise or room augmentation are closer to what a game microphone captures, and a
   few hours of in-domain game phrases would likely matter more than another 100 h of audiobooks.
3. **Give real speech content-dependent teacher tasks.** The LibriSpeech prompts were generic; spoken
   questions with QA or multiple-choice prompts would train exactly the families that lag.
4. **Do not pursue the lexical objectives or encoder-input injection further at this data scale.** Both
   were consistent losses on exact match; revisit B only with much more data (it is also slower on CPU,
   since T5's encoder then runs on the audio tokens).
5. **flan-t5-base is still not the next step;** about $5 of the $20 remains if you want one more
   experiment (e.g. Common Voice, ~$3).

---

## Round 1 (2026-11-08)

**Bottom line.** A 10.2M-parameter bridge trained for $7.18 of GPU time lets flan-t5-small act on
speech, and it is 2.5x faster than going through a Whisper transcript (73-77 ms vs 185-198 ms median per
utterance on CPU). But it carries the *gist* of what was said far better than the *words*: it matches
the text-prompted teacher's output exactly 35% of the time, against 59% for the plain ASR route and 17%
with no input at all. It is good enough for gist-level tasks (sentiment 0.76, question type 0.74 exact
match) and not for tasks that hinge on exact content (closed-book QA 0.32, multiple choice 0.27, which is
no better than ignoring the input).

### What was built

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

### Results

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

### CPU wall time

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

### Cost

| item | amount |
|---|---|
| VM.GPU.A10.1, 02:57-06:32 UTC on 2026-11-09 | 3.59 h × $2.00 = **$7.18** |
| boot volume, 150 GB | ~$0.21 per day while the instance exists (about $0.07 so far) |
| round 2 (see above) | 3.87 h × $2.00 = **$7.74** |
| **total, both rounds** | **7.46 h, $14.92 compute + ~$0.07 boot volume ≈ $15.00** of the $20 budget |

GPU time by phase (approx.): setup and TTS benchmarking 0.3 h; TTS 2.1 h (eval clips first, then
training clips until a 105-min cap, with teacher caching and eval features alongside); training
features 0.1 h; three training runs and three evaluations 0.6 h; about 0.35 h idle (waiting on checks
and an IP change). The TTS model ran at 3.3x real time out of the box; sizing its KV cache to each
batch (its code reads `config.max_seq_len` at call time) brought that to 13-14x.

The instance is **stopped, not terminated**: its disk still holds the 22 h of audio, the Whisper
states and the teacher cache, which would make a flan-t5-base run cheap. Terminate it to stop the
boot-volume charge.

### Deliverables

* **Round 3: `release_v3/`** (R3, best on task accuracy). **Round 2: `release_v2/`** (C2): `bridge.safetensors`, `bridge_spec.json`, `reference_io.safetensors`;
  same layout as round 1, better on the game's window. Local only, not committed or published.

* `release/bridge.safetensors`: the released bridge (stage 1 + aug), float32, 52 flat-named tensors,
  spec embedded in the metadata. **Not committed or published** (awaiting your go-ahead).
* `release/bridge_spec.json`: layer shapes, op order, GELU (erf), pre-norm LayerNorms with eps 1e-5,
  Whisper window and log-mel rules, and how the output joins T5.
* `release/reference_io.safetensors`: one Whisper state sequence and the bridge output for it, to check
  a candle port (PyTorch reloads reproduce it to within 1e-6).
* Stage-2 LoRA weights are in `runs/stage2/` (not recommended for use).

### Caveats

* Checkpoints were selected on a 576-pair slice of the eval_seen clips, which share inputs with the
  held-out-voice set. With 12 candidate checkpoints per run the bias is small, but a separate
  validation split would be cleaner next time.
* All speech is synthetic (one TTS model, LibriSpeech voices, clean). Real microphones, noise and
  spontaneous speech will be harder; expect lower numbers in the game.
* The QC step drops clips Whisper misrecognises, which removes the hardest clips (mostly 2-4 word
  keyword lists) from both training and evaluation.

### Recommendations

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
