# whisper-t5-bridge

A trained adapter ("bridge") that lets `google/flan-t5-small` take speech as input: Whisper base.en's
encoder states go through the bridge into flan-t5-small's decoder, alongside the T5-encoded instruction text.
Trained by behaviour-alignment distillation (after BLSP, arXiv 2309.00916, and BLSP-KD, arXiv 2405.19041),
with flan-t5-small reading the full text prompt as the teacher.

Work in progress.
