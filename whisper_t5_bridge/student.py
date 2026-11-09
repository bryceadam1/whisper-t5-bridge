"""Student and teacher plumbing around frozen flan-t5-small and Whisper base.en.

Student: T5-encode the instruction text, bridge the Whisper states of the spoken input, concatenate
both along the sequence axis and hand the result to the frozen T5 decoder as its encoder states.
Teacher: flan-t5-small on the full text prompt.
"""

from __future__ import annotations

import contextlib
import math
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

T5_ID = "google/flan-t5-small"
WHISPER_ID = "openai/whisper-base.en"
MAX_NEW_TOKENS = 96


def load_t5(device, dtype=torch.float32):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(T5_ID)
    model = T5ForConditionalGeneration.from_pretrained(T5_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def load_whisper(device, dtype=torch.float32):
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return WhisperFeatureExtractor.from_pretrained(WHISPER_ID), WhisperTokenizer.from_pretrained(WHISPER_ID), model


# ---------------------------------------------------------------- LoRA on T5 cross-attention

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.rank, self.scale = base, rank, alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype) * self.scale


@contextlib.contextmanager
def lora_disabled(adapters: dict[str, "LoRALinear"]):
    """Run plain T5 (the teacher's model) with the adapters switched off."""
    for m in adapters.values():
        m.enabled = False
    try:
        yield
    finally:
        for m in adapters.values():
            m.enabled = True


def add_cross_attn_lora(t5, rank=8, alpha=16.0) -> dict[str, LoRALinear]:
    """Wrap q, k, v, o of every decoder cross-attention (EncDecAttention) with a LoRA adapter."""
    adapters = {}
    for i, block in enumerate(t5.decoder.block):
        attn = block.layer[1].EncDecAttention
        for name in ("q", "k", "v", "o"):
            lora = LoRALinear(getattr(attn, name), rank, alpha).to(t5.device)
            setattr(attn, name, lora)
            adapters[f"decoder.block.{i}.layer.1.EncDecAttention.{name}"] = lora
    return adapters


def lora_state(adapters: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in adapters.items():
        out[f"{name}.lora_A"] = m.A.detach().float().cpu().contiguous()
        out[f"{name}.lora_B"] = m.B.detach().float().cpu().contiguous()
    return out


# ---------------------------------------------------------------- forward passes

def encode_text(t5, ids, mask):
    return t5.encoder(input_ids=ids, attention_mask=mask).last_hidden_state


def student_encoder_states(t5, bridge, instr_ids, instr_mask, whisper_states, whisper_lens):
    """(B, Li + La, 512) encoder states and mask: [T5(instruction) ; bridge(audio)]."""
    with torch.no_grad():
        text = encode_text(t5, instr_ids, instr_mask)
    audio, audio_mask = bridge(whisper_states, whisper_lens)
    states = torch.cat([text, audio.to(text.dtype)], dim=1)
    return states, torch.cat([instr_mask.bool(), audio_mask], dim=1)


def decoder_logits(t5, enc_states, enc_mask, labels):
    """Teacher-forced decoder logits. labels: (B, L) target ids, padded with -100."""
    dec_in = t5._shift_right(labels.clamp_min(0).masked_fill(labels < 0, t5.config.pad_token_id))
    out = t5(encoder_outputs=BaseModelOutput(last_hidden_state=enc_states), attention_mask=enc_mask.long(),
             decoder_input_ids=dec_in)
    return out.logits


def kl_topk(student_logits, topk_ids, topk_logp, valid):
    """KL(teacher || student) on the teacher's top-k support (teacher renormalised over it).

    student_logits (B, L, V); topk_ids/topk_logp (B, L, K); valid (B, L) bool."""
    logp_s = F.log_softmax(student_logits.float(), dim=-1).gather(-1, topk_ids)
    logp_t = topk_logp.float() - torch.logsumexp(topk_logp.float(), dim=-1, keepdim=True)
    kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    return (kl * valid).sum() / valid.sum().clamp_min(1)


def kl_full(student_logits, teacher_logits, valid):
    """Exact KL(teacher || student) over the full vocabulary, per position: (B, L)."""
    lt = F.log_softmax(teacher_logits.float(), -1)
    ls = F.log_softmax(student_logits.float(), -1)
    return (lt.exp() * (lt - ls)).sum(-1) * valid


@torch.no_grad()
def greedy(t5, enc_states, enc_mask, max_new_tokens=MAX_NEW_TOKENS):
    out = t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc_states),
                      attention_mask=enc_mask.long(), max_new_tokens=max_new_tokens, do_sample=False,
                      num_beams=1)
    return out[:, 1:]  # drop the decoder start token


def compression_ratio(text: str) -> float:
    data = text.encode()
    return len(data) / max(1, len(zlib.compress(data)))


@torch.no_grad()
def _whisper_decode(whisper, tokenizer, enc_states, temperature, generator, max_new_tokens):
    dec, cfg = whisper.model.decoder, whisper.generation_config
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    nots = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eot = tokenizer.eos_token_id
    suppress = torch.tensor(list(cfg.suppress_tokens or []), device=enc_states.device, dtype=torch.long)
    begin_suppress = torch.tensor(list(cfg.begin_suppress_tokens or []), device=enc_states.device, dtype=torch.long)
    ids = torch.tensor([[sot, nots]], device=enc_states.device)
    past, out = None, []
    for step in range(max_new_tokens):
        o = dec(input_ids=ids if past is None else ids[:, -1:], encoder_hidden_states=enc_states,
                past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = whisper.proj_out(o.last_hidden_state[:, -1]).float()[0]
        logits[suppress] = -float("inf")
        if step == 0:
            logits[begin_suppress] = -float("inf")
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, -1)
            nxt = int(torch.multinomial(probs.cpu(), 1, generator=generator))
        if nxt == eot:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tokenizer.decode(out, skip_special_tokens=True).strip()


@torch.no_grad()
def whisper_transcribe(whisper, tokenizer, enc_states, max_new_tokens=128,
                       temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4):
    """Whisper decoding from precomputed encoder states of any window length (one utterance), following
    openai/whisper's defaults: suppress_tokens and begin_suppress_tokens from the generation config,
    greedy first, and temperature fallback while the output's gzip compression ratio exceeds 2.4
    (the repetition loops that short windows are prone to). Seeded, so results are reproducible.

    Prompt for an English-only model: <|startoftranscript|><|notimestamps|>."""
    generator = torch.Generator().manual_seed(0)
    for t in temperatures:
        text = _whisper_decode(whisper, tokenizer, enc_states, t, generator, max_new_tokens)
        if compression_ratio(text) <= compression_ratio_threshold:
            break
    return text
