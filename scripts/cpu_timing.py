"""CPU wall time per utterance: the bridge path against the ASR text path.

Both paths share the Whisper encoder on the truncated window (timed separately).
  bridge path: bridge + T5 encoder on the instruction + T5 decoder (greedy)
  ASR path:    Whisper decoder (greedy transcript) + T5 encoder on the full prompt + T5 decoder (greedy)
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch

from whisper_t5_bridge.audio import LogMel, encode, window_samples
from whisper_t5_bridge.bridge import load_bridge
from whisper_t5_bridge.student import encode_text, encoder_input, greedy, load_t5, load_whisper, whisper_transcribe
from scripts.whisper_features import with_lead_in


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/audio")
    ap.add_argument("--teacher", default="data/teacher")
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--split", default="eval_seen")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    fe, wtok, whisper = load_whisper("cpu")
    tok, t5 = load_t5("cpu")
    bridge = load_bridge(args.bridge).eval()
    logmel = LogMel(fe.mel_filters)
    clips = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.audio) / "manifest.jsonl")
             if json.loads(l)["split"] == args.split}
    records = [r for r in torch.load(Path(args.teacher) / "eval.pt", weights_only=False) if r["id"] in clips]
    records = records[: args.n + 3]
    rows = []
    for k, r in enumerate(records):
        audio = with_lead_in(sf.read(str(Path(args.audio) / clips[r["id"]]["path"]), dtype="float32")[0], r["id"])
        states, t_enc = timed(lambda: encode(whisper.model.encoder, logmel(torch.from_numpy(window_samples(audio, "truncated")))[None]))

        def bridge_path():
            a, am = bridge(states)
            if bridge.config.injection == "encoder_input":
                before, _, after = r["prompt"].partition(r["input"])
                ids = lambda text: tok(text.strip(), add_special_tokens=False).input_ids
                x, mask = encoder_input(t5, [ids(before)], a, [a.shape[1]], [ids(after)])
                enc = t5.encoder(inputs_embeds=x, attention_mask=mask.long()).last_hidden_state
            else:
                ids = torch.tensor([r["instruction_ids"]])
                enc = torch.cat([encode_text(t5, ids, torch.ones_like(ids)), a], 1)
            return greedy(t5, enc, torch.ones(enc.shape[:2], dtype=torch.bool))

        (b_out, t_bridge_total) = timed(bridge_path)
        _, t_bridge_only = timed(lambda: bridge(states))
        transcript, t_wdec = timed(lambda: whisper_transcribe(whisper, wtok, states))
        prompt = r["prompt"].replace(r["input"], transcript, 1)
        enc_ids = tok([prompt], return_tensors="pt")
        p_states, t_t5enc = timed(lambda: encode_text(t5, enc_ids.input_ids, enc_ids.attention_mask))
        a_out, t_dec = timed(lambda: greedy(t5, p_states, enc_ids.attention_mask.bool()))
        if k < 3:
            continue  # warm-up
        rows.append({"audio_s": len(audio) / 16000, "whisper_encoder_ms": t_enc, "bridge_ms": t_bridge_only,
                     "bridge_path_ms": t_bridge_total, "whisper_decoder_ms": t_wdec, "t5_encoder_prompt_ms": t_t5enc,
                     "asr_t5_decoder_ms": t_dec, "asr_path_ms": t_wdec + t_t5enc + t_dec,
                     "bridge_out_tokens": int(b_out.shape[1]), "asr_out_tokens": int(a_out.shape[1])})
    summary = {"machine": f"{platform.machine()} {platform.processor()} {platform.platform()}",
               "torch": torch.__version__, "threads": args.threads, "n": len(rows)}
    for key in rows[0]:
        vals = [x[key] for x in rows]
        summary[key] = {"mean": statistics.mean(vals), "median": statistics.median(vals)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for key, v in summary.items():
        print(f"{key:24} {v['mean']:9.1f} mean {v['median']:9.1f} median" if isinstance(v, dict) else f"{key:24} {v}")


if __name__ == "__main__":
    main()
