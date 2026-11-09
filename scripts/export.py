"""Export a trained bridge for the candle port: weights (safetensors, flat names), bridge_spec.json, and a
reference input/output pair so a reimplementation can be checked numerically.

  release/bridge.safetensors      float32 weights; the spec is also embedded in the file metadata
  release/bridge_spec.json        shapes, op order, activation, norm placement, epsilon, window rules
  release/reference_io.safetensors   one Whisper state sequence and the bridge's output for it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from whisper_t5_bridge.audio import HOP, SAMPLE_RATE, SILENCE_SECONDS
from whisper_t5_bridge.bridge import load_bridge, save_bridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="release")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bridge = load_bridge(args.bridge).eval()
    spec = save_bridge(bridge, str(out / "bridge.safetensors"))
    spec["whisper_window"] = {
        "model": "openai/whisper-base.en encoder, including its final layer_norm",
        "samples": f"16 kHz mono audio followed by {SILENCE_SECONDS:g} s of zeros, then zero-padded to a "
                   f"multiple of {2 * HOP} samples (an even number of mel frames)",
        "log_mel": "openai/whisper log_mel_spectrogram: n_fft 400, hop 160, Hann window, centred (reflect "
                   "padding), last frame dropped, power, 80 mel bins, log10 clamp 1e-10, max(x, x.max() - 8), "
                   "(x + 4) / 4",
        "encoder": "conv1 + gelu, conv2 (stride 2) + gelu, add the first T_mel/2 rows of the positional "
                   "embedding, encoder layers, layer_norm -> (T_mel/2, 512)",
        "sample_rate": SAMPLE_RATE,
    }
    spec["t5"] = {
        "model": "google/flan-t5-small, unmodified",
        "use": "encoder_hidden_states = concat([t5_encoder(instruction_tokens), bridge(whisper_states)], seq "
               "axis); encoder attention mask all ones for both parts; decode greedily with the T5 decoder",
    }
    (out / "bridge_spec.json").write_text(json.dumps(spec, indent=1))

    # Reference pair from the first eval clip.
    meta = json.loads(open(Path(args.features) / "eval_heldout_meta.jsonl").readline())
    arr = np.load(Path(args.features) / "eval_heldout_truncated.npy", mmap_mode="r")
    states = torch.from_numpy(np.asarray(arr[meta["truncated_offset"]:meta["truncated_offset"] + meta["truncated_frames"]],
                                         dtype=np.float32))[None]
    with torch.no_grad():
        y, _ = bridge(states)
    save_file({"whisper_states": states[0].contiguous(), "bridge_output": y[0].contiguous()},
              str(out / "reference_io.safetensors"))
    reloaded = load_bridge(str(out / "bridge.safetensors")).eval()
    with torch.no_grad():
        assert torch.equal(reloaded(states)[0], y)
    print(f"exported to {out}: {sum(p.numel() for p in bridge.parameters()):,} parameters, "
          f"reference clip {meta['id']} ({states.shape[1]} -> {y.shape[1]} frames)")


if __name__ == "__main__":
    main()
