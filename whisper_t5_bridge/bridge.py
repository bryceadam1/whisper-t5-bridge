"""The bridge: Whisper base.en encoder states (50/s, width 512) -> flan-t5-small encoder space (width 512).

  conv:    Conv1d(512 -> 512, kernel 3, stride 2, padding 1), then GELU (erf)   T -> ceil(T / 2)
  layers:  N pre-norm transformer layers, each
             x = x + o(attn(q(n1(x)), k(n1(x)), v(n1(x))))     n1 = LayerNorm, 8 heads of 64
             x = x + down(gelu(up(n2(x))))                     n2 = LayerNorm, up 512 -> 2048
  final:   LayerNorm(512)

All LayerNorms have weight and bias and eps 1e-5. Attention is scaled dot-product with a key padding
mask only (no causal mask, no positional term: Whisper's states already carry absolute positions).
Tensor names are flat and match `spec()` so the module can be rewritten in candle.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass
class BridgeConfig:
    in_dim: int = 512
    d_model: int = 512
    n_layers: int = 3
    n_heads: int = 8
    ffn_dim: int = 2048
    conv_kernel: int = 3
    conv_stride: int = 2
    eps: float = 1e-5
    dropout: float = 0.1


class Layer(nn.Module):
    def __init__(self, c: BridgeConfig):
        super().__init__()
        self.n_heads = c.n_heads
        self.attn_norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.o = nn.Linear(c.d_model, c.d_model)
        self.ffn_norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.up = nn.Linear(c.d_model, c.ffn_dim)
        self.down = nn.Linear(c.ffn_dim, c.d_model)
        self.dropout = c.dropout

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, d = x.shape
        h = self.attn_norm(x)
        q, k, v = (proj(h).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        attn_mask = None if key_mask is None else key_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + F.dropout(self.o(a.transpose(1, 2).reshape(b, t, d)), self.dropout, self.training)
        h = self.down(F.gelu(self.up(self.ffn_norm(x))))
        return x + F.dropout(h, self.dropout, self.training)


class Bridge(nn.Module):
    def __init__(self, config: BridgeConfig | None = None):
        super().__init__()
        self.config = c = config or BridgeConfig()
        self.conv = nn.Conv1d(c.in_dim, c.d_model, c.conv_kernel, stride=c.conv_stride,
                              padding=c.conv_kernel // 2)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.n_layers))
        self.final_norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        c = self.config
        return (lengths + 2 * (c.conv_kernel // 2) - c.conv_kernel) // c.conv_stride + 1

    def forward(self, states: torch.Tensor, lengths: torch.Tensor | None = None):
        """states: (B, T, 512) Whisper encoder output, lengths: (B,) valid frames.
        Returns (B, T', 512) and a (B, T') bool mask of valid positions."""
        x = F.gelu(self.conv(states.transpose(1, 2))).transpose(1, 2)
        mask = None
        if lengths is not None:
            out_len = self.out_lengths(lengths)
            mask = torch.arange(x.shape[1], device=x.device)[None, :] < out_len[:, None]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return x, mask

    def spec(self) -> dict:
        c = self.config
        d, f = c.d_model, c.ffn_dim
        tensors = {"conv.weight": [d, c.in_dim, c.conv_kernel], "conv.bias": [d]}
        for i in range(c.n_layers):
            p = f"layers.{i}."
            for n in ("attn_norm", "ffn_norm"):
                tensors[p + n + ".weight"] = [d]
                tensors[p + n + ".bias"] = [d]
            for n in ("q", "k", "v", "o"):
                tensors[p + n + ".weight"] = [d, d]
                tensors[p + n + ".bias"] = [d]
            tensors[p + "up.weight"], tensors[p + "up.bias"] = [f, d], [f]
            tensors[p + "down.weight"], tensors[p + "down.bias"] = [d, f], [d]
        tensors["final_norm.weight"], tensors["final_norm.bias"] = [d], [d]
        return {
            "format": "whisper-t5-bridge/1",
            "config": {k: v for k, v in asdict(c).items() if k != "dropout"},
            "input": "Whisper base.en encoder output (after its final layer_norm), shape (T, 512), 50 states/s",
            "output": "states in flan-t5-small encoder space, shape (ceil(T/2), 512); concatenated with the "
                      "T5-encoded instruction along the sequence axis and passed to the T5 decoder as "
                      "encoder_hidden_states (order is irrelevant: T5 cross-attention has no position term)",
            "layout": "Linear weights are (out, in) and compute x @ W.T + b; Conv1d weight is (out, in, kernel)",
            "ops": [
                f"x = gelu_erf(conv1d(x, conv.weight, conv.bias, stride={c.conv_stride}, "
                f"padding={c.conv_kernel // 2}))   # over time, channels-first",
                f"for i in 0..{c.n_layers}: pre-norm block",
                "  h = layer_norm(x, attn_norm, eps); q,k,v = linear(h, q|k|v); "
                f"split {c.n_heads} heads of {d // c.n_heads}; a = softmax(q k^T / sqrt({d // c.n_heads})) v "
                "(padded keys masked); x = x + linear(merge(a), o)",
                "  x = x + linear(gelu_erf(linear(layer_norm(x, ffn_norm, eps), up)), down)",
                "x = layer_norm(x, final_norm, eps)",
            ],
            "activation": "GELU, exact erf form (not the tanh approximation)",
            "norm": f"LayerNorm with weight and bias, eps={c.eps}; pre-norm in every block, one final norm",
            "dtype": "float32",
            "tensors": tensors,
        }


def save_bridge(bridge: Bridge, path: str, extra_meta: dict | None = None):
    state = {k: v.detach().float().contiguous().cpu() for k, v in bridge.state_dict().items()}
    spec = bridge.spec()
    assert set(state) == set(spec["tensors"]), set(state) ^ set(spec["tensors"])
    meta = {"spec": json.dumps(spec)}
    if extra_meta:
        meta.update({k: json.dumps(v) for k, v in extra_meta.items()})
    save_file(state, path, metadata=meta)
    return spec


def load_bridge(path: str) -> Bridge:
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        spec = json.loads(f.metadata()["spec"])
    bridge = Bridge(BridgeConfig(**spec["config"]))
    bridge.load_state_dict(load_file(path))
    return bridge


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def init_output_scale(bridge: Bridge, t5_encoder_states: torch.Tensor, mask: torch.Tensor):
    """Start the final LayerNorm at the per-channel scale and offset of real T5 encoder outputs, so the
    frozen decoder sees inputs of a familiar magnitude from step 0."""
    x = t5_encoder_states[mask].float()
    with torch.no_grad():
        bridge.final_norm.weight.copy_(x.std(0).clamp_min(1e-3))
        bridge.final_norm.bias.copy_(x.mean(0))
    return math.sqrt(float((x ** 2).mean()))
