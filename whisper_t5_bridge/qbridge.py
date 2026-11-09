"""A Q-Former bridge initialised from Whisper base.en's decoder (after DiVA, arXiv 2410.02678).

Learned query vectors (one shared vector plus Whisper's decoder positional embeddings) pass through the 6
Whisper decoder layers, which cross-attend to the Whisper encoder states, then a LayerNorm, a linear map
and a LayerNorm into flan-t5-small's encoder-output space. Unlike Whisper's decoding, the queries are read
out in one pass: self-attention is bidirectional and there are no token embeddings.

  n_queries = ceil(T_enc * queries_per_second / 50)          T_enc = Whisper encoder frames (50/s)
  x = query + pos[:n_queries]
  per layer (pre-norm, same weights and names as WhisperDecoderLayer):
    x = x + self_attn(self_attn_layer_norm(x))                bidirectional, padded queries masked
    x = x + encoder_attn(encoder_attn_layer_norm(x), whisper_states)
    x = x + fc2(gelu(fc1(final_layer_norm(x))))
  out = final_norm(proj(decoder_norm(x)))

Attention follows Whisper: q scaled by head_dim^-0.5, k_proj has no bias, GELU is the exact (erf) form,
LayerNorm eps 1e-5.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

FORMAT = "whisper-t5-bridge/qformer-1"
ENCODER_FPS = 50


@dataclass
class QBridgeConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048
    queries_per_second: float = 6.0
    max_queries: int = 448
    eps: float = 1e-5
    dropout: float = 0.1
    injection: str = "decoder"


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, kv, key_mask, dropout):
        b, n, d = x.shape
        split = lambda t: t.view(b, -1, self.n_heads, d // self.n_heads).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(kv)), split(self.v_proj(kv))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :], dropout_p=dropout)
        return self.out_proj(a.transpose(1, 2).reshape(b, n, d))


class QLayer(nn.Module):
    def __init__(self, c: QBridgeConfig):
        super().__init__()
        d = c.d_model
        self.self_attn = Attention(d, c.n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=c.eps)
        self.encoder_attn = Attention(d, c.n_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d, eps=c.eps)
        self.fc1 = nn.Linear(d, c.ffn_dim)
        self.fc2 = nn.Linear(c.ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=c.eps)
        self.dropout = c.dropout

    def forward(self, x, q_mask, states, s_mask):
        p = self.dropout if self.training else 0.0
        h = self.self_attn_layer_norm(x)
        x = x + F.dropout(self.self_attn(h, h, q_mask, p), p, self.training)
        x = x + F.dropout(self.encoder_attn(self.encoder_attn_layer_norm(x), states, s_mask, p), p, self.training)
        return x + F.dropout(self.fc2(F.gelu(self.fc1(self.final_layer_norm(x)))), p, self.training)


class QBridge(nn.Module):
    def __init__(self, config: QBridgeConfig | None = None):
        super().__init__()
        self.config = c = config or QBridgeConfig()
        self.query = nn.Parameter(torch.zeros(c.d_model))
        self.pos = nn.Parameter(torch.zeros(c.max_queries, c.d_model))
        self.layers = nn.ModuleList(QLayer(c) for _ in range(c.n_layers))
        self.decoder_norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.final_norm = nn.LayerNorm(c.d_model, eps=c.eps)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(c.d_model))
            self.proj.bias.zero_()

    @classmethod
    def from_whisper(cls, whisper, config: QBridgeConfig | None = None) -> "QBridge":
        """Initialise layers, positions and the decoder norm from a transformers Whisper model."""
        bridge = cls(config)
        dec = whisper.model.decoder
        state = {f"layers.{k}": v for k, v in dec.layers.state_dict().items()}
        state["decoder_norm.weight"], state["decoder_norm.bias"] = dec.layer_norm.weight, dec.layer_norm.bias
        state["pos"] = dec.embed_positions.weight[: bridge.config.max_queries]
        missing, unexpected = bridge.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) == {"query", "proj.weight", "proj.bias", "final_norm.weight",
                                                    "final_norm.bias"}, (missing, unexpected)
        return bridge

    def n_queries(self, lengths: torch.Tensor) -> torch.Tensor:
        c = self.config
        return torch.ceil(lengths.float() * c.queries_per_second / ENCODER_FPS).long().clamp(1, c.max_queries)

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.n_queries(lengths)

    def forward(self, states: torch.Tensor, lengths: torch.Tensor | None = None):
        """states (B, T, 512) Whisper encoder output; lengths (B,) valid frames.
        Returns (B, Nq, 512) and a (B, Nq) bool mask of valid queries."""
        b, t, _ = states.shape
        if lengths is None:
            lengths = torch.full((b,), t, device=states.device)
        nq = self.n_queries(lengths)
        n = int(nq.max())
        q_mask = torch.arange(n, device=states.device)[None, :] < nq[:, None]
        s_mask = torch.arange(t, device=states.device)[None, :] < lengths[:, None]
        x = (self.query + self.pos[:n]).expand(b, n, -1).to(states.dtype)
        for layer in self.layers:
            x = layer(x, q_mask, states, s_mask)
        return self.final_norm(self.proj(self.decoder_norm(x))), q_mask

    def new_parameters(self):
        """Parameters not taken from Whisper (trained at a higher learning rate)."""
        return [self.query, *self.proj.parameters(), *self.final_norm.parameters()]

    def spec(self) -> dict:
        c = self.config
        return {
            "format": FORMAT,
            "config": {k: v for k, v in asdict(c).items() if k != "dropout"},
            "input": "Whisper base.en encoder output (after its final layer_norm), shape (T, 512), 50 states/s",
            "output": "shape (n_queries, 512), n_queries = ceil(T * queries_per_second / 50), in flan-t5-small "
                      "encoder-output space: concatenate with the T5-encoded instruction along the sequence axis "
                      "and pass to the T5 decoder as encoder_hidden_states (order is irrelevant: T5 "
                      "cross-attention has no position term)",
            "layout": "Linear weights are (out, in) and compute x @ W.T + b; layer tensors use the same names and "
                      "shapes as transformers' WhisperDecoderLayer (model.decoder.layers.*)",
            "ops": __doc__.split("\n\n", 2)[1].strip().splitlines(),
            "attention": f"{c.n_heads} heads of {c.d_model // c.n_heads}; q scaled by head_dim^-0.5; k_proj has no "
                         "bias; self-attention is bidirectional over valid queries; cross-attention masks padded "
                         "encoder frames",
            "activation": "GELU, exact erf form",
            "norm": f"LayerNorm with weight and bias, eps={c.eps}; pre-norm in each layer",
            "dtype": "float32",
            "tensors": {k: list(v.shape) for k, v in self.state_dict().items()},
        }


def save_qbridge(bridge: QBridge, path: str, extra_meta: dict | None = None):
    state = {k: v.detach().float().contiguous().cpu() for k, v in bridge.state_dict().items()}
    spec = bridge.spec()
    meta = {"spec": json.dumps(spec)}
    if extra_meta:
        meta.update({k: json.dumps(v) for k, v in extra_meta.items()})
    save_file(state, path, metadata=meta)
    return spec
