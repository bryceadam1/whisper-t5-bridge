"""Whisper log-mel features and encoder forward for truncated windows.

The game mod runs Whisper base.en's encoder on the audio plus 3 s (300 mel frames) of silence, rounded
up to an even frame count, rather than on the standard 30 s window (cf. whisper.cpp's audio_ctx). The
encoder then sees T_mel frames and emits T_mel / 2 states, using the first T_mel / 2 rows of its
positional embedding. Both window kinds go through the same code here:

  truncated: samples = audio ++ zeros(3 s), zero-padded to a multiple of 320 samples (even mel count)
  full:      samples = audio zero-padded (or cut) to exactly 30 s (3000 mel frames, 1500 states)

The log-mel matches openai/whisper's `log_mel_spectrogram`: 400-point Hann STFT, hop 160, centred with
reflect padding, the last frame dropped, power spectrum, 80 Slaney mel bins, log10 clamped at 1e-10,
floored at (max - 8), then (x + 4) / 4. The max is taken over the whole window; silence never raises it,
so a truncated window yields the same values as the matching frames of the full window.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
HOP = 160
N_FFT = 400
SILENCE_SECONDS = 3.0
FULL_SECONDS = 30.0


def window_samples(audio: np.ndarray, window: str) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if window == "truncated":
        n = len(audio) + int(SILENCE_SECONDS * SAMPLE_RATE)
        n = -(-n // (2 * HOP)) * (2 * HOP)
    elif window == "full":
        n = int(FULL_SECONDS * SAMPLE_RATE)
        audio = audio[:n]
    else:
        raise ValueError(window)
    out = np.zeros(n, dtype=np.float32)
    out[: len(audio)] = audio
    return out


class LogMel(torch.nn.Module):
    def __init__(self, mel_filters: np.ndarray):
        """mel_filters: (n_freq=201, n_mels=80), as in WhisperFeatureExtractor.mel_filters."""
        super().__init__()
        self.register_buffer("filters", torch.as_tensor(mel_filters, dtype=torch.float32).T.contiguous())
        self.register_buffer("window", torch.hann_window(N_FFT))

    @torch.no_grad()
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (N,) float32 at 16 kHz -> (80, N / 160) log-mel."""
        stft = torch.stft(samples, N_FFT, HOP, window=self.window, return_complex=True)
        power = stft[..., :-1].abs() ** 2
        mel = self.filters @ power
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.max() - 8.0)
        return (log + 4.0) / 4.0


@torch.no_grad()
def encode(encoder, mel: torch.Tensor) -> torch.Tensor:
    """Run a transformers WhisperEncoder on (B, 80, T_mel) features of any even length <= 3000.

    Same computation as WhisperEncoder.forward, minus its fixed 3000-frame check: two GELU convs, the
    first T_mel / 2 positional embeddings, the encoder layers, the final layer norm.
    """
    x = F.gelu(encoder.conv1(mel))
    x = F.gelu(encoder.conv2(x)).permute(0, 2, 1)
    x = x + encoder.embed_positions.weight[: x.shape[1]].to(x.dtype)
    for layer in encoder.layers:
        out = layer(x, None, layer_head_mask=None)
        x = out[0] if isinstance(out, tuple) else out
    return encoder.layer_norm(x)
