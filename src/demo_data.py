from __future__ import annotations

import numpy as np


def generate_demo_eeg(fs: float = 500.0, seconds: float = 60.0, seed: int = 6041):
    """Create a deterministic EEG-like signal for the public demo.

    The synthetic trace is not intended to model neonatal EEG clinically. It only
    provides a non-sensitive signal with low-frequency structure, drift and noise
    so the DSP pipeline can be executed without external EEG files.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds), dtype=float) / fs
    signal = (
        18.0 * np.sin(2.0 * np.pi * 1.7 * t)
        + 7.0 * np.sin(2.0 * np.pi * 5.2 * t + 0.3)
        + 3.0 * np.sin(2.0 * np.pi * 10.5 * t + 1.1)
        + 5.0 * np.sin(2.0 * np.pi * 0.18 * t)
    )
    envelope = 1.0 + 0.25 * np.sin(2.0 * np.pi * 0.025 * t)
    noise = rng.normal(0.0, 2.0, size=t.shape)
    return envelope * signal + noise


