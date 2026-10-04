"""Voice biomarker extraction using librosa.

Extracts MFCCs, spectral centroid, and jitter/shimmer proxies from audio.
Falls back to mock data when no audio file is provided.

All proxy values are normalized to 0-1 range via tanh so they work
consistently for both real and mock data.
"""
from __future__ import annotations

import math

import numpy as np


def extract_voice_features(audio_path: str | None = None, sr: int = 22050) -> dict:
    """Extract voice biomarkers from an audio file or return mock features.

    Args:
        audio_path: Path to a .wav file. If None, returns synthetic mock data.
        sr: Sample rate for librosa.load.

    Returns:
        dict with keys: mfcc_mean, mfcc_std, spectral_centroid_mean,
        spectral_centroid_std, jitter_proxy, shimmer_proxy, rms_mean, rms_std,
        zero_crossing_rate, duration_s, is_mock
    """
    if audio_path is None:
        return _mock_voice_features()

    import librosa

    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std = mfccs.std(axis=1)

    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = float(cent.mean())
    cent_std = float(cent.std())

    # Jitter proxy: frame-to-frame RMS energy variation (normalized to 0-1)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(rms.mean())
    rms_std = float(rms.std())
    jitter_raw = rms_std / (rms_mean + 1e-8)
    jitter_proxy = float(math.tanh(jitter_raw))

    # Shimmer proxy: frame-to-frame peak amplitude variation (normalized to 0-1)
    peaks = np.array([np.max(np.abs(y[i * sr // 100:(i + 1) * sr // 100]))
                      for i in range(min(100, len(y) // (sr // 100)))])
    shimmer_raw = float(peaks.std() / (peaks.mean() + 1e-8)) if len(peaks) > 1 else 0.0
    shimmer_proxy = float(math.tanh(shimmer_raw))

    # Zero crossing rate
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    return {
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
        "spectral_centroid_mean": cent_mean,
        "spectral_centroid_std": cent_std,
        "jitter_proxy": jitter_proxy,
        "shimmer_proxy": shimmer_proxy,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zero_crossing_rate": zcr,
        "duration_s": round(duration, 2),
        "is_mock": False,
    }


def _mock_voice_features() -> dict:
    """Return synthetic voice features for demo without audio hardware."""
    rng = np.random.default_rng(42)
    return {
        "mfcc_mean": rng.normal(0, 1, 13).round(3).tolist(),
        "mfcc_std": rng.uniform(0.1, 0.5, 13).round(3).tolist(),
        "spectral_centroid_mean": round(float(rng.normal(1800, 300)), 1),
        "spectral_centroid_std": round(float(rng.uniform(50, 200)), 1),
        "jitter_proxy": round(float(rng.uniform(0.1, 0.5)), 4),
        "shimmer_proxy": round(float(rng.uniform(0.1, 0.4)), 4),
        "rms_mean": round(float(rng.uniform(0.05, 0.2)), 4),
        "rms_std": round(float(rng.uniform(0.005, 0.03)), 4),
        "zero_crossing_rate": round(float(rng.uniform(0.02, 0.15)), 4),
        "duration_s": 30.0,
        "is_mock": True,
    }
