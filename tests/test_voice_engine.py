"""Unit tests for voice_engine module."""
import pytest
import numpy as np
from voice_engine import extract_voice_features


def test_mock_voice_features():
    """Test that mock voice features are returned when no file is provided."""
    result = extract_voice_features(None)
    assert result["is_mock"] is True
    assert "mfcc_mean" in result
    assert "spectral_centroid_mean" in result
    assert "jitter_proxy" in result
    assert "shimmer_proxy" in result
    assert result["duration_s"] > 0


def test_mock_voice_mfcc_shape():
    """Test that MFCC mean vector has expected shape (13 coefficients)."""
    result = extract_voice_features(None)
    assert len(result["mfcc_mean"]) == 13


def test_mock_voice_jitter_positive():
    """Test that jitter proxy is a positive value."""
    result = extract_voice_features(None)
    assert result["jitter_proxy"] > 0


def test_mock_voice_shimmer_positive():
    """Test that shimmer proxy is a positive value."""
    result = extract_voice_features(None)
    assert result["shimmer_proxy"] > 0


def test_mock_voice_spectral_centroid_positive():
    """Test that spectral centroid is a positive value."""
    result = extract_voice_features(None)
    assert result["spectral_centroid_mean"] > 0


def test_mock_voice_deterministic():
    """Test that mock features are deterministic (same seed)."""
    r1 = extract_voice_features(None)
    r2 = extract_voice_features(None)
    assert r1["jitter_proxy"] == r2["jitter_proxy"]
    assert r1["shimmer_proxy"] == r2["shimmer_proxy"]
