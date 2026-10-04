"""Unit tests for agent module."""
import pytest
from agent import evaluate_biomarkers


def test_evaluate_mock_data():
    """Test agent evaluation with mock data (no OpenAI key)."""
    voice_stats = {
        "is_mock": True,
        "mfcc_mean": [0.0] * 13,
        "spectral_centroid": 1500.0,
        "jitter_proxy": 0.02,
        "shimmer_proxy": 0.05,
        "duration_s": 3.0,
    }
    gait_stats = {
        "is_mock": True,
        "asymmetry_index": 5.0,
        "step_frequency": 0.5,
        "frames_processed": 90,
    }
    result = evaluate_biomarkers(voice_stats, gait_stats)
    assert result["is_mock"] is True
    assert "biological_age_estimate" in result
    assert "frailty_indicator" in result
    assert "anomalies" in result
    assert "recommendations" in result
    assert isinstance(result["anomalies"], list)
    assert isinstance(result["recommendations"], list)


def test_evaluate_biological_age_reasonable():
    """Test that biological age estimate is in a reasonable range."""
    voice_stats = {
        "is_mock": True,
        "mfcc_mean": [0.0] * 13,
        "spectral_centroid": 1500.0,
        "jitter_proxy": 0.02,
        "shimmer_proxy": 0.05,
        "duration_s": 3.0,
    }
    gait_stats = {
        "is_mock": True,
        "asymmetry_index": 5.0,
        "step_frequency": 0.5,
        "frames_processed": 90,
    }
    result = evaluate_biomarkers(voice_stats, gait_stats)
    age = result["biological_age_estimate"]
    assert 20 <= age <= 100


def test_evaluate_frailty_indicator_valid():
    """Test that frailty indicator is one of the expected values."""
    voice_stats = {
        "is_mock": True,
        "mfcc_mean": [0.0] * 13,
        "spectral_centroid": 1500.0,
        "jitter_proxy": 0.02,
        "shimmer_proxy": 0.05,
        "duration_s": 3.0,
    }
    gait_stats = {
        "is_mock": True,
        "asymmetry_index": 5.0,
        "step_frequency": 0.5,
        "frames_processed": 90,
    }
    result = evaluate_biomarkers(voice_stats, gait_stats)
    assert result["frailty_indicator"] in ["low", "moderate", "high"]


def test_evaluate_high_jitter_flags_anomaly():
    """Test that high jitter triggers an anomaly flag."""
    voice_stats = {
        "is_mock": True,
        "mfcc_mean": [0.0] * 13,
        "spectral_centroid": 1500.0,
        "jitter_proxy": 0.15,  # very high jitter
        "shimmer_proxy": 0.05,
        "duration_s": 3.0,
    }
    gait_stats = {
        "is_mock": True,
        "asymmetry_index": 5.0,
        "step_frequency": 0.5,
        "frames_processed": 90,
    }
    result = evaluate_biomarkers(voice_stats, gait_stats)
    # High jitter should produce at least one anomaly
    assert len(result["anomalies"]) > 0


def test_evaluate_recommendations_present():
    """Test that recommendations are always present."""
    voice_stats = {
        "is_mock": True,
        "mfcc_mean": [0.0] * 13,
        "spectral_centroid": 1500.0,
        "jitter_proxy": 0.02,
        "shimmer_proxy": 0.05,
        "duration_s": 3.0,
    }
    gait_stats = {
        "is_mock": True,
        "asymmetry_index": 5.0,
        "step_frequency": 0.5,
        "frames_processed": 90,
    }
    result = evaluate_biomarkers(voice_stats, gait_stats)
    assert len(result["recommendations"]) > 0
