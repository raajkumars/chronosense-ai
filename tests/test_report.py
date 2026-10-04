"""Unit tests for report module."""
import os
import pytest
from report import generate_pdf


def test_generate_pdf_creates_file():
    """Test that PDF generation creates a file."""
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
    evaluation = {
        "is_mock": True,
        "biological_age_estimate": 45,
        "frailty_indicator": "low",
        "anomalies": ["Elevated vocal shimmer"],
        "recommendations": ["Vocal rest: 10min silence every hour"],
    }
    pdf_path = generate_pdf(voice_stats, gait_stats, evaluation)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    assert pdf_path.endswith(".pdf")
    # Clean up
    os.unlink(pdf_path)


def test_generate_pdf_content():
    """Test that generated PDF has reasonable size (not empty)."""
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
    evaluation = {
        "is_mock": True,
        "biological_age_estimate": 45,
        "frailty_indicator": "low",
        "anomalies": ["Elevated vocal shimmer"],
        "recommendations": ["Vocal rest: 10min silence every hour"],
    }
    pdf_path = generate_pdf(voice_stats, gait_stats, evaluation)
    size = os.path.getsize(pdf_path)
    # A real PDF with content should be at least 1KB
    assert size > 1000
    os.unlink(pdf_path)
