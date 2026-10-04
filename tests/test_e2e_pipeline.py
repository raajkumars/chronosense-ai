"""E2E integration test: full pipeline from mock data to PDF report."""
import os
import pytest
from voice_engine import extract_voice_features
from gait_engine import extract_gait_features
from agent import evaluate_biomarkers
from report import generate_pdf


def test_full_pipeline_mock_data():
    """Test the complete pipeline: voice + gait -> agent -> PDF."""
    # Step 1: Extract voice features (mock)
    voice_stats = extract_voice_features(None)
    assert voice_stats["is_mock"] is True

    # Step 2: Extract gait features (mock)
    gait_stats = extract_gait_features(None)
    assert gait_stats["is_mock"] is True

    # Step 3: Agent evaluation
    evaluation = evaluate_biomarkers(voice_stats, gait_stats)
    assert evaluation["is_mock"] is True
    assert "biological_age_estimate" in evaluation
    assert "recommendations" in evaluation

    # Step 4: Generate PDF
    pdf_path = generate_pdf(voice_stats, gait_stats, evaluation)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000

    # Clean up
    os.unlink(pdf_path)


def test_full_pipeline_produces_consistent_results():
    """Test that the full pipeline produces deterministic results."""
    voice_stats = extract_voice_features(None)
    gait_stats = extract_gait_features(None)
    evaluation = evaluate_biomarkers(voice_stats, gait_stats)

    # Run again
    voice_stats2 = extract_voice_features(None)
    gait_stats2 = extract_gait_features(None)
    evaluation2 = evaluate_biomarkers(voice_stats2, gait_stats2)

    # Results should be identical (deterministic mock data)
    assert evaluation["biological_age_estimate"] == evaluation2["biological_age_estimate"]
    assert evaluation["frailty_indicator"] == evaluation2["frailty_indicator"]
    assert evaluation["anomalies"] == evaluation2["anomalies"]
    assert evaluation["recommendations"] == evaluation2["recommendations"]


def test_pipeline_handles_edge_cases():
    """Test pipeline with edge case inputs."""
    # Minimal voice stats
    voice_stats = {
        "is_mock": True,
        "mfcc_mean": [0.0] * 13,
        "spectral_centroid": 0.0,
        "jitter_proxy": 0.0,
        "shimmer_proxy": 0.0,
        "duration_s": 0.1,
    }
    # Minimal gait stats
    gait_stats = {
        "is_mock": True,
        "asymmetry_index": 0.0,
        "step_frequency": 0.0,
        "frames_processed": 1,
    }
    evaluation = evaluate_biomarkers(voice_stats, gait_stats)
    assert "biological_age_estimate" in evaluation
    assert "recommendations" in evaluation
