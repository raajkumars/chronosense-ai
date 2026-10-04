"""Unit tests for gait_engine module."""
import pytest
import numpy as np
from gait_engine import extract_gait_features, _angle, _estimate_step_frequency


def test_mock_gait_features():
    """Test that mock gait features are returned when no video is provided."""
    result = extract_gait_features(None)
    assert result["is_mock"] is True
    assert "left_knee_angles" in result
    assert "right_knee_angles" in result
    assert "left_hip_angles" in result
    assert "right_hip_angles" in result
    assert "asymmetry_index" in result
    assert "step_frequency" in result
    assert result["frames_processed"] > 0


def test_mock_gait_asymmetry_non_negative():
    """Test that asymmetry index is non-negative."""
    result = extract_gait_features(None)
    assert result["asymmetry_index"] >= 0


def test_mock_gait_step_frequency_positive():
    """Test that step frequency is positive."""
    result = extract_gait_features(None)
    assert result["step_frequency"] > 0


def test_mock_gait_knee_angles_reasonable():
    """Test that knee angles are in a reasonable range (0-180 degrees)."""
    result = extract_gait_features(None)
    for angle in result["left_knee_angles"]:
        assert 0 <= angle <= 180
    for angle in result["right_knee_angles"]:
        assert 0 <= angle <= 180


def test_mock_gait_hip_angles_reasonable():
    """Test that hip angles are in a reasonable range (0-180 degrees)."""
    result = extract_gait_features(None)
    for angle in result["left_hip_angles"]:
        assert 0 <= angle <= 180
    for angle in result["right_hip_angles"]:
        assert 0 <= angle <= 180


def test_mock_gait_deterministic():
    """Test that mock features are deterministic (same seed)."""
    r1 = extract_gait_features(None)
    r2 = extract_gait_features(None)
    assert r1["asymmetry_index"] == r2["asymmetry_index"]
    assert r1["step_frequency"] == r2["step_frequency"]


def test_angle_calculation():
    """Test the _angle helper function with known geometry."""
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Right angle: (0,1) - (0,0) - (1,0) = 90 degrees
    a = Point(0, 1)
    b = Point(0, 0)
    c = Point(1, 0)
    angle = _angle(a, b, c)
    assert abs(angle - 90.0) < 0.1


def test_estimate_step_frequency():
    """Test step frequency estimation with synthetic oscillation."""
    # Create a simple sine wave with 3 peaks over 30 frames
    t = np.linspace(0, 3, 30)
    angles = 30 + 15 * np.sin(2 * np.pi * 0.5 * t)
    freq = _estimate_step_frequency(angles.tolist())
    assert freq > 0


def test_estimate_step_frequency_short_input():
    """Test step frequency with too-short input returns 0."""
    freq = _estimate_step_frequency([1.0, 2.0, 3.0])
    assert freq == 0.0
