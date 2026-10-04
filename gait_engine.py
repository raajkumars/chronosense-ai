"""Gait biomarker extraction using MediaPipe Pose.

Extracts hip/knee/ankle landmarks from video, computes joint angles,
and derives an Asymmetry Index. Falls back to mock data when no video
is provided.
"""
from __future__ import annotations

import numpy as np

# MediaPipe Pose landmark indices
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28


def extract_gait_features(video_path: str | None = None) -> dict:
    """Extract gait biomarkers from a video file or return mock features.

    Args:
        video_path: Path to a video file. If None, returns synthetic mock data.

    Returns:
        dict with keys: left_knee_angles, right_knee_angles, left_hip_angles,
        right_hip_angles, asymmetry_index, step_frequency, frames_processed,
        is_mock
    """
    if video_path is None:
        return _mock_gait_features()

    import cv2
    import mediapipe as mp

    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception:
        # MediaPipe failed to init (e.g. missing system libs on cloud) — fall back to mock
        return _mock_gait_features()

    cap = cv2.VideoCapture(video_path)
    left_knee_angles, right_knee_angles = [], []
    left_hip_angles, right_hip_angles = [], []
    frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            continue
        lm = results.pose_landmarks.landmark

        # Knee angles
        lk = _angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
        rk = _angle(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])
        left_knee_angles.append(lk)
        right_knee_angles.append(rk)

        # Hip angles (torso-thigh angle using shoulder-hip-knee)
        lh = _angle(lm[11], lm[LEFT_HIP], lm[LEFT_KNEE])   # left shoulder-hip-knee
        rh = _angle(lm[12], lm[RIGHT_HIP], lm[RIGHT_KNEE])  # right shoulder-hip-knee
        left_hip_angles.append(lh)
        right_hip_angles.append(rh)

    cap.release()
    pose.close()

    if not left_knee_angles:
        return _mock_gait_features()

    asymmetry = abs(max(left_knee_angles) - max(right_knee_angles))
    step_freq = _estimate_step_frequency(left_knee_angles)

    return {
        "left_knee_angles": [round(a, 1) for a in left_knee_angles],
        "right_knee_angles": [round(a, 1) for a in right_knee_angles],
        "left_hip_angles": [round(a, 1) for a in left_hip_angles],
        "right_hip_angles": [round(a, 1) for a in right_hip_angles],
        "asymmetry_index": round(asymmetry, 1),
        "step_frequency": round(step_freq, 2),
        "frames_processed": frames,
        "is_mock": False,
    }


def _angle(a, b, c) -> float:
    """Compute angle at point b given three landmarks (x, y)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _estimate_step_frequency(knee_angles: list[float]) -> float:
    """Estimate step frequency from knee angle oscillation."""
    if len(knee_angles) < 10:
        return 0.0
    arr = np.array(knee_angles)
    peaks = 0
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > 30:
            peaks += 1
    duration_s = len(arr) / 30.0  # assume 30fps
    return peaks / duration_s if duration_s > 0 else 0.0


def _mock_gait_features() -> dict:
    """Return synthetic gait features for demo without video hardware."""
    rng = np.random.default_rng(7)
    n = 90  # 3 seconds at 30fps
    t = np.linspace(0, 3, n)
    left_knee = 25 + 15 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 1, n)
    right_knee = 25 + 15 * np.sin(2 * np.pi * 0.5 * t + 0.3) + rng.normal(0, 1, n)
    left_hip = 170 + 5 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.5, n)
    right_hip = 170 + 5 * np.sin(2 * np.pi * 0.5 * t + 0.2) + rng.normal(0, 0.5, n)
    asymmetry = abs(max(left_knee) - max(right_knee))
    return {
        "left_knee_angles": [round(a, 1) for a in left_knee],
        "right_knee_angles": [round(a, 1) for a in right_knee],
        "left_hip_angles": [round(a, 1) for a in left_hip],
        "right_hip_angles": [round(a, 1) for a in right_hip],
        "asymmetry_index": round(float(asymmetry), 1),
        "step_frequency": 0.5,
        "frames_processed": n,
        "is_mock": True,
    }
