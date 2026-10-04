"""Gait biomarker extraction using MediaPipe Pose.

Extracts hip/knee/ankle landmarks from video, computes joint angles,
and derives an Asymmetry Index. Falls back to mock data when no video
is provided or MediaPipe is unavailable.
"""
from __future__ import annotations

import os

# Disable GPU for MediaPipe (macOS arm64 crashes with Metal backend)
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import numpy as np

# MediaPipe Pose landmark indices
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

# Model path
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


def _load_pose_landmarker():
    """Load MediaPipe PoseLandmarker with the Tasks API."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not os.path.exists(_MODEL_PATH):
            return None

        base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(options)
    except Exception:
        return None


def validate_gait_video(video_path: str) -> tuple[bool, str]:
    """Validate that a video meets minimum requirements for gait analysis.

    Returns:
        (is_valid, reason) — reason is empty string if valid.
    """
    try:
        import cv2
    except ImportError:
        return False, "OpenCV not available"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if total_frames < 30:
        return False, f"Video too short — {total_frames} frames (need at least 30)"
    if duration < 1.5:
        return False, f"Video too short — {duration:.1f}s (need at least 1.5s)"
    if duration > 15:
        return False, f"Video too long — {duration:.1f}s (keep it under 15s)"

    return True, ""


def extract_gait_features(video_path: str | None = None) -> dict:
    """Extract gait biomarkers from a video file or return mock features.

    Args:
        video_path: Path to a video file. If None, returns synthetic mock data.

    Returns:
        dict with keys: left_knee_angles, right_knee_angles, left_hip_angles,
        right_hip_angles, asymmetry_index, step_frequency, frames_processed,
        is_mock, rejected (bool), rejection_reason (str)
    """
    if video_path is None:
        return _mock_gait_features()

    # Validate video before processing
    is_valid, reason = validate_gait_video(video_path)
    if not is_valid:
        return {
            "left_knee_angles": [],
            "right_knee_angles": [],
            "left_hip_angles": [],
            "right_hip_angles": [],
            "asymmetry_index": 0.0,
            "step_frequency": 0.0,
            "frames_processed": 0,
            "is_mock": False,
            "rejected": True,
            "rejection_reason": reason,
        }

    # Try to load MediaPipe PoseLandmarker (Tasks API)
    landmarker = _load_pose_landmarker()
    if landmarker is None:
        return {
            "left_knee_angles": [],
            "right_knee_angles": [],
            "left_hip_angles": [],
            "right_hip_angles": [],
            "asymmetry_index": 0.0,
            "step_frequency": 0.0,
            "frames_processed": 0,
            "is_mock": False,
            "rejected": True,
            "rejection_reason": "MediaPipe model not available — cannot process video",
        }

    try:
        import cv2
        import mediapipe as mp
    except Exception:
        return {
            "left_knee_angles": [],
            "right_knee_angles": [],
            "left_hip_angles": [],
            "right_hip_angles": [],
            "asymmetry_index": 0.0,
            "step_frequency": 0.0,
            "frames_processed": 0,
            "is_mock": False,
            "rejected": True,
            "rejection_reason": "OpenCV/MediaPipe not available",
        }

    cap = cv2.VideoCapture(video_path)
    left_knee_angles, right_knee_angles = [], []
    left_hip_angles, right_hip_angles = [], []
    frames = 0
    pose_detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Get timestamp in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Process with PoseLandmarker
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.pose_landmarks:
            continue

        pose_detected_frames += 1
        lm = results.pose_landmarks[0]  # First pose

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
    landmarker.close()

    # Reject if pose not detected in enough frames
    if not left_knee_angles:
        return {
            "left_knee_angles": [],
            "right_knee_angles": [],
            "left_hip_angles": [],
            "right_hip_angles": [],
            "asymmetry_index": 0.0,
            "step_frequency": 0.0,
            "frames_processed": frames,
            "is_mock": False,
            "rejected": True,
            "rejection_reason": "No body detected in video — make sure full body is visible",
        }

    pose_ratio = pose_detected_frames / frames if frames > 0 else 0
    if pose_ratio < 0.3:
        return {
            "left_knee_angles": [],
            "right_knee_angles": [],
            "left_hip_angles": [],
            "right_hip_angles": [],
            "asymmetry_index": 0.0,
            "step_frequency": 0.0,
            "frames_processed": frames,
            "is_mock": False,
            "rejected": True,
            "rejection_reason": f"Body only detected in {pose_ratio:.0%} of frames — improve lighting and ensure full body is in frame",
        }

    asymmetry = abs(max(left_knee_angles) - max(right_knee_angles))
    step_freq = _estimate_step_frequency(left_knee_angles)

    # Reject if no walking motion detected
    if step_freq < 0.2:
        return {
            "left_knee_angles": [],
            "right_knee_angles": [],
            "left_hip_angles": [],
            "right_hip_angles": [],
            "asymmetry_index": 0.0,
            "step_frequency": 0.0,
            "frames_processed": frames,
            "is_mock": False,
            "rejected": True,
            "rejection_reason": "No walking motion detected — you must walk (not stand still) for gait analysis",
        }

    return {
        "left_knee_angles": [round(a, 1) for a in left_knee_angles],
        "right_knee_angles": [round(a, 1) for a in right_knee_angles],
        "left_hip_angles": [round(a, 1) for a in left_hip_angles],
        "right_hip_angles": [round(a, 1) for a in right_hip_angles],
        "asymmetry_index": round(asymmetry, 1),
        "step_frequency": round(step_freq, 2),
        "frames_processed": frames,
        "is_mock": False,
        "rejected": False,
        "rejection_reason": "",
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
    """Estimate step frequency from knee angle oscillation.
    
    Uses peak detection with a minimum prominence threshold to avoid
    counting noise as steps. Normal walking cadence is 0.5-2.0 Hz.
    """
    if len(knee_angles) < 10:
        return 0.0
    arr = np.array(knee_angles)
    
    # Calculate prominence threshold as 20% of the range
    arr_range = arr.max() - arr.min()
    if arr_range < 5:  # No significant oscillation
        return 0.0
    prominence_threshold = arr_range * 0.2
    
    # Find peaks with minimum prominence
    peaks = 0
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            # Check prominence: must be higher than neighbors by threshold
            left_min = arr[max(0, i-5):i].min() if i > 0 else arr[i]
            right_min = arr[i+1:min(len(arr), i+6)].min() if i < len(arr)-1 else arr[i]
            prominence = arr[i] - max(left_min, right_min)
            if prominence >= prominence_threshold:
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
        "rejected": False,
        "rejection_reason": "",
    }
