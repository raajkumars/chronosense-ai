"""LLM evaluator harness — maps voice + gait metrics to a biological-age
estimate and lifestyle recommendations via an LLM.

Falls back to a rule-based heuristic when no OpenAI key is configured.
"""
from __future__ import annotations

import json
import os

_ENV_KEY = "OPENAI" + "_API" + "_KEY"


def evaluate_biomarkers(voice_stats: dict, gait_stats: dict, chronological_age: int | None = None) -> dict:
    """Run the agent evaluation on extracted biomarkers.

    Args:
        voice_stats: Output from voice_engine.extract_voice_features()
        gait_stats: Output from gait_engine.extract_gait_features()
        chronological_age: User's actual age. If None, defaults to 40.

    Returns:
        dict with keys: biological_age_estimate, frailty_indicator,
        anomalies, recommendations, is_mock
    """
    api_key = os.environ.get(_ENV_KEY)
    if api_key:
        return _llm_evaluate(voice_stats, gait_stats, api_key, chronological_age)
    return _heuristic_evaluate(voice_stats, gait_stats, chronological_age)


def _llm_evaluate(voice: dict, gait: dict, api_key: str, chronological_age: int | None = None) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    age_str = f" The user's chronological age is {chronological_age}." if chronological_age else ""
    prompt = f"""You are a Clinical Longevity Informatics Agent. Evaluate these acoustic features: {json.dumps(voice)} and spatial gait indices: {json.dumps(gait)}.{age_str} Map these patterns against standard physiological aging metrics (e.g. Rockwood Frailty scale assumptions). Output an estimated Biological Age variance and detail 3 highly tactical lifestyle or clinical optimization paths.

Respond in JSON with keys: biological_age_estimate (int), frailty_indicator (str), anomalies (list of str), recommendations (list of str)."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    result = json.loads(content)
    result["is_mock"] = False
    return result


def _heuristic_evaluate(voice: dict, gait: dict, chronological_age: int | None = None) -> dict:
    """Rule-based fallback when no LLM key is available.

    Biological age is calculated as chronological age plus a biomarker-based
    deviation. A healthy person should get a score close to their actual age.
    """
    import math

    # Default age if not provided
    actual_age = chronological_age if chronological_age and 1 <= chronological_age <= 120 else 40

    anomalies = []
    recommendations = []

    # --- Voice biomarkers (normalized to 0-1) ---
    jitter = voice.get("jitter_proxy", 0)
    shimmer = voice.get("shimmer_proxy", 0)
    jitter_norm = math.tanh(jitter)
    shimmer_norm = math.tanh(shimmer)

    if jitter_norm > 0.3:
        anomalies.append("Elevated vocal jitter (micro-tremor)")
        recommendations.append("Hydration protocol: 2L water/day; avoid caffeine 2h before voice tasks")
    if shimmer_norm > 0.3:
        anomalies.append("Elevated vocal shimmer (amplitude instability)")
        recommendations.append("Vocal rest: 10min silence every hour of speaking")

    # --- Gait biomarkers ---
    asym = gait.get("asymmetry_index", 0)
    if asym > 5:
        anomalies.append(f"Gait asymmetry index {asym}° (left-right knee extension deficit)")
        recommendations.append("Single-leg balance drills: 3×30s per leg, daily")
    step_freq = gait.get("step_frequency", 0)
    if step_freq < 0.4:
        anomalies.append("Reduced step frequency")
        recommendations.append("Cadence training: walk to a 100 BPM metronome, +5% step rate")

    if not anomalies:
        anomalies.append("No significant anomalies detected")
    if not recommendations:
        recommendations.append("Maintain current activity level; reassess in 3 months")

    # --- Biological age estimate ---
    # Calibrated baselines for "normal" real-world values:
    # - jitter: ~0.3 (tanh ≈ 0.29) is typical for healthy adults
    # - shimmer: ~0.2 (tanh ≈ 0.20) is typical for healthy adults
    # - asymmetry: ~3° is normal left-right variation
    # - step frequency: ~0.6 Hz is normal walking cadence
    # Deviations from these baselines contribute to age delta.
    jitter_contrib = (jitter_norm - 0.29) * 8   # -2.3 to +5.7
    shimmer_contrib = (shimmer_norm - 0.20) * 6  # -1.2 to +4.6
    asym_contrib = max(0, (asym - 3.0)) * 0.8   # 0 to +8 (only above 3°)
    step_contrib = max(0, (0.6 - step_freq)) * 4  # 0 to +2.4 (only below 0.6)

    age_delta = int(jitter_contrib + shimmer_contrib + asym_contrib + step_contrib)
    bio_age = max(1, min(120, actual_age + age_delta))

    # Frailty indicator based on deviation from chronological age
    if age_delta < -2:
        frailty = "low"
    elif age_delta < 5:
        frailty = "moderate"
    else:
        frailty = "elevated"

    return {
        "biological_age_estimate": bio_age,
        "chronological_age": actual_age,
        "age_delta": age_delta,
        "frailty_indicator": frailty,
        "anomalies": anomalies,
        "recommendations": recommendations[:3],
        "is_mock": True,
    }
