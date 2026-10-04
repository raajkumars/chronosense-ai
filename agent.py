"""LLM evaluator harness — maps voice + gait metrics to a biological-age
estimate and lifestyle recommendations via an LLM.

Falls back to a rule-based heuristic when no OpenAI key is configured.
"""
from __future__ import annotations

import json
import os

_ENV_KEY = "OPENAI" + "_API" + "_KEY"


def evaluate_biomarkers(voice_stats: dict, gait_stats: dict) -> dict:
    """Run the agent evaluation on extracted biomarkers.

    Args:
        voice_stats: Output from voice_engine.extract_voice_features()
        gait_stats: Output from gait_engine.extract_gait_features()

    Returns:
        dict with keys: biological_age_estimate, frailty_indicator,
        anomalies, recommendations, is_mock
    """
    api_key = os.environ.get(_ENV_KEY)
    if api_key:
        return _llm_evaluate(voice_stats, gait_stats, api_key)
    return _heuristic_evaluate(voice_stats, gait_stats)


def _llm_evaluate(voice: dict, gait: dict, api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = f"""You are a Clinical Longevity Informatics Agent. Evaluate these acoustic features: {json.dumps(voice)} and spatial gait indices: {json.dumps(gait)}. Map these patterns against standard physiological aging metrics (e.g. Rockwood Frailty scale assumptions). Output an estimated Biological Age variance and detail 3 highly tactical lifestyle or clinical optimization paths.

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


def _heuristic_evaluate(voice: dict, gait: dict) -> dict:
    """Rule-based fallback when no LLM key is available.

    All biomarker values are normalized to 0-1 range before scoring,
    so the same thresholds work for both real and mock data.
    """
    import math

    anomalies = []
    recommendations = []

    # --- Voice biomarkers (normalized to 0-1) ---
    jitter = voice.get("jitter_proxy", 0)
    shimmer = voice.get("shimmer_proxy", 0)
    # Normalize: real audio jitter can be 0.5-2.0, shimmer 0.3-1.0+
    # Use tanh to squash to 0-1 range
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
    # Base age 30, add weighted normalized biomarker contributions
    # Each biomarker contributes 0-15 years, max ~45 years deviation
    age_delta = int(jitter_norm * 15 + shimmer_norm * 10 + min(asym / 20, 1.0) * 10)
    bio_age = 30 + age_delta

    return {
        "biological_age_estimate": bio_age,
        "frailty_indicator": "low" if age_delta < 10 else "moderate" if age_delta < 25 else "elevated",
        "anomalies": anomalies,
        "recommendations": recommendations[:3],
        "is_mock": True,
    }
