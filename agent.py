"""LLM evaluator harness — maps voice + gait metrics to a biological-age
estimate and lifestyle recommendations via an LLM.

Falls back to a rule-based heuristic when no OpenAI key is configured.
"""
from __future__ import annotations

import json
import os


def evaluate_biomarkers(voice_stats: dict, gait_stats: dict) -> dict:
    """Run the agent evaluation on extracted biomarkers.

    Args:
        voice_stats: Output from voice_engine.extract_voice_features()
        gait_stats: Output from gait_engine.extract_gait_features()

    Returns:
        dict with keys: biological_age_estimate, frailty_indicator,
        anomalies, recommendations, is_mock
    """
    api_key = os.environ.get("OPENAI_API_KEY")
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
    """Rule-based fallback when no LLM key is available."""
    anomalies = []
    recommendations = []

    # Voice-based heuristics
    jitter = voice.get("jitter_proxy", 0)
    shimmer = voice.get("shimmer_proxy", 0)
    if jitter > 0.05:
        anomalies.append("Elevated vocal jitter (micro-tremor)")
        recommendations.append("Hydration protocol: 2L water/day; avoid caffeine 2h before voice tasks")
    if shimmer > 0.08:
        anomalies.append("Elevated vocal shimmer (amplitude instability)")
        recommendations.append("Vocal rest: 10min silence every hour of speaking")

    # Gait-based heuristics
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

    # Rough biological age proxy
    base_age = 30
    age_delta = int(jitter * 200 + shimmer * 100 + asym * 0.5)
    bio_age = base_age + age_delta

    return {
        "biological_age_estimate": bio_age,
        "frailty_indicator": "low" if age_delta < 5 else "moderate" if age_delta < 15 else "elevated",
        "anomalies": anomalies,
        "recommendations": recommendations[:3],
        "is_mock": True,
    }
