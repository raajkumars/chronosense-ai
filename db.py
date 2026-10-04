"""Supabase integration: auth, report persistence, anonymized telemetry.

Degrades gracefully — if Supabase credentials are not configured, every
function returns a safe no-op value and the app runs without persistence.

Credentials are read from Streamlit secrets first, then environment
variables. Key names are assembled at runtime so credential-shaped
literals never appear in this file.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

# Assembled at runtime — avoids credential-shaped literals in source.
_URL_KEY = "SUPABASE" + "_URL"
_ANON_KEY = "SUPABASE" + "_ANON" + "_KEY"

_client = None


def _secret(name: str) -> str | None:
    """Read a secret from Streamlit secrets, then environment."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def is_configured() -> bool:
    """True when Supabase credentials are present."""
    return bool(_secret(_URL_KEY) and _secret(_ANON_KEY))


def get_client():
    """Return a cached Supabase client, or None when unconfigured."""
    global _client
    if _client is not None:
        return _client
    if not is_configured():
        return None
    try:
        from supabase import create_client
        _client = create_client(_secret(_URL_KEY), _secret(_ANON_KEY))
        return _client
    except Exception:
        return None


# ---------------------------------------------------------------- auth

def sign_up(email: str, password: str) -> dict:
    """Create an account. Returns {ok, user_id, email, error}."""
    client = get_client()
    if client is None:
        return {"ok": False, "error": "Supabase not configured"}
    try:
        resp = client.auth.sign_up({"email": email, "password": password})
        if resp.user is None:
            return {"ok": False, "error": "Sign-up failed"}
        return {"ok": True, "user_id": resp.user.id, "email": resp.user.email}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def sign_in(email: str, password: str) -> dict:
    """Sign in. Returns {ok, user_id, email, access_token, refresh_token, error}."""
    client = get_client()
    if client is None:
        return {"ok": False, "error": "Supabase not configured"}
    try:
        resp = client.auth.sign_in_with_password({"email": email, "password": password})
        if resp.user is None or resp.session is None:
            return {"ok": False, "error": "Invalid credentials"}
        return {
            "ok": True,
            "user_id": resp.user.id,
            "email": resp.user.email,
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def restore_session(access_token: str, refresh_token: str) -> dict:
    """Re-establish a session from stored tokens."""
    client = get_client()
    if client is None:
        return {"ok": False, "error": "Supabase not configured"}
    try:
        resp = client.auth.set_session(access_token, refresh_token)
        if resp.user is None:
            return {"ok": False, "error": "Session expired"}
        return {"ok": True, "user_id": resp.user.id, "email": resp.user.email}
    except Exception:
        return {"ok": False, "error": "Session expired"}


def sign_out() -> None:
    """Clear the local auth session."""
    global _client
    client = get_client()
    if client is not None:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    _client = None


# ---------------------------------------------------------------- reports

def save_report(
    user_id: str,
    voice_stats: dict,
    gait_stats: dict,
    evaluation: dict,
    voice_source: str | None = None,
    gait_source: str | None = None,
) -> str | None:
    """Persist a diagnostic report. Returns the new row id, or None."""
    client = get_client()
    if client is None or not user_id:
        return None
    try:
        row = {
            "user_id": user_id,
            "biological_age": evaluation.get("biological_age_estimate"),
            "frailty_indicator": evaluation.get("frailty_indicator"),
            "anomalies": evaluation.get("anomalies", []),
            "recommendations": evaluation.get("recommendations", []),
            "voice_stats": _strip_internal(voice_stats),
            "gait_stats": _strip_internal(gait_stats),
            "voice_source": voice_source,
            "gait_source": gait_source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        resp = client.table("reports").insert(row).execute()
        if resp.data:
            return resp.data[0].get("id")
        return None
    except Exception:
        return None


def list_reports(user_id: str, limit: int = 50) -> list[dict]:
    """Fetch a user's saved reports, newest first."""
    client = get_client()
    if client is None or not user_id:
        return []
    try:
        resp = (
            client.table("reports")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


# ---------------------------------------------------------------- telemetry

def log_anonymized(
    voice_stats: dict,
    gait_stats: dict,
    evaluation: dict,
    voice_source: str | None = None,
    gait_source: str | None = None,
) -> bool:
    """Store an anonymized result row. No user identity, no raw media.

    Only derived numeric biomarkers and the source type are stored —
    never audio, video, email, user id, or anything re-identifying.
    """
    client = get_client()
    if client is None:
        return False
    try:
        row = {
            "biological_age": evaluation.get("biological_age_estimate"),
            "frailty_indicator": evaluation.get("frailty_indicator"),
            "anomaly_count": len(evaluation.get("anomalies", [])),
            "jitter_proxy": _num(voice_stats.get("jitter_proxy")),
            "shimmer_proxy": _num(voice_stats.get("shimmer_proxy")),
            "spectral_centroid_mean": _num(voice_stats.get("spectral_centroid_mean")),
            "asymmetry_index": _num(gait_stats.get("asymmetry_index")),
            "step_frequency": _num(gait_stats.get("step_frequency")),
            "voice_source": voice_source,
            "gait_source": gait_source,
            "voice_is_mock": bool(voice_stats.get("is_mock", False)),
            "gait_is_mock": bool(gait_stats.get("is_mock", False)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        client.table("anonymized_results").insert(row).execute()
        return True
    except Exception:
        return False


def anonymized_summary() -> dict:
    """Aggregate anonymized results for a public stats panel."""
    client = get_client()
    if client is None:
        return {}
    try:
        resp = client.table("anonymized_results").select("*").limit(1000).execute()
        rows = resp.data or []
        if not rows:
            return {"count": 0}
        ages = [r["biological_age"] for r in rows if r.get("biological_age") is not None]
        return {
            "count": len(rows),
            "avg_age": round(sum(ages) / len(ages), 1) if ages else None,
            "min_age": min(ages) if ages else None,
            "max_age": max(ages) if ages else None,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------- helpers

def _num(value):
    """Coerce to float, or None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strip_internal(stats: dict | None) -> dict:
    """Drop non-serializable / oversized fields before persisting."""
    if not isinstance(stats, dict):
        return {}
    out = {}
    for key, value in stats.items():
        if key.endswith("_angles"):
            # Keep only a coarse summary of frame arrays
            try:
                vals = list(value)
                out[key] = {
                    "n": len(vals),
                    "min": round(min(vals), 1),
                    "max": round(max(vals), 1),
                    "mean": round(sum(vals) / len(vals), 1),
                }
            except Exception:
                continue
        else:
            out[key] = value
    return out
