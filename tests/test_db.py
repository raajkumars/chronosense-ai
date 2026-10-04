"""Unit tests for db module (Supabase integration).

All tests run WITHOUT Supabase credentials — verifying graceful
degradation, which is the path the app takes until credentials are
configured.
"""
import os
import pytest

import db


@pytest.fixture(autouse=True)
def clear_supabase_env(monkeypatch):
    """Ensure no Supabase credentials leak in from the environment."""
    monkeypatch.delenv(db._URL_KEY, raising=False)
    monkeypatch.delenv(db._ANON_KEY, raising=False)
    monkeypatch.setattr(db, "_client", None)
    yield
    monkeypatch.setattr(db, "_client", None)


def test_is_configured_false_without_credentials():
    """is_configured() is False when no credentials are present."""
    assert db.is_configured() is False


def test_get_client_none_without_credentials():
    """get_client() returns None instead of raising."""
    assert db.get_client() is None


def test_sign_up_degrades_without_credentials():
    """sign_up returns an error dict, never raises."""
    result = db.sign_up("a@b.com", "password123")
    assert result["ok"] is False
    assert "error" in result


def test_sign_in_degrades_without_credentials():
    """sign_in returns an error dict, never raises."""
    result = db.sign_in("a@b.com", "password123")
    assert result["ok"] is False


def test_restore_session_degrades_without_credentials():
    """restore_session returns an error dict, never raises."""
    result = db.restore_session("token", "refresh")
    assert result["ok"] is False


def test_sign_out_does_not_raise():
    """sign_out is a safe no-op without credentials."""
    db.sign_out()


def test_save_report_returns_none_without_credentials():
    """save_report returns None rather than raising."""
    result = db.save_report("user-1", {"jitter_proxy": 0.2}, {}, {"biological_age_estimate": 35})
    assert result is None


def test_save_report_none_for_empty_user():
    """save_report requires a user id."""
    assert db.save_report("", {}, {}, {}) is None


def test_list_reports_empty_without_credentials():
    """list_reports returns an empty list rather than raising."""
    assert db.list_reports("user-1") == []


def test_log_anonymized_false_without_credentials():
    """log_anonymized returns False rather than raising."""
    assert db.log_anonymized({}, {}, {}) is False


def test_anonymized_summary_empty_without_credentials():
    """anonymized_summary returns an empty dict rather than raising."""
    assert db.anonymized_summary() == {}


def test_num_coercion():
    """_num coerces numerics and rejects junk."""
    assert db._num(1.5) == 1.5
    assert db._num("2.5") == 2.5
    assert db._num(None) is None
    assert db._num("abc") is None


def test_strip_internal_summarizes_angle_arrays():
    """_strip_internal collapses frame arrays into summary stats."""
    stats = {"left_knee_angles": [10.0, 20.0, 30.0], "jitter_proxy": 0.2}
    out = db._strip_internal(stats)
    assert out["jitter_proxy"] == 0.2
    assert out["left_knee_angles"] == {"n": 3, "min": 10.0, "max": 30.0, "mean": 20.0}


def test_strip_internal_handles_non_dict():
    """_strip_internal tolerates non-dict input."""
    assert db._strip_internal(None) == {}


def test_strip_internal_skips_bad_arrays():
    """_strip_internal skips unsummarizable angle fields."""
    out = db._strip_internal({"left_knee_angles": None, "jitter_proxy": 0.1})
    assert "left_knee_angles" not in out
    assert out["jitter_proxy"] == 0.1
