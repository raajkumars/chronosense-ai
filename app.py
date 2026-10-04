"""ChronoSense AI — Streamlit dashboard.

Redesigned UX: [Capture][Upload][Test Data] buttons per tab,
automatic processing when data is ready, evaluation section that
clears when sample changes and only appears when there is data.

Adds optional accounts (Supabase) for saving reports and tracking
progress, plus opt-in anonymized result collection.
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st
import plotly.graph_objects as go

from voice_engine import extract_voice_features
from gait_engine import extract_gait_features
from agent import evaluate_biomarkers
from report import generate_pdf
import db

st.set_page_config(page_title="ChronoSense AI", page_icon="🧬", layout="wide")

st.title("🧬 ChronoSense AI — Longevity Biomarker Diagnostics Agent")
st.caption("Voice & Gait Biomarker Analysis · Sundai Hack 143 · Biomarkers of Aging")

# --- Session state initialization ---
_DEFAULTS = {
    "voice_stats": None,
    "gait_stats": None,
    "evaluation": None,
    "pdf_path": None,
    "voice_source": None,
    "gait_source": None,
    "user": None,
    "saved_report_id": None,
    "telemetry_logged": False,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def clear_evaluation():
    """Clear evaluation results when sample changes."""
    st.session_state["evaluation"] = None
    st.session_state["pdf_path"] = None
    st.session_state["saved_report_id"] = None
    st.session_state["telemetry_logged"] = False


def process_voice(source: str, file_path: str | None = None):
    """Process voice data from the given source."""
    with st.spinner("Extracting voice features..."):
        stats = extract_voice_features(file_path)
    st.session_state["voice_stats"] = stats
    st.session_state["voice_source"] = source
    clear_evaluation()

    # Show waveform
    if stats.get("is_mock"):
        import numpy as np
        t = np.linspace(0, 3, 22050)
        y = 0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t)
        fig = go.Figure(go.Scatter(x=t, y=y, line=dict(color="#6366f1")))
        fig.update_layout(title="Test Data Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)
    elif file_path is not None:
        import librosa
        import numpy as np
        y, sr = librosa.load(file_path, sr=22050)
        t = np.linspace(0, len(y) / sr, len(y))
        fig = go.Figure(go.Scatter(x=t, y=y, line=dict(color="#6366f1")))
        fig.update_layout(title="Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

    st.json(stats)


def process_gait(source: str, file_path: str | None = None):
    """Process gait data from the given source."""
    with st.spinner("Processing gait video..."):
        stats = extract_gait_features(file_path)
    st.session_state["gait_stats"] = stats
    st.session_state["gait_source"] = source
    clear_evaluation()

    # Show knee angles
    if stats.get("left_knee_angles"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=stats["left_knee_angles"], name="Left Knee", line=dict(color="#6366f1")))
        fig.add_trace(go.Scatter(y=stats["right_knee_angles"], name="Right Knee", line=dict(color="#f59e0b")))
        fig.update_layout(title="Knee Joint Angles Over Time", xaxis_title="Frame", yaxis_title="Angle (°)")
        st.plotly_chart(fig, use_container_width=True)

    st.json(stats)


# --- Sidebar: account + privacy ---
with st.sidebar:
    st.header("👤 Account")

    if not db.is_configured():
        st.caption("Accounts unavailable — Supabase not configured. The app works fully without signing in.")
    elif st.session_state["user"] is None:
        auth_mode = st.radio("Sign in or create an account", ["Sign in", "Sign up"], horizontal=True)
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")

        if st.button("Create account" if auth_mode == "Sign up" else "Sign in", type="primary", use_container_width=True):
            if not email or not password:
                st.error("Enter both email and password.")
            elif auth_mode == "Sign up":
                result = db.sign_up(email, password)
                if result["ok"]:
                    st.success("Account created — you can sign in now.")
                else:
                    st.error(result.get("error", "Sign-up failed"))
            else:
                result = db.sign_in(email, password)
                if result["ok"]:
                    st.session_state["user"] = {"id": result["user_id"], "email": result["email"]}
                    st.rerun()
                else:
                    st.error(result.get("error", "Sign-in failed"))
    else:
        st.success(f"Signed in as {st.session_state['user']['email']}")
        if st.button("Sign out", use_container_width=True):
            db.sign_out()
            st.session_state["user"] = None
            st.rerun()

    st.divider()
    st.header("🔒 Privacy")
    share_anonymized = st.checkbox(
        "Contribute anonymized results",
        value=True,
        help=(
            "Stores derived biomarker numbers only — no audio, no video, no email, "
            "no account id. Used to improve the aging model."
        ),
    )

    st.divider()
    with st.expander("📈 Community stats"):
        summary = db.anonymized_summary()
        if summary.get("count"):
            st.metric("Contributed results", summary["count"])
            if summary.get("avg_age") is not None:
                st.metric("Average estimated age", summary["avg_age"])
        else:
            st.caption("No contributed results yet.")

# --- Voice Tab ---
tab_voice, tab_gait, tab_history = st.tabs(["🎙️ Voice Diagnostic", "🚶 Gait Track", "📚 My Reports"])

with tab_voice:
    st.subheader("Voice Biomarker Analysis")

    col_cap, col_up, col_test = st.columns(3)
    with col_cap:
        if st.button("🎤 Capture", key="voice_capture", use_container_width=True):
            st.session_state["voice_capture_active"] = True
    with col_up:
        if st.button("📁 Upload", key="voice_upload", use_container_width=True):
            st.session_state["voice_upload_active"] = True
    with col_test:
        if st.button("🧪 Test Data", key="voice_test", use_container_width=True):
            process_voice("test", None)

    # Capture audio
    if st.session_state.get("voice_capture_active"):
        st.info("🎤 Record a voice sample below")
        audio_value = st.audio_input("Record voice sample")
        if audio_value is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_value.read())
                voice_path = tmp.name
            st.success("✅ Voice captured!")
            st.audio(voice_path)
            process_voice("capture", voice_path)
            st.session_state["voice_capture_active"] = False

    # Upload audio
    if st.session_state.get("voice_upload_active"):
        uploaded = st.file_uploader("Upload a WAV clip", type=["wav", "mp3", "ogg", "m4a"])
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                voice_path = tmp.name
            st.audio(voice_path)
            process_voice("upload", voice_path)
            st.session_state["voice_upload_active"] = False

    # Show current voice stats summary
    if st.session_state["voice_stats"] is not None:
        src = st.session_state.get("voice_source", "unknown")
        _v = st.session_state["voice_stats"]
        st.caption(
            f"Source: {src} · Jitter: {_v.get('jitter_proxy', 0):.4f} · "
            f"Shimmer: {_v.get('shimmer_proxy', 0):.4f}"
        )

# --- Gait Tab ---
with tab_gait:
    st.subheader("Gait Biomarker Analysis")

    col_cap, col_up, col_test = st.columns(3)
    with col_cap:
        if st.button("📹 Capture", key="gait_capture", use_container_width=True):
            st.session_state["gait_capture_active"] = True
    with col_up:
        if st.button("📁 Upload", key="gait_upload", use_container_width=True):
            st.session_state["gait_upload_active"] = True
    with col_test:
        if st.button("🧪 Test Data", key="gait_test", use_container_width=True):
            process_gait("test", None)

    # Capture still (camera_input captures a photo, not video)
    if st.session_state.get("gait_capture_active"):
        st.info("📹 Take a walking snapshot below. For full gait tracking, upload a short video.")
        video_value = st.camera_input("Capture walking snapshot")
        if video_value is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(video_value.read())
                gait_path = tmp.name
            st.success("✅ Snapshot captured!")
            st.image(gait_path)
            process_gait("capture", gait_path)
            st.session_state["gait_capture_active"] = False

    # Upload video
    if st.session_state.get("gait_upload_active"):
        uploaded = st.file_uploader("Upload a walking video", type=["mp4", "avi", "mov", "webm"])
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                gait_path = tmp.name
            st.video(gait_path)
            process_gait("upload", gait_path)
            st.session_state["gait_upload_active"] = False

    # Show current gait stats summary
    if st.session_state["gait_stats"] is not None:
        src = st.session_state.get("gait_source", "unknown")
        _g = st.session_state["gait_stats"]
        st.caption(
            f"Source: {src} · Asymmetry: {_g.get('asymmetry_index', 0)}° · "
            f"Step Freq: {_g.get('step_frequency', 0)} Hz"
        )

# --- My Reports Tab ---
with tab_history:
    st.subheader("My Reports")
    if st.session_state["user"] is None:
        st.info("Sign in from the sidebar to save reports and track progress over time.")
    else:
        reports = db.list_reports(st.session_state["user"]["id"])
        if not reports:
            st.info("No saved reports yet. Run an analysis and save it to start tracking progress.")
        else:
            st.caption(f"{len(reports)} saved report(s)")

            ages = [r.get("biological_age") for r in reports if r.get("biological_age") is not None]
            if len(ages) >= 2:
                fig = go.Figure(go.Scatter(
                    y=list(reversed(ages)),
                    mode="lines+markers",
                    line=dict(color="#6366f1"),
                ))
                fig.update_layout(
                    title="Biological Age Over Time",
                    xaxis_title="Report (oldest → newest)",
                    yaxis_title="Estimated Age",
                )
                st.plotly_chart(fig, use_container_width=True)

            for r in reports:
                label = f"{r.get('created_at', '')[:19]} — Age {r.get('biological_age', 'N/A')} ({r.get('frailty_indicator', 'n/a')})"
                with st.expander(label):
                    c1, c2 = st.columns(2)
                    c1.metric("Biological Age", r.get("biological_age", "N/A"))
                    c2.metric("Frailty", str(r.get("frailty_indicator", "n/a")).title())
                    if r.get("anomalies"):
                        st.write("**Anomalies**")
                        for a in r["anomalies"]:
                            st.warning(a)
                    if r.get("recommendations"):
                        st.write("**Recommendations**")
                        for rec in r["recommendations"]:
                            st.success(rec)

# --- Evaluation Section (only visible when data exists) ---
st.divider()

voice_ready = st.session_state["voice_stats"] is not None
gait_ready = st.session_state["gait_stats"] is not None

if voice_ready or gait_ready:
    st.subheader("📊 Biological Age Evaluation")

    if st.button("🧬 Analyze Biomarkers", type="primary", key="eval_btn"):
        voice_stats = st.session_state.get("voice_stats") or extract_voice_features(None)
        gait_stats = st.session_state.get("gait_stats") or extract_gait_features(None)

        with st.spinner("Agent evaluating biomarkers..."):
            evaluation = evaluate_biomarkers(voice_stats, gait_stats)
        st.session_state["evaluation"] = evaluation

        pdf_path = generate_pdf(voice_stats, gait_stats, evaluation)
        st.session_state["pdf_path"] = pdf_path

        # Opt-in anonymized contribution (no identity, no raw media)
        if share_anonymized:
            if db.log_anonymized(
                voice_stats, gait_stats, evaluation,
                st.session_state.get("voice_source"),
                st.session_state.get("gait_source"),
            ):
                st.session_state["telemetry_logged"] = True

    # Show results if evaluation has been run
    if st.session_state["evaluation"] is not None:
        evaluation = st.session_state["evaluation"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Biological Age", f"{evaluation.get('biological_age_estimate', 'N/A')}")
        col2.metric("Frailty Indicator", str(evaluation.get("frailty_indicator", "N/A")).title())
        col3.metric("Anomalies", len(evaluation.get("anomalies", [])))

        st.write("**Detected Anomalies:**")
        for a in evaluation.get("anomalies", []):
            st.warning(a)

        st.write("**Agent Recommendations:**")
        for r in evaluation.get("recommendations", []):
            st.success(r)

        if st.session_state.get("telemetry_logged"):
            st.caption("📊 Anonymized result contributed — thank you.")

        col_dl, col_save = st.columns(2)

        with col_dl:
            if st.session_state["pdf_path"] and os.path.exists(st.session_state["pdf_path"]):
                with open(st.session_state["pdf_path"], "rb") as f:
                    st.download_button(
                        "📄 Download Diagnostic Summary PDF",
                        data=f,
                        file_name="chronosense_report.pdf",
                        mime="application/pdf",
                        key="pdf_download",
                    )

        with col_save:
            if st.session_state["user"] is None:
                st.caption("Sign in to save this report.")
            elif st.session_state.get("saved_report_id"):
                st.success("Report saved to My Reports.")
            elif st.button("💾 Save to My Reports", use_container_width=True):
                report_id = db.save_report(
                    st.session_state["user"]["id"],
                    st.session_state.get("voice_stats") or {},
                    st.session_state.get("gait_stats") or {},
                    evaluation,
                    st.session_state.get("voice_source"),
                    st.session_state.get("gait_source"),
                )
                if report_id:
                    st.session_state["saved_report_id"] = report_id
                    st.success("Report saved to My Reports.")
                else:
                    st.error("Could not save report.")
else:
    st.info("👆 Add voice and/or gait data above to run the biomarker analysis.")
