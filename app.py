"""ChronoSense AI — Streamlit dashboard.

Redesigned UX: [Capture][Upload][Test Data] buttons per tab,
automatic processing when data is ready, evaluation section that
clears when sample changes and only appears when there is data.
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

st.set_page_config(page_title="ChronoSense AI", page_icon="🧬", layout="wide")

st.title("🧬 ChronoSense AI — Longevity Biomarker Diagnostics Agent")
st.caption("Voice & Gait Biomarker Analysis · Sundai Hack 143 · Biomarkers of Aging")

# --- Session state initialization ---
if "voice_stats" not in st.session_state:
    st.session_state["voice_stats"] = None
if "gait_stats" not in st.session_state:
    st.session_state["gait_stats"] = None
if "evaluation" not in st.session_state:
    st.session_state["evaluation"] = None
if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None
if "voice_source" not in st.session_state:
    st.session_state["voice_source"] = None  # "capture", "upload", "test"
if "gait_source" not in st.session_state:
    st.session_state["gait_source"] = None


def clear_evaluation():
    """Clear evaluation results when sample changes."""
    st.session_state["evaluation"] = None
    st.session_state["pdf_path"] = None


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


# --- Voice Tab ---
tab_voice, tab_gait = st.tabs(["🎙️ Voice Diagnostic", "🚶 Gait Track"])

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
        st.caption(f"Source: {src} · Jitter: {st.session_state['voice_stats'].get('jitter_proxy', 'N/A'):.4f} · Shimmer: {st.session_state['voice_stats'].get('shimmer_proxy', 'N/A'):.4f}")

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

    # Capture video
    if st.session_state.get("gait_capture_active"):
        st.info("📹 Record a walking video below")
        video_value = st.camera_input("Record gait video")
        if video_value is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_value.read())
                gait_path = tmp.name
            st.success("✅ Video captured!")
            st.video(gait_path)
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
        st.caption(f"Source: {src} · Asymmetry: {st.session_state['gait_stats'].get('asymmetry_index', 'N/A')}° · Step Freq: {st.session_state['gait_stats'].get('step_frequency', 'N/A')} Hz")

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

    # Show results if evaluation has been run
    if st.session_state["evaluation"] is not None:
        evaluation = st.session_state["evaluation"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Biological Age", f"{evaluation.get('biological_age_estimate', 'N/A')}")
        col2.metric("Frailty Indicator", evaluation.get("frailty_indicator", "N/A").title())
        col3.metric("Anomalies", len(evaluation.get("anomalies", [])))

        st.write("**Detected Anomalies:**")
        for a in evaluation.get("anomalies", []):
            st.warning(a)

        st.write("**Agent Recommendations:**")
        for r in evaluation.get("recommendations", []):
            st.success(r)

        if st.session_state["pdf_path"] and os.path.exists(st.session_state["pdf_path"]):
            with open(st.session_state["pdf_path"], "rb") as f:
                st.download_button(
                    "📄 Download Diagnostic Summary PDF",
                    data=f,
                    file_name="chronosense_report.pdf",
                    mime="application/pdf",
                    key="pdf_download",
                )
else:
    st.info("👆 Add voice and/or gait data above to run the biomarker analysis.")
