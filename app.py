"""ChronoSense AI — Streamlit dashboard.

Two tabs: Voice Diagnostic and Gait Track. Each can process real media
or use mock data for demo purposes.
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

# Sidebar
st.sidebar.header("⚙️ Options")
use_mock_voice = st.sidebar.checkbox("Use mock voice data (no mic)", value=True)
use_mock_gait = st.sidebar.checkbox("Use mock gait data (no camera)", value=True)

# Webcam capture toggle
enable_webcam = st.sidebar.checkbox("📷 Enable webcam capture", value=False)

tab_voice, tab_gait = st.tabs(["🎙️ Voice Diagnostic", "🚶 Gait Track"])

# --- Voice Tab ---
with tab_voice:
    st.subheader("Voice Biomarker Analysis")
    voice_file = None
    if not use_mock_voice:
        # Webcam/mic capture
        if enable_webcam:
            st.info("🎤 Click 'Start' to record 30 seconds of audio from your microphone")
            audio_value = st.audio_input("Record voice sample")
            if audio_value is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_value.read())
                    voice_file = tmp.name
                st.success("✅ Voice captured!")
                st.audio(voice_file)
        
        # File upload (always available)
        uploaded = st.file_uploader(
            "Or upload a 30-second WAV clip", type=["wav", "mp3", "ogg", "m4a"]
        )
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                voice_file = tmp.name
            st.audio(voice_file)

    if st.button("🔘 Process Acoustic Biomarkers", key="voice_btn"):
        with st.spinner("Extracting voice features..."):
            voice_stats = extract_voice_features(voice_file)
        st.session_state["voice_stats"] = voice_stats

        if voice_stats.get("is_mock"):
            import numpy as np
            t = np.linspace(0, 3, 22050)
            y = 0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t)
            fig = go.Figure(go.Scatter(x=t, y=y, line=dict(color="#6366f1")))
            fig.update_layout(title="Mock Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True)
        elif voice_file is not None:
            # Plot real waveform
            import librosa
            import numpy as np
            y, sr = librosa.load(voice_file, sr=22050)
            t = np.linspace(0, len(y) / sr, len(y))
            fig = go.Figure(go.Scatter(x=t, y=y, line=dict(color="#6366f1")))
            fig.update_layout(title="Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True)

        st.json(voice_stats)

# --- Gait Tab ---
with tab_gait:
    st.subheader("Gait Biomarker Analysis")
    gait_file = None
    if not use_mock_gait:
        # Webcam video capture
        if enable_webcam:
            st.info("📹 Click 'Start' to record a walking video from your webcam")
            video_value = st.camera_input("Record gait video")
            if video_value is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_value.read())
                    gait_file = tmp.name
                st.success("✅ Video captured!")
                st.video(gait_file)

        # File upload (always available)
        uploaded = st.file_uploader(
            "Or upload a walking video (lateral view)", type=["mp4", "avi", "mov", "webm"]
        )
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                gait_file = tmp.name
            st.video(gait_file)

    if st.button("🔘 Map Kinetic Deviations", key="gait_btn"):
        with st.spinner("Processing gait video..."):
            gait_stats = extract_gait_features(gait_file)
        st.session_state["gait_stats"] = gait_stats

        if gait_stats.get("left_knee_angles"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=gait_stats["left_knee_angles"], name="Left Knee", line=dict(color="#6366f1")))
            fig.add_trace(go.Scatter(y=gait_stats["right_knee_angles"], name="Right Knee", line=dict(color="#f59e0b")))
            fig.update_layout(title="Knee Joint Angles Over Time", xaxis_title="Frame", yaxis_title="Angle (°)")
            st.plotly_chart(fig, use_container_width=True)

        st.json(gait_stats)

# --- Evaluation ---
st.divider()
st.subheader("📊 Agent Biological Age Evaluation")

# Initialize session state for evaluation results
if "evaluation" not in st.session_state:
    st.session_state["evaluation"] = None
if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None

if st.button("🧠 Run Agent Evaluation", type="primary", key="eval_btn"):
    voice_stats = st.session_state.get("voice_stats", extract_voice_features(None))
    gait_stats = st.session_state.get("gait_stats", extract_gait_features(None))

    with st.spinner("Agent evaluating biomarkers..."):
        evaluation = evaluate_biomarkers(voice_stats, gait_stats)
    st.session_state["evaluation"] = evaluation

    # Generate PDF immediately
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

    # Download button — always visible after evaluation
    if st.session_state["pdf_path"] and os.path.exists(st.session_state["pdf_path"]):
        with open(st.session_state["pdf_path"], "rb") as f:
            st.download_button(
                "📄 Download Diagnostic Summary PDF",
                data=f,
                file_name="chronosense_report.pdf",
                mime="application/pdf",
                key="pdf_download",
            )
