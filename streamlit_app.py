import os
import sys
import json
import tempfile
import time

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Complaint Auto-Routing System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark header band */
.main-header {
    background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #00d4aa;
}
.main-header h1 {
    color: #e8f4f0;
    font-size: 1.9rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7fbfaf;
    margin: 0.4rem 0 0;
    font-size: 0.9rem;
}

/* Metric cards */
.metric-card {
    background: #0f1923;
    border: 1px solid #1e3040;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    height: 100%;
}
.metric-card .label {
    color: #7fbfaf;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8f4f0;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #4a7060;
    margin-top: 0.3rem;
}

/* Priority badges */
.badge-high   { background:#ff4757; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.badge-medium { background:#ffa502; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.badge-low    { background:#2ed573; color:#000; padding:3px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }

/* Similar complaint row */
.sim-row {
    background: #0f1923;
    border: 1px solid #1e3040;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.sim-row .sim-text { color: #c8ddd8; font-size: 0.87rem; }
.sim-row .sim-meta { color: #4a7060; font-size: 0.75rem; margin-top: 0.3rem; font-family: 'IBM Plex Mono', monospace; }
.sim-score { float:right; background:#1e3040; color:#00d4aa; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-family:'IBM Plex Mono',monospace;}

/* Officer card */
.officer-card {
    background: linear-gradient(135deg, #0d2233 0%, #0f1923 100%);
    border: 1px solid #00d4aa33;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}
.officer-card .name { color:#00d4aa; font-size:1.15rem; font-weight:600; }
.officer-card .domain { color:#4a7060; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase; margin-top:0.2rem; }

/* Transcription box */
.transcribed-box {
    background:#0a1520;
    border:1px dashed #1e3040;
    border-radius:8px;
    padding:1rem 1.2rem;
    color:#7fbfaf;
    font-size:0.88rem;
    font-family:'IBM Plex Mono',monospace;
    margin-top:0.5rem;
}

/* Step label */
.step-label {
    font-size:0.7rem;
    color:#4a7060;
    letter-spacing:2px;
    text-transform:uppercase;
    font-weight:600;
    margin-bottom:0.3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:#0d1822;
}
</style>
""", unsafe_allow_html=True)


# ── Bootstrap: auto-train if models missing ───────────────────────────────────
def ensure_models():
    from src.inference import get_engine
    engine = get_engine()
    if not engine.models_exist():
        with st.spinner("⚙️ First-time setup: training ML models (takes ~30s)..."):
            from src.bootstrap import run_bootstrap
            run_bootstrap()
        st.success("✅ Models trained and ready!")
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🏛️ System Info")
        st.caption("Complaint Auto-Routing System")
        st.markdown("---")

        st.markdown("**Models**")
        models_dir = "models"
        model_files = {
            "Priority Classifier": "priority_model.pkl",
            "ETA Regressor":       "eta_model.pkl",
            "Officer Router":      "officer_model.pkl",
            "Similarity Index":    "similarity_vectorizer.pkl",
        }
        for label, fname in model_files.items():
            exists = os.path.exists(f"{models_dir}/{fname}")
            icon = "🟢" if exists else "🔴"
            st.markdown(f"{icon} {label}")

        # Show eval metrics if available
        metrics_path = f"{models_dir}/eval_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.markdown("---")
            st.markdown("**Eval Metrics**")
            p = metrics.get("priority", {})
            e = metrics.get("eta", {})
            o = metrics.get("officer", {})
            s = metrics.get("similarity", {})
            st.caption(f"Priority F1: `{p.get('f1_weighted','—')}`")
            st.caption(f"ETA MAE: `{e.get('mae_days','—')} days`")
            st.caption(f"Officer F1: `{o.get('f1_weighted','—')}`")
            st.caption(f"Recall@5: `{s.get('recall_at_5','—')}`")

        st.markdown("---")
        st.markdown("**Accepted Inputs**")
        st.caption("📝 Text (any language)")
        st.caption("🎙️ Audio (mp3/wav/m4a)")
        st.caption("🎥 Video (mp4/mov)")
        st.markdown("---")
        st.markdown("**Retrain Models**")
        if st.button("🔄 Retrain from scratch"):
            _force_retrain()

        st.markdown("---")
        st.caption("No external APIs. Fully offline.")


def _force_retrain():
    import shutil
    if os.path.exists("models"):
        shutil.rmtree("models")
    with st.spinner("Retraining..."):
        from src.bootstrap import run_bootstrap
        run_bootstrap()
    st.success("Retrained!")
    st.rerun()


# ── Result rendering ──────────────────────────────────────────────────────────
def render_results(result):
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    # Row 1: 4 metric cards
    c1, c2, c3, c4 = st.columns(4)

    priority_color = {"High": "#ff4757", "Medium": "#ffa502", "Low": "#2ed573"}.get(result.priority, "#ccc")
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Priority</div>
            <div class="value" style="color:{priority_color}">{result.priority}</div>
            <div class="sub">Confidence: {result.priority_confidence*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">ETA</div>
            <div class="value">{result.eta_days:.0f}d</div>
            <div class="sub">Expected resolution</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Assigned Officer</div>
            <div class="value" style="font-size:1rem;padding-top:4px">{result.officer_name}</div>
            <div class="sub">{result.officer_domain.replace('_',' ').title()}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        sim_score = result.similar_complaints[0]["similarity_score"] if result.similar_complaints else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Top Match Score</div>
            <div class="value">{sim_score:.2f}</div>
            <div class="sub">Cosine similarity</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Officer card + Similar complaints
    col_officer, col_similar = st.columns([1, 2])

    with col_officer:
        st.markdown("#### 👤 Assigned Officer")
        st.markdown(f"""
        <div class="officer-card">
            <div class="name">{result.officer_name}</div>
            <div class="domain">🏷️ {result.officer_domain.replace('_', ' ')}</div>
            <br>
            <div style="color:#7fbfaf;font-size:0.83rem;">Officer ID: <code style="color:#00d4aa">{result.officer_id}</code></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Priority Breakdown")
        priority_val = {"High": 1.0, "Medium": 0.6, "Low": 0.25}.get(result.priority, 0.5)
        badge_cls = f"badge-{result.priority.lower()}"
        st.markdown(f'<span class="{badge_cls}">{result.priority} Priority</span>', unsafe_allow_html=True)
        st.progress(priority_val)
        st.caption(f"Model confidence: **{result.priority_confidence*100:.1f}%**")

    with col_similar:
        st.markdown("#### 🔍 Similar Past Complaints")
        if result.similar_complaints:
            for item in result.similar_complaints:
                p_badge = f"badge-{item['priority'].lower()}"
                st.markdown(f"""
                <div class="sim-row">
                    <span class="sim-score">{item['similarity_score']:.3f}</span>
                    <div class="sim-text">{item['text'][:140]}{'…' if len(item['text'])>140 else ''}</div>
                    <div class="sim-meta">
                        ID: {item['complaint_id']} &nbsp;|&nbsp;
                        <span class="{p_badge}" style="padding:1px 7px;font-size:0.7rem">{item['priority']}</span>
                        &nbsp;|&nbsp; ETA: {item['eta_days']}d &nbsp;|&nbsp; {item['officer_name']}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No similar complaints found in the store.")


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    ensure_models()
    render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Complaint Auto-Routing System</h1>
        <p>ML-powered routing · Priority prediction · ETA estimation · Semantic similarity</p>
    </div>""", unsafe_allow_html=True)

    # Input method tabs
    tab_text, tab_audio, tab_video = st.tabs(["📝 Text", "🎙️ Audio", "🎥 Video"])

    complaint_text = None
    transcribed_from = None

    # ── Text tab ──────────────────────────────────────────────────────────────
    with tab_text:
        st.markdown('<div class="step-label">Enter Complaint</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            label="",
            placeholder="Describe your complaint in any language. E.g., 'The road in sector 7 has a large pothole causing accidents daily...'",
            height=140,
            key="text_input",
            label_visibility="collapsed",
        )
        col_btn, col_ex = st.columns([1, 3])
        with col_btn:
            text_submit = st.button("🚀 Analyse Complaint", key="text_btn", type="primary")
        with col_ex:
            example_options = [
                "Select an example...",
                "Sewage water is mixing with drinking water supply in our area for 3 days now",
                "Power outage since 12 hours, transformer sparking at main junction",
                "Bribe demanded by officer for property registration",
                "Garbage not collected for 2 weeks near sector 4, health hazard",
                "Mere ghar ke paas ki sadak mein bahut bada gadha hai jo haadsa ka karan ban raha hai",
            ]
            example = st.selectbox("", example_options, key="example_sel", label_visibility="collapsed")
            if example != "Select an example...":
                st.session_state["prefill"] = example

        if "prefill" in st.session_state and st.session_state["prefill"]:
            complaint_text = st.session_state.pop("prefill")
            st.rerun()

        if text_submit and text_input.strip():
            complaint_text = text_input.strip()

    # ── Audio tab ─────────────────────────────────────────────────────────────
    with tab_audio:
        st.markdown('<div class="step-label">Upload Audio Complaint</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "", type=["mp3", "wav", "m4a", "ogg", "flac"],
            key="audio_upload", label_visibility="collapsed"
        )
        whisper_size = st.selectbox("Whisper model size", ["tiny", "base", "small"], index=1, key="whisper_audio")
        audio_submit = st.button("🚀 Transcribe & Analyse", key="audio_btn", type="primary")

        if audio_submit and audio_file:
            try:
                from src.transcription import transcribe_uploaded_file, _whisper_available
                if not _whisper_available():
                    st.error("openai-whisper not installed. Run: `pip install openai-whisper`")
                else:
                    with tempfile.NamedTemporaryFile(suffix=f".{audio_file.name.split('.')[-1]}", delete=False) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    with st.spinner("Transcribing audio..."):
                        transcribed = transcribe_uploaded_file(tmp_path, audio_file.name, whisper_size)
                    os.unlink(tmp_path)
                    complaint_text = transcribed
                    transcribed_from = "audio"
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    # ── Video tab ─────────────────────────────────────────────────────────────
    with tab_video:
        st.markdown('<div class="step-label">Upload Video Complaint</div>', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "", type=["mp4", "avi", "mov", "mkv", "webm"],
            key="video_upload", label_visibility="collapsed"
        )
        whisper_size_v = st.selectbox("Whisper model size", ["tiny", "base", "small"], index=1, key="whisper_video")
        video_submit = st.button("🚀 Extract & Analyse", key="video_btn", type="primary")

        if video_submit and video_file:
            try:
                from src.transcription import transcribe_uploaded_file, _whisper_available, _moviepy_available
                if not _whisper_available():
                    st.error("openai-whisper not installed. Run: `pip install openai-whisper`")
                elif not _moviepy_available():
                    st.error("moviepy not installed. Run: `pip install moviepy`")
                else:
                    with tempfile.NamedTemporaryFile(suffix=f".{video_file.name.split('.')[-1]}", delete=False) as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name
                    with st.spinner("Extracting audio and transcribing..."):
                        transcribed = transcribe_uploaded_file(tmp_path, video_file.name, whisper_size_v)
                    os.unlink(tmp_path)
                    complaint_text = transcribed
                    transcribed_from = "video"
            except Exception as e:
                st.error(f"Video processing failed: {e}")

    # ── Run inference ─────────────────────────────────────────────────────────
    if complaint_text:
        if transcribed_from:
            st.markdown(f"""
            <div class="transcribed-box">
                <div style="color:#00d4aa;font-size:0.7rem;letter-spacing:2px;margin-bottom:0.4rem;">
                    TRANSCRIBED FROM {transcribed_from.upper()}
                </div>
                {complaint_text}
            </div>""", unsafe_allow_html=True)

        if not complaint_text.strip():
            st.warning("Transcription produced empty text. Please try again.")
            return

        with st.spinner("Running ML inference..."):
            try:
                from src.inference import get_engine
                engine = get_engine()
                result = engine.predict(complaint_text.strip())
                render_results(result)
            except FileNotFoundError:
                st.error("Models not found. Please retrain from the sidebar.")
            except Exception as e:
                st.error(f"Inference error: {e}")
                raise

    # ── Placeholder when no input ─────────────────────────────────────────────
    else:
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#2a4a40;">
            <div style="font-size:3rem;margin-bottom:1rem;">🏛️</div>
            <div style="font-size:1.1rem;color:#4a7060;">Submit a complaint above to see AI-powered routing</div>
            <div style="font-size:0.8rem;margin-top:0.5rem;color:#2a4a40;">
                Priority · ETA · Officer Assignment · Semantic Similarity
            </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
