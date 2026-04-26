import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from utils.preprocessing import preprocess_image
from PIL import Image
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* ── Root variables ── */
:root {
    --neon-cyan:   #00f5ff;
    --neon-pink:   #ff2d78;
    --neon-green:  #39ff14;
    --neon-orange: #ff8c00;
    --bg-dark:     #080c14;
    --glass:       rgba(255,255,255,0.04);
    --glass-border:rgba(255,255,255,0.10);
}

/* ── Full-page dark background with moving mesh ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 20%, rgba(0,245,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 80% 80%, rgba(255,45,120,0.08) 0%, transparent 60%);
    color: #c9d6e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0,0,0,0.18) 3px,
        rgba(0,0,0,0.18) 4px
    );
    pointer-events: none;
    z-index: 9999;
    animation: scanMove 8s linear infinite;
}
@keyframes scanMove {
    from { background-position: 0 0; }
    to   { background-position: 0 100px; }
}

/* ── Title ── */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 900;
    text-align: center;
    margin: 1.5rem 0 0.4rem;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink), var(--neon-cyan));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    letter-spacing: 0.08em;
}
@keyframes shimmer {
    to { background-position: 200% center; }
}

.hero-sub {
    text-align: center;
    color: rgba(160,200,220,0.55);
    font-size: 0.82rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(0,245,255,0.3) !important;
    border-radius: 12px !important;
    background: var(--glass) !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 18px rgba(0,245,255,0.15);
}
[data-testid="stFileUploaderDropzone"] label {
    color: rgba(160,200,220,0.6) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── File name badge ── */
.file-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    background: linear-gradient(135deg, rgba(0,245,255,0.07), rgba(255,45,120,0.07));
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    padding: 14px 20px;
    margin: 1.4rem 0;
    animation: slideIn 0.5s cubic-bezier(0.22,1,0.36,1);
    backdrop-filter: blur(8px);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.file-icon {
    font-size: 1.6rem;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); }
    50%      { transform: scale(1.12); }
}
.file-info { flex: 1; }
.file-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--neon-cyan);
    letter-spacing: 0.05em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 320px;
}
.file-label {
    font-size: 0.68rem;
    color: rgba(160,200,220,0.4);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-top: 3px;
}
.status-dot {
    width: 8px; height: 8px;
    background: var(--neon-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--neon-green);
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.2; }
}

/* ── Analyse button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(255,45,120,0.12)) !important;
    border: 1.5px solid rgba(0,245,255,0.4) !important;
    border-radius: 8px !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    padding: 0.7rem 1.2rem !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,245,255,0.22), rgba(255,45,120,0.22)) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.25), 0 0 40px rgba(255,45,120,0.12) !important;
    transform: translateY(-2px) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-top: 1.6rem;
    animation: popIn 0.6s cubic-bezier(0.22,1,0.36,1);
    position: relative;
    overflow: hidden;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
}
.result-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: inherit;
    filter: blur(24px);
    opacity: 0.35;
    z-index: -1;
}
.result-real      { background: linear-gradient(135deg, rgba(57,255,20,0.15),  rgba(0,200,100,0.08));  border: 1.5px solid rgba(57,255,20,0.4); }
.result-fake      { background: linear-gradient(135deg, rgba(255,45,120,0.18), rgba(200,0,60,0.08));   border: 1.5px solid rgba(255,45,120,0.4); }
.result-uncertain { background: linear-gradient(135deg, rgba(255,140,0,0.15),  rgba(200,100,0,0.08));  border: 1.5px solid rgba(255,140,0,0.4); }

.result-verdict {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.5rem, 3.5vw, 2rem);
    font-weight: 900;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.verdict-real      { color: var(--neon-green); text-shadow: 0 0 16px var(--neon-green); }
.verdict-fake      { color: var(--neon-pink);  text-shadow: 0 0 16px var(--neon-pink); }
.verdict-uncertain { color: var(--neon-orange);text-shadow: 0 0 16px var(--neon-orange); }

.result-conf {
    font-size: 0.78rem;
    color: rgba(160,200,220,0.5);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 18px;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin: 0 auto;
    max-width: 320px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    animation: growBar 1s cubic-bezier(0.22,1,0.36,1);
}
@keyframes growBar {
    from { width: 0%; }
}
.bar-real      { background: linear-gradient(90deg, #39ff14, #00c864); box-shadow: 0 0 10px #39ff1466; }
.bar-fake      { background: linear-gradient(90deg, #ff2d78, #c8003c); box-shadow: 0 0 10px #ff2d7866; }
.bar-uncertain { background: linear-gradient(90deg, #ff8c00, #c86400); box-shadow: 0 0 10px #ff8c0066; }

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: var(--neon-cyan) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--neon-cyan) !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Hide Streamlit footer ── */
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero header ────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>⬡ AI IMAGE DETECTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Neural scan • Real vs AI-generated analysis</div>", unsafe_allow_html=True)

# ─── Load model (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    state = torch.load("model/model_checkpoint.pth", map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

model = load_model()

# ─── File uploader ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop an image file here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # File name badge (no preview image)
    fname = uploaded_file.name
    ext   = fname.rsplit(".", 1)[-1].upper()
    st.markdown(f"""
    <div class="file-badge">
        <div class="file-icon">🖼️</div>
        <div class="file-info">
            <div class="file-name">{fname}</div>
            <div class="file-label">{ext} &nbsp;•&nbsp; Ready to scan</div>
        </div>
        <div class="status-dot"></div>
    </div>
    """, unsafe_allow_html=True)

    # Analyse button
    if st.button("⟳  RUN NEURAL SCAN"):
        with st.spinner("Scanning image through neural network…"):
            time.sleep(1)
            image_tensor = preprocess_image(uploaded_file)
            with torch.no_grad():
                output = model(image_tensor)
                probs  = torch.softmax(output, dim=1)[0]
                confidence, predicted_class = torch.max(probs, dim=0)

        conf_val   = confidence.item()
        pred_label = "Fake" if predicted_class.item() == 0 else "Real"
        if conf_val < 0.70:
            pred_label = "Uncertain"

        # Map to CSS classes
        card_cls    = {"Real": "result-real",      "Fake": "result-fake",      "Uncertain": "result-uncertain"}[pred_label]
        verdict_cls = {"Real": "verdict-real",      "Fake": "verdict-fake",     "Uncertain": "verdict-uncertain"}[pred_label]
        bar_cls     = {"Real": "bar-real",          "Fake": "bar-fake",         "Uncertain": "bar-uncertain"}[pred_label]
        icon        = {"Real": "✅",                 "Fake": "⚠️",               "Uncertain": "🔶"}[pred_label]
        bar_pct     = int(conf_val * 100)

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="result-verdict {verdict_cls}">{icon} &nbsp;{pred_label.upper()}</div>
            <div class="result-conf">Confidence &nbsp;·&nbsp; {conf_val:.2%}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill {bar_cls}" style="width:{bar_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)