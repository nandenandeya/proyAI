"""
FocusWebCam - Deteksi Fokus Wajah dengan AI
============================================
Streamlit Community Cloud compatible version
- MediaPipe FaceLandmarker API baru (>=0.10)
- streamlit-webrtc untuk real-time webcam
- Logistic Regression (tanpa sklearn dependency)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import urllib.request
import threading
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# MediaPipe API baru (>=0.10)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ============================================================================
# HALAMAN CONFIG — HARUS BARIS PERTAMA STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="FocusWebCam",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_COEF = {
    'ear': 1.0494,
    'head_pose': -2.6625,
    'mouth_ratio': 2.0005
}
MODEL_INTERCEPT = -0.5234

SCALER_PARAMS = {
    'ear':         {'mean': 0.214,  'std': 0.098},
    'head_pose':   {'mean': 0.178,  'std': 0.245},
    'mouth_ratio': {'mean': 0.068,  'std': 0.082}
}

CLASSIFICATION_THRESHOLD = 0.5

LANDMARKS = {
    'LEFT_EYE':     [33, 133, 155, 154, 153, 145],
    'RIGHT_EYE':    [362, 263, 387, 386, 385, 373],
    'MOUTH_TOP':    13,
    'MOUTH_BOTTOM': 14,
    'MOUTH_LEFT':   78,
    'MOUTH_RIGHT':  308,
    'NOSE_TIP':     1,
    'FACE_LEFT':    234,
    'FACE_RIGHT':   454
}

CONFIG = {
    'ALERT_THRESHOLD':  40,
    'FOCUS_THRESHOLD':  65,
    'SMOOTHING_WINDOW': 5,
    'EAR_OPEN':         0.25,
    'EAR_CLOSED':       0.15
}

# STUN server wajib untuk Streamlit Cloud (TURN baru meningkatkan keandalan)
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #0d0d1a 0%, #12122a 50%, #0d1a1a 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,255,136,0.3);
        box-shadow: 0 0 40px rgba(0,255,136,0.08);
    }
    .main-header h1 {
        font-family: 'Space Mono', monospace;
        color: #00ff88;
        font-size: 2rem;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(0,255,136,0.4);
    }
    .main-header p { color: #6b7280; font-size: 0.9rem; margin: 0.2rem 0; }
    .main-header .tags { color: #374151; font-size: 0.75rem; margin-top: 0.5rem; }

    .explanation-box {
        background: rgba(22,33,62,0.8);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #ffcc00;
        font-size: 0.9rem;
    }
    .metric-focus   { color: #00ff88; font-weight: 700; }
    .metric-warning { color: #ffcc00; font-weight: 700; }
    .metric-danger  { color: #ff4444; font-weight: 700; }

    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .badge-focus   { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.4); }
    .badge-warning { background: rgba(255,204,0,0.15);  color: #ffcc00; border: 1px solid rgba(255,204,0,0.4); }
    .badge-danger  { background: rgba(255,68,68,0.15);  color: #ff4444; border: 1px solid rgba(255,68,68,0.4); }

    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: #374151;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1f2937;
        font-family: 'Space Mono', monospace;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL DOWNLOAD (cached — hanya sekali per instance)
# ============================================================================

@st.cache_resource(show_spinner=False)
def download_model() -> str:
    """Download MediaPipe FaceLandmarker .task file jika belum ada."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Mengunduh model MediaPipe (~10 MB) — hanya sekali…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FaceFeatureExtractor:
    """
    Mengekstrak 3 fitur geometrik wajah:
      1. EAR  — Eye Aspect Ratio (keterbukaan mata)
      2. Head Pose — seberapa kepala menoleh dari sumbu kamera
      3. Mouth Ratio — rasio bukaan mulut
    """

    def __init__(self, model_path: str = MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    # ── helpers ────────────────────────────────────────────────────

    def _ear(self, lm, eye_idx, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in eye_idx]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    def _head_pose(self, lm, w, h):
        nose_x  = lm[LANDMARKS['NOSE_TIP']].x   * w
        left_x  = lm[LANDMARKS['FACE_LEFT']].x  * w
        right_x = lm[LANDMARKS['FACE_RIGHT']].x * w
        center  = (left_x + right_x) / 2.0
        fw      = abs(right_x - left_x)
        return min(abs(nose_x - center) / fw, 0.5) if fw else 0.0

    def _mouth(self, lm, w, h):
        top    = lm[LANDMARKS['MOUTH_TOP']]
        bottom = lm[LANDMARKS['MOUTH_BOTTOM']]
        left   = lm[LANDMARKS['MOUTH_LEFT']]
        right  = lm[LANDMARKS['MOUTH_RIGHT']]
        vert   = abs((top.y - bottom.y) * h)
        horiz  = abs((left.x - right.x) * w)
        return min(vert / horiz, 0.5) if horiz else 0.0

    # ── public ─────────────────────────────────────────────────────

    def extract(self, rgb_image: np.ndarray) -> dict:
        h, w = rgb_image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result   = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return {'face_detected': False, 'ear': 0.0,
                    'head_pose': 0.0, 'mouth_ratio': 0.0, 'landmarks': None}

        lm = result.face_landmarks[0]
        ear_l = self._ear(lm, LANDMARKS['LEFT_EYE'],  w, h)
        ear_r = self._ear(lm, LANDMARKS['RIGHT_EYE'], w, h)

        return {
            'face_detected': True,
            'ear':         (ear_l + ear_r) / 2.0,
            'head_pose':   self._head_pose(lm, w, h),
            'mouth_ratio': self._mouth(lm, w, h),
            'landmarks':   lm
        }

    def release(self):
        self.detector.close()


# ============================================================================
# PREDICTOR (Logistic Regression tanpa sklearn)
# ============================================================================

class FocusPredictor:
    """
    Logistic Regression manual:
      P(fokus) = sigmoid(w1·EAR_norm + w2·Head_norm + w3·Mouth_norm + b)
    """

    def __init__(self):
        self._history: list = []

    def _norm(self, v, feat):
        p = SCALER_PARAMS[feat]
        return (v - p['mean']) / p['std']

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _smooth(self, score: float) -> int:
        self._history.append(score)
        if len(self._history) > CONFIG['SMOOTHING_WINDOW']:
            self._history.pop(0)
        return int(round(sum(self._history) / len(self._history)))

    def predict(self, ear: float, head_pose: float, mouth_ratio: float) -> dict:
        z = (MODEL_COEF['ear']        * self._norm(ear,        'ear') +
             MODEL_COEF['head_pose']   * self._norm(head_pose,  'head_pose') +
             MODEL_COEF['mouth_ratio'] * self._norm(mouth_ratio,'mouth_ratio') +
             MODEL_INTERCEPT)

        proba = self._sigmoid(z)
        score = self._smooth(proba * 100)
        label = 1 if proba >= CLASSIFICATION_THRESHOLD else 0

        contribs = self._contributions(ear, head_pose, mouth_ratio)
        expl     = self._explain(ear, head_pose, mouth_ratio, score)

        return {
            'label': label,
            'score': score,
            'probability': float(proba),
            'features': {'ear': ear, 'head_pose': head_pose, 'mouth_ratio': mouth_ratio},
            'contributions': contribs,
            'explanation': expl
        }

    def _contributions(self, ear, head_pose, mouth_ratio):
        ec = MODEL_COEF['ear']        * self._norm(ear,        'ear')
        hc = MODEL_COEF['head_pose']   * self._norm(head_pose,  'head_pose')
        mc = MODEL_COEF['mouth_ratio'] * self._norm(mouth_ratio,'mouth_ratio')
        total = abs(ec) + abs(hc) + abs(mc) or 1
        return {
            'ear': ec, 'head_pose': hc, 'mouth_ratio': mc,
            'ear_pct':   abs(ec) / total * 100,
            'head_pct':  abs(hc) / total * 100,
            'mouth_pct': abs(mc) / total * 100,
        }

    def _explain(self, ear, head_pose, mouth_ratio, score):
        ear_s   = ("baik (mata terbuka)"     if ear > 0.22
                   else "kurang (mata tertutup)" if ear < 0.18 else "cukup")
        head_s  = ("baik (menghadap kamera)" if head_pose < 0.08
                   else "kurang (menoleh)"   if head_pose > 0.15 else "cukup")
        mouth_s = ("baik (mulut tertutup)"   if mouth_ratio < 0.04
                   else "kurang (mulut terbuka)" if mouth_ratio > 0.08 else "cukup")

        if score >= CONFIG['FOCUS_THRESHOLD']:
            return {
                'level': 'FOKUS',
                'summary': f"Skor fokus {score}% — Kondisi sangat baik!",
                'details': f"Mata: {ear_s}, Kepala: {head_s}, Mulut: {mouth_s}",
                'suggestion': "Pertahankan kondisi ini."
            }
        elif score >= CONFIG['ALERT_THRESHOLD']:
            issues = ([f"mata cenderung tertutup"]   if ear < 0.18 else []) + \
                     ([f"kepala sering menoleh"]      if head_pose > 0.12 else []) + \
                     ([f"mulut sering terbuka"]       if mouth_ratio > 0.06 else [])
            return {
                'level': 'PERHATIAN',
                'summary': f"Skor fokus {score}% — Perlu sedikit peningkatan",
                'details': ("Perhatikan: " + ", ".join(issues)) if issues
                           else f"Mata: {ear_s}, Kepala: {head_s}, Mulut: {mouth_s}",
                'suggestion': "Hadapkan wajah ke kamera dan jaga mata tetap terbuka."
            }
        else:
            issues = ([f"mata tertutup/mengantuk"]      if ear < 0.15 else []) + \
                     ([f"kepala menoleh dari layar"]     if head_pose > 0.15 else []) + \
                     ([f"mulut terbuka (mungkin menguap)"] if mouth_ratio > 0.1 else [])
            return {
                'level': 'TIDAK FOKUS',
                'summary': f"Skor fokus {score}% — Perlu perhatian",
                'details': ("Faktor: " + ", ".join(issues)) if issues
                           else f"Mata: {ear_s}, Kepala: {head_s}, Mulut: {mouth_s}",
                'suggestion': "Atur posisi, pastikan wajah terlihat jelas, istirahatkan mata."
            }

    def reset(self):
        self._history.clear()


# ============================================================================
# VISUALIZER (OpenCV overlay — dipakai di VideoProcessor)
# ============================================================================

class Visualizer:
    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks) -> np.ndarray:
        if not landmarks:
            return frame
        h, w = frame.shape[:2]
        for idx in LANDMARKS['LEFT_EYE'] + LANDMARKS['RIGHT_EYE']:
            if idx < len(landmarks):
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 136), -1)
        xs = [lm.x * w for lm in landmarks[:100]]
        ys = [lm.y * h for lm in landmarks[:100]]
        if xs and ys:
            cv2.rectangle(frame,
                          (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))),
                          (80, 80, 80), 1)
        return frame

    @staticmethod
    def draw_overlay(frame: np.ndarray, pred: dict) -> np.ndarray:
        score = pred['score']
        level = pred['explanation']['level']
        color = ((0, 255, 136) if level == 'FOKUS'
                 else (255, 204, 0) if level == 'PERHATIAN'
                 else (255, 68, 68))
        h, w  = frame.shape[:2]
        ov    = frame.copy()
        cv2.rectangle(ov, (10, 10), (360, 115), (0, 0, 0), -1)
        frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"STATUS: {level}",           (20, 38),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,         2)
        cv2.putText(frame, f"SCORE: {score}%",            (20, 62),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (200,200,200), 1)
        f = pred['features']
        cv2.putText(frame,
                    f"EAR:{f['ear']:.3f}  Head:{f['head_pose']:.3f}  Mouth:{f['mouth_ratio']:.3f}",
                    (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        bw = int(w * 0.25);  bx = w - bw - 20
        cv2.rectangle(frame, (bx, 26), (bx + bw, 34),                          (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, 26), (bx + int(bw * score / 100), 34), color, -1)
        return frame

    @staticmethod
    def draw_no_face(frame: np.ndarray) -> np.ndarray:
        ov = frame.copy()
        cv2.rectangle(ov, (10, 10), (420, 75), (0, 0, 0), -1)
        frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "TIDAK ADA WAJAH TERDETEKSI",                           (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        cv2.putText(frame, "Pastikan wajah terlihat jelas & pencahayaan cukup",    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        return frame


# ============================================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================================

class FocusVideoProcessor(VideoProcessorBase):
    """Per-frame processor untuk streamlit-webrtc."""

    def __init__(self):
        model_path = download_model()
        self.extractor  = FaceFeatureExtractor(model_path)
        self.predictor  = FocusPredictor()
        self._lock      = threading.Lock()
        self.last_pred  = None
        self.history: list = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img     = frame.to_ndarray(format="bgr24")
        img     = cv2.flip(img, 1)                          # mirror
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        feats = self.extractor.extract(img_rgb)

        if feats['face_detected']:
            pred = self.predictor.predict(
                feats['ear'], feats['head_pose'], feats['mouth_ratio']
            )
            img = Visualizer.draw_landmarks(img, feats['landmarks'])
            img = Visualizer.draw_overlay(img, pred)
            with self._lock:
                self.last_pred = pred
                self.history.append({
                    'timestamp':   datetime.now().strftime("%H:%M:%S"),
                    'score':       pred['score'],
                    'ear':         round(feats['ear'], 4),
                    'head_pose':   round(feats['head_pose'], 4),
                    'mouth_ratio': round(feats['mouth_ratio'], 4),
                    'label':       'FOKUS' if pred['label'] == 1 else 'TIDAK FOKUS'
                })
        else:
            img = Visualizer.draw_no_face(img)
            with self._lock:
                self.last_pred = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        try:
            self.extractor.release()
        except Exception:
            pass


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session():
    defaults = {
        'history':      [],
        'frame_count':  0,
        'last_pred':    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 FocusWebCam</h1>
        <p>Deteksi Tingkat Fokus Wajah dengan Machine Learning</p>
        <p class="tags">📚 Supervised Learning &nbsp;|&nbsp; 🧬 Feature Extraction
           &nbsp;|&nbsp; 📊 Logistic Regression &nbsp;|&nbsp; 🤖 Real-time Inference</p>
    </div>
    """, unsafe_allow_html=True)


def show_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 Model Info")
        st.markdown("---")
        st.markdown("**Bobot Model (Coefficients):**")

        coef_df = pd.DataFrame({
            'Fitur':    ['EAR (Mata)', 'Head Pose', 'Mouth Ratio'],
            'Bobot':    [MODEL_COEF['ear'], MODEL_COEF['head_pose'], MODEL_COEF['mouth_ratio']],
            'Pengaruh': [
                '↑ Fokus' if MODEL_COEF['ear'] > 0        else '↓ Fokus',
                '↑ Fokus' if MODEL_COEF['head_pose'] > 0  else '↓ Fokus',
                '↑ Fokus' if MODEL_COEF['mouth_ratio'] > 0 else '↓ Fokus',
            ]
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**📊 Performa Model:**")
        st.metric("Accuracy", "81.45%")
        st.metric("F1 Score", "0.834")

        st.markdown("---")
        st.markdown("**📚 Konsep ML:**")
        concepts = {
            "Supervised Learning":  "Model belajar dari data berlabel",
            "Feature Extraction":   "EAR, Head Pose, Mouth Ratio",
            "Logistic Regression":  "Klasifikasi biner + probabilitas",
            "Sigmoid Function":     "Output ke probabilitas 0–1",
            "Standardization":      "Normalisasi fitur (mean=0, std=1)"
        }
        for k, v in concepts.items():
            st.markdown(f"**{k}**  \n<small style='color:#6b7280'>{v}</small>",
                        unsafe_allow_html=True)


def show_live_detection():
    st.markdown("### 🎥 Live Detection")
    st.info(
        "💡 **Petunjuk:** Hadapkan wajah ke kamera, pastikan pencahayaan cukup. "
        "Model mendeteksi fokus berdasarkan bukaan mata, posisi kepala, dan bukaan mulut. "
        "**Klik START ▶ untuk mulai.**"
    )

    col_r, _ = st.columns([1, 4])
    with col_r:
        if st.button("🔄 Reset Sesi", use_container_width=True):
            st.session_state.history    = []
            st.session_state.frame_count = 0
            st.session_state.last_pred  = None
            st.success("Sesi direset!")

    # ── WebRTC ──────────────────────────────────────────────────────────────
    ctx = webrtc_streamer(
        key="focus-detection",
        video_processor_factory=FocusVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Metric panel (live update) ──────────────────────────────────────────
    metrics_ph = st.empty()

    if ctx.video_processor:
        while True:
            with ctx.video_processor._lock:
                pred    = ctx.video_processor.last_pred
                history = list(ctx.video_processor.history)

            if pred:
                score = pred['score']
                level = pred['explanation']['level']
                badge_cls = (
                    "badge-focus"   if level == 'FOKUS' else
                    "badge-warning" if level == 'PERHATIAN' else
                    "badge-danger"
                )
                score_cls = (
                    "metric-focus"   if score >= 65 else
                    "metric-warning" if score >= 40 else
                    "metric-danger"
                )

                with metrics_ph.container():
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            f"<div style='text-align:center'>"
                            f"<small>🎯 SKOR FOKUS</small><br>"
                            f"<span class='{score_cls}' style='font-size:1.8rem'>{score}%</span><br>"
                            f"<span class='status-badge {badge_cls}'>{level}</span>"
                            f"</div>", unsafe_allow_html=True
                        )
                    with c2:
                        st.metric("👁️ EAR",        f"{pred['features']['ear']:.3f}")
                    with c3:
                        st.metric("🔄 Head Pose",   f"{pred['features']['head_pose']:.3f}")
                    with c4:
                        st.metric("👄 Mouth Ratio", f"{pred['features']['mouth_ratio']:.3f}")

                    exp = pred['explanation']
                    st.markdown(
                        f'<div class="explanation-box">'
                        f"<b>📊 {exp['summary']}</b><br>"
                        f"<small>{exp['details']}</small><br>"
                        f"<small>💡 <i>{exp['suggestion']}</i></small>"
                        f"</div>", unsafe_allow_html=True
                    )

                st.session_state.history    = history
                st.session_state.frame_count = len(history)

            time.sleep(1)

    elif st.session_state.history:
        show_session_summary()


def show_session_summary():
    if not st.session_state.history:
        return

    st.markdown("### 📊 Ringkasan Sesi Terakhir")
    df = pd.DataFrame(st.session_state.history)

    avg   = df['score'].mean()
    fo    = (df['score'] >= 65).mean() * 100
    wa    = ((df['score'] >= 40) & (df['score'] < 65)).mean() * 100
    un    = (df['score'] < 40).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rata-rata", f"{avg:.1f}%")
    with c2: st.metric("FOKUS",     f"{fo:.1f}%")
    with c3: st.metric("PERHATIAN", f"{wa:.1f}%")
    with c4: st.metric("TDK FOKUS", f"{un:.1f}%")

    # Line chart skor
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df['score'].tolist(), mode='lines+markers', name='Skor',
        line=dict(color='#00ff88', width=2),
        marker=dict(size=4, color='#00ff88')
    ))
    fig.add_hline(y=65, line_dash="dash", line_color="#00ff88",
                  annotation_text="FOKUS", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="#ffcc00",
                  annotation_text="PERHATIAN", annotation_position="right")
    fig.update_layout(
        title="Perkembangan Skor Fokus",
        xaxis_title="Frame ke-", yaxis_title="Skor (%)",
        yaxis_range=[0, 100], height=300, template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # EAR / Head Pose / Mouth trend
    with st.expander("📈 Detail Fitur per Frame"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=df['ear'].tolist(),        name='EAR',        line=dict(color='#00bfff')))
        fig2.add_trace(go.Scatter(y=df['head_pose'].tolist(),  name='Head Pose',  line=dict(color='#ff7f50')))
        fig2.add_trace(go.Scatter(y=df['mouth_ratio'].tolist(),name='Mouth Ratio',line=dict(color='#da70d6')))
        fig2.update_layout(height=250, template="plotly_dark",
                           title="Nilai Fitur per Frame", xaxis_title="Frame")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Unduh Data Sesi"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "focus_session.csv", "text/csv")


def show_model_explanation():
    st.markdown("### 🧠 Bagaimana Model Bekerja?")

    with st.expander("📖 Logistic Regression", expanded=True):
        st.markdown("""
**Logistic Regression** adalah algoritma klasifikasi biner yang menggunakan fungsi sigmoid
untuk mengubah kombinasi linier fitur menjadi probabilitas.

#### Rumus:
```
P(FOKUS) = 1 / (1 + e^-(w₁·EAR + w₂·HeadPose + w₃·Mouth + b))
```

| Komponen | Nilai | Interpretasi |
|---|---|---|
| w₁ (EAR) | +1.05 | EAR tinggi (mata terbuka) → lebih fokus |
| w₂ (Head Pose) | −2.66 | Head pose tinggi (menoleh) → kurang fokus |
| w₃ (Mouth) | +2.00 | Mouth ratio rendah (mulut tertutup) → lebih fokus |
| b (Intercept) | −0.52 | Bias model |

#### Pipeline Prediksi:
1. **Feature Extraction** — MediaPipe mengekstrak 478 landmark wajah
2. **Geometric Features** — Hitung EAR, Head Pose, Mouth Ratio dari landmark
3. **Standardization** — `(x − mean) / std` agar skala seragam
4. **Linear Combination** — `z = w₁x₁ + w₂x₂ + w₃x₃ + b`
5. **Sigmoid** — `P = 1 / (1 + e^−z)` → nilai 0–1
6. **Threshold** — P > 0.5 → **FOKUS**, P ≤ 0.5 → **TIDAK FOKUS**
        """)

    with st.expander("👁️ Penjelasan 3 Fitur"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
**1. EAR (Eye Aspect Ratio)**

Mengukur seberapa terbuka mata.

| Nilai | Status |
|---|---|
| > 0.22 | Mata terbuka ✅ |
| 0.15–0.22 | Kedipan normal |
| < 0.15 | Mengantuk ❌ |

*Rumus:* `(|p2−p6| + |p3−p5|) / (2·|p1−p4|)`
            """)
        with c2:
            st.markdown("""
**2. Head Pose**

Seberapa kepala menoleh dari kamera.

| Nilai | Status |
|---|---|
| < 0.08 | Menghadap kamera ✅ |
| 0.08–0.15 | Sedikit menoleh |
| > 0.15 | Menoleh ❌ |

*Rumus:* `|hidung − tengah wajah| / lebar wajah`
            """)
        with c3:
            st.markdown("""
**3. Mouth Ratio**

Rasio bukaan mulut (menguap = tidak fokus).

| Nilai | Status |
|---|---|
| < 0.04 | Mulut tertutup ✅ |
| 0.04–0.08 | Sedikit terbuka |
| > 0.08 | Menguap ❌ |

*Rumus:* `tinggi mulut / lebar mulut`
            """)

    with st.expander("📊 Koefisien Model & Interpretasi"):
        fig = go.Figure(go.Bar(
            x=['EAR (Mata)', 'Head Pose', 'Mouth Ratio'],
            y=[MODEL_COEF['ear'], MODEL_COEF['head_pose'], MODEL_COEF['mouth_ratio']],
            marker_color=['#00ff88', '#ff4444', '#00bfff'],
            text=[f"{v:+.4f}" for v in MODEL_COEF.values()],
            textposition='outside'
        ))
        fig.update_layout(
            title="Bobot Logistic Regression",
            yaxis_title="Koefisien",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
> **Interpretasi:** Koefisien positif (+) berarti fitur tersebut meningkatkan probabilitas FOKUS.
> Koefisien negatif (−) berarti fitur menurunkan probabilitas FOKUS (Head Pose negatif
> karena kepala menoleh = tidak fokus).
        """)


def show_about():
    st.markdown("""
### ℹ️ Tentang Aplikasi

**FocusWebCam** adalah aplikasi demonstrasi Machine Learning untuk deteksi tingkat fokus
berbasis kamera web secara real-time.

#### 🎯 Tujuan Pembelajaran
1. **Supervised Learning** — Klasifikasi biner dengan data berlabel
2. **Feature Extraction** — Mengubah citra menjadi fitur numerik bermakna
3. **Logistic Regression** — Algoritma klasifikasi probabilistik
4. **Real-time Inference** — Prediksi langsung dari video stream
5. **Explainable AI** — Memberikan penjelasan prediksi yang dapat dipahami

#### 🛠️ Stack Teknologi
| Komponen | Teknologi |
|---|---|
| Web Framework | Streamlit 1.57 |
| Webcam Stream | streamlit-webrtc 0.47 |
| Face Detection | MediaPipe FaceLandmarker (API ≥0.10) |
| Image Processing | OpenCV 4.8 |
| Numerik | NumPy 1.26 |
| Visualisasi | Plotly |

#### 📈 Performa Model
| Metrik | Nilai |
|---|---|
| Accuracy | 81.45% |
| F1 Score | 0.834 |
| Precision (FOKUS) | 0.79 |
| Recall (FOKUS) | 0.88 |

#### 📁 Dataset
Model dilatih dari dataset gambar yang dikategorikan menjadi:
- **Engaged** (Fokus): Confused, Focused, Frustrated
- **Not Engaged** (Tidak Fokus): Bored, Drowsy, Looking Away

Total ~2000 sampel fitur (lihat `features.csv`).
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Pastikan model sudah tersedia
    download_model()

    init_session()
    show_header()
    show_sidebar()

    tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "🧠 Model Explanation", "ℹ️ About"])

    with tab1:
        show_live_detection()

    with tab2:
        show_model_explanation()

    with tab3:
        show_about()

    st.markdown("""
    <div class="footer">
    FocusWebCam &nbsp;|&nbsp; MediaPipe · streamlit-webrtc · Logistic Regression<br>
    Supervised Learning &nbsp;|&nbsp; Feature Extraction &nbsp;|&nbsp; Real-time Inference
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
