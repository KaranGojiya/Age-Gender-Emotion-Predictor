import io
import json
import time
import hashlib

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps

# Optional: face detection. The app still works without OpenCV.
try:
    import cv2
    _CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    FACE_DETECTION_AVAILABLE = not _CASCADE.empty()
except Exception:
    cv2 = None
    _CASCADE = None
    FACE_DETECTION_AVAILABLE = False


# =============================================================
# Page config
# =============================================================
st.set_page_config(
    page_title="Age, Gender & Emotion Predictor",
    page_icon="🙂",
    layout="wide",
)

# =============================================================
# Constants
# =============================================================
IMG_SIZE = 224
AGE_SCALE = 120
MAX_UPLOAD_MB = 10

EMOTION_LABELS = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Sadness",
    6: "Surprise",
}
EMOTION_EMOJI = {
    "Anger": "😠",
    "Contempt": "😒",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sadness": "😢",
    "Surprise": "😲",
}

# UTKFace: 0 = Male, 1 = Female
GENDER_LABELS = {0: "Male", 1: "Female"}

# -------------------------------------------------------------
# FILL THESE IN with the real numbers from your test set.
# Anything left as None is hidden, so the app never shows
# made-up or empty values. Use strings, e.g. "91.2%" or "5.8".
# -------------------------------------------------------------
MODEL_METRICS = {
    "Age MAE (years)": None,
    "Gender accuracy": None,
    "Emotion accuracy": None,
    "Emotion macro F1": None,
}
AGE_GENDER_DATASET = "UTKFace"
EMOTION_DATASET = None  # e.g. "your dataset name"


# =============================================================
# Styling (works in light and dark themes)
# =============================================================
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1150px; }
    .result-card {
        border: 1px solid rgba(128,128,128,0.28);
        background: rgba(128,128,128,0.06);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        height: 100%;
    }
    .result-card .label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.15rem; }
    .result-card .value { font-size: 1.9rem; font-weight: 650; line-height: 1.2; }
    .result-card .sub   { font-size: 0.85rem; opacity: 0.75; margin-top: 0.3rem; }
    .small-note { font-size: 0.82rem; opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)


def card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
# Load models (unchanged)
# =============================================================
@st.cache_resource(show_spinner="Loading models...")
def load_age_gender_model():
    return tf.keras.models.load_model("model.h5", compile=False)


@st.cache_resource(show_spinner="Loading models...")
def load_emotion_model():
    return tf.keras.models.load_model(
        "emotion_model.h5",
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": tf.keras.applications.vgg16.preprocess_input
        },
    )


try:
    age_gender_model = load_age_gender_model()
    emotion_model = load_emotion_model()
except Exception as exc:
    st.error(
        "The models could not be loaded. Check that model.h5 and "
        "emotion_model.h5 are in the app folder and that requirements.txt "
        "matches the TensorFlow version used for training."
    )
    st.exception(exc)
    st.stop()


# =============================================================
# Image helpers
# =============================================================
def load_image(raw_bytes):
    """Open an image, fix phone rotation (EXIF), and force RGB."""
    image = Image.open(io.BytesIO(raw_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def detect_largest_face(image, margin=0.25):
    """
    Returns (crop, box, n_faces). box is (x1, y1, x2, y2) in original pixels.
    Returns (None, None, 0) when no face is found or OpenCV is missing.
    """
    if not FACE_DETECTION_AVAILABLE:
        return None, None, 0

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None, 0

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.width, x + w + pad_x)
    y2 = min(image.height, y + h + pad_y)
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), len(faces)


def draw_box(image, box):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    width = max(3, image.width // 200)
    draw.rectangle(box, outline=(46, 204, 113), width=width)
    return preview


def to_display(image, max_side=640):
    display = image.copy()
    display.thumbnail((max_side, max_side))
    return display


# =============================================================
# Preprocessing (unchanged logic)
# =============================================================
def preprocess_for_age_gender(image):
    """Age/gender model was trained with VGG16 preprocess_input applied outside the model."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.vgg16.preprocess_input(img_array)


def preprocess_for_emotion(image):
    """Emotion model already contains preprocess_input internally."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32")
    return np.expand_dims(img_array, axis=0)


# =============================================================
# Prediction helpers
# =============================================================
def softmax(x):
    x = np.asarray(x, dtype="float64")
    e = np.exp(x - np.max(x))
    return e / e.sum()


def normalized_entropy(probs):
    """0 = model is certain, 1 = completely undecided."""
    probs = np.clip(np.asarray(probs, dtype="float64"), 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum() / np.log(len(probs)))


def confidence_band(pct):
    if pct >= 80:
        return "High"
    if pct >= 60:
        return "Medium"
    return "Low"


def age_group(age):
    if age < 13:
        return "Child"
    if age < 20:
        return "Teenager"
    if age < 35:
        return "Young adult"
    if age < 55:
        return "Adult"
    return "Senior"


def predict_age_gender(image):
    processed = preprocess_for_age_gender(image)

    start = time.perf_counter()
    predictions = age_gender_model.predict(processed, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Output order of the multi-task model: age, gender (, emotion)
    if isinstance(predictions, dict):
        age_pred = predictions["age_output"]
        gender_pred = predictions["gender_output"]
    else:
        age_pred = predictions[0]
        gender_pred = predictions[1]

    age = float(np.clip(age_pred[0][0] * AGE_SCALE, 1, 116))

    gender_pred = np.asarray(gender_pred)
    if gender_pred.shape[-1] == 2:  # softmax head
        female_prob = float(gender_pred[0][1])
    else:  # single sigmoid unit
        female_prob = float(gender_pred[0][0])

    gender_class = 1 if female_prob >= 0.5 else 0
    gender_conf = (female_prob if gender_class == 1 else 1 - female_prob) * 100

    return {
        "age": age,
        "gender": GENDER_LABELS[gender_class],
        "gender_confidence": float(gender_conf),
        "age_gender_ms": float(elapsed_ms),
    }


def predict_emotion(image):
    processed = preprocess_for_emotion(image)

    start = time.perf_counter()
    emotion_pred = emotion_model.predict(processed, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = np.asarray(emotion_pred[0], dtype="float64")
    # Safety: if the model returned logits, convert to probabilities.
    if probs.min() < 0 or not np.isclose(probs.sum(), 1.0, atol=1e-2):
        probs = softmax(probs)

    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]

    return {
        "emotion": EMOTION_LABELS[int(top)],
        "emotion_confidence": float(probs[top] * 100),
        "runner_up": EMOTION_LABELS[int(second)],
        "runner_up_confidence": float(probs[second] * 100),
        "margin": float((probs[top] - probs[second]) * 100),
        "uncertainty": normalized_entropy(probs),
        "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(probs))},
        "emotion_ms": float(elapsed_ms),
    }


@st.cache_data(show_spinner=False, max_entries=32)
def run_inference(raw_bytes, use_face_crop):
    """Cached per (image, setting) so reruns are instant and predictions stay stable."""
    image = load_image(raw_bytes)
    original_size = image.size

    face_crop, box, n_faces = (None, None, 0)
    if use_face_crop:
        face_crop, box, n_faces = detect_largest_face(image)

    model_input = face_crop if face_crop is not None else image

    result = {}
    result.update(predict_age_gender(model_input))
    result.update(predict_emotion(model_input))
    result.update(
        {
            "original_size": original_size,
            "face_box": box,
            "faces_found": n_faces,
            "used_face_crop": face_crop is not None,
        }
    )
    return result


# =============================================================
# Sidebar
# =============================================================
with st.sidebar:
    st.header("Settings")

    if FACE_DETECTION_AVAILABLE:
        use_face_crop = st.toggle(
            "Detect and crop the face",
            value=True,
            help="Finds the largest face and crops it before prediction. "
                 "Turn off if your photo is already a tight face crop.",
        )
    else:
        use_face_crop = False
        st.caption(
            "Face detection is off. Add `opencv-python-headless` to "
            "requirements.txt to enable it."
        )

    st.divider()
    st.header("How it works")
    st.markdown(
        f"""
        - Two CNN models run on the same 224×224 face image.
        - **Age and gender:** multi-task model trained on {AGE_GENDER_DATASET}.
        - **Emotion:** separate 7-class model{f" trained on {EMOTION_DATASET}" if EMOTION_DATASET else ""}.
        - Images are processed in memory and are not stored by this app.
        """
    )
    st.caption("Predictions are estimates and should not be used to make decisions about people.")


# =============================================================
# Header
# =============================================================
st.title("Age, Gender & Emotion Predictor")
st.caption("Upload a face photo, or take one with your camera, to get all three predictions.")

# =============================================================
# Input
# =============================================================
tab_upload, tab_camera = st.tabs(["Upload a photo", "Use camera"])

raw_bytes = None

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"That file is larger than {MAX_UPLOAD_MB} MB. Upload a smaller image.")
        else:
            raw_bytes = uploaded.getvalue()

with tab_camera:
    camera_shot = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_shot is not None and raw_bytes is None:
        raw_bytes = camera_shot.getvalue()

if raw_bytes is None:
    st.info("Upload or capture a clear, front-facing photo to see the predictions.")
    st.stop()

try:
    image = load_image(raw_bytes)
except Exception:
    st.error("This file could not be read as an image. Try another JPG or PNG.")
    st.stop()

# =============================================================
# Inference
# =============================================================
with st.spinner("Analyzing image..."):
    res = run_inference(raw_bytes, use_face_crop)

# =============================================================
# Results
# =============================================================
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("Photo")
    if res["face_box"] is not None:
        st.image(to_display(draw_box(image, res["face_box"])), caption="Detected face (used for prediction)")
        if res["faces_found"] > 1:
            st.warning(f"{res['faces_found']} faces found. Using the largest one.")
    else:
        st.image(to_display(image), caption="Full image (used for prediction)")
        if use_face_crop:
            st.warning("No face detected, so the full image was used. Try a clearer, front-facing photo.")

with right:
    st.subheader("Predictions")

    c1, c2, c3 = st.columns(3)
    with c1:
        card(
            "Age",
            f"{res['age']:.0f} years",
            age_group(res["age"]),
        )
    with c2:
        card(
            "Gender",
            res["gender"],
            f"{res['gender_confidence']:.1f}% confidence "
            f"({confidence_band(res['gender_confidence'])})",
        )
    with c3:
        emoji = EMOTION_EMOJI.get(res["emotion"], "")
        card(
            "Emotion",
            f"{emoji} {res['emotion']}",
            f"{res['emotion_confidence']:.1f}% confidence "
            f"({confidence_band(res['emotion_confidence'])})",
        )

    st.write("")
    st.markdown("**Emotion probabilities**")

    prob_df = (
        pd.DataFrame(
            {"Emotion": list(res["probabilities"].keys()),
             "Probability (%)": [p * 100 for p in res["probabilities"].values()]}
        )
        .sort_values("Probability (%)", ascending=False)
        .reset_index(drop=True)
    )

    chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title=None),
            y=alt.Y("Emotion:N", sort=list(prob_df["Emotion"]), title=None),
            color=alt.condition(
                alt.datum.Emotion == res["emotion"],
                alt.value("#2E86DE"),
                alt.value("#9AA5B1"),
            ),
            tooltip=["Emotion", alt.Tooltip("Probability (%):Q", format=".2f")],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)

# =============================================================
# Prediction quality metrics
# =============================================================
st.divider()
st.subheader("Prediction details")

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Emotion margin",
    f"{res['margin']:.1f} pts",
    help=f"Gap between the top emotion ({res['emotion']}) and the runner-up "
         f"({res['runner_up']}, {res['runner_up_confidence']:.1f}%). "
         "A small gap means the model is torn between two emotions.",
)
m2.metric(
    "Emotion uncertainty",
    f"{res['uncertainty']:.2f}",
    help="Normalized entropy of the emotion probabilities. "
         "0 means the model is certain, 1 means it is completely undecided.",
)
m3.metric(
    "Inference time",
    f"{res['age_gender_ms'] + res['emotion_ms']:.0f} ms",
    help=f"Age/gender: {res['age_gender_ms']:.0f} ms, "
         f"emotion: {res['emotion_ms']:.0f} ms (both models combined).",
)
m4.metric(
    "Image size",
    f"{res['original_size'][0]}×{res['original_size'][1]}",
    help="Original resolution. Models receive a 224×224 resize.",
)

if res["emotion_confidence"] < 60 or res["gender_confidence"] < 60:
    st.info(
        "One or more predictions have low confidence. Better lighting, a "
        "front-facing pose, and a clearer face usually improve results."
    )

# =============================================================
# Model performance (only shown when you fill in MODEL_METRICS)
# =============================================================
available_metrics = {k: v for k, v in MODEL_METRICS.items() if v is not None}
if available_metrics:
    st.divider()
    st.subheader("Model performance")
    st.caption("Measured on the held-out test split during training.")
    cols = st.columns(len(available_metrics))
    for col, (name, value) in zip(cols, available_metrics.items()):
        col.metric(name, value)

# =============================================================
# Download
# =============================================================
export = {
    "age_years": round(res["age"], 1),
    "age_group": age_group(res["age"]),
    "gender": res["gender"],
    "gender_confidence_pct": round(res["gender_confidence"], 2),
    "emotion": res["emotion"],
    "emotion_confidence_pct": round(res["emotion_confidence"], 2),
    "emotion_probabilities_pct": {
        k: round(v * 100, 2) for k, v in res["probabilities"].items()
    },
    "face_detected": res["used_face_crop"],
    "inference_ms": round(res["age_gender_ms"] + res["emotion_ms"], 1),
}
st.download_button(
    "Download results (JSON)",
    data=json.dumps(export, indent=2),
    file_name="prediction_results.json",
    mime="application/json",
)

# =============================================================
# Footer
# =============================================================
st.divider()
st.markdown(
    '<p class="small-note">Developed by <b>Karan Gojiya</b> · '
    '<a href="https://github.com/KaranGojiya">GitHub</a> · '
    '<a href="https://linkedin.com/in/karan-gojiya">LinkedIn</a></p>',
    unsafe_allow_html=True,
)
