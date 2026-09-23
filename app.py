import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Age, Gender & Emotion Predictor",
    page_icon="🙂",
    layout="centered"
)

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = 224
AGE_SCALE = 120

emotion_labels = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Sadness",
    6: "Surprise"
}

# UTKFace:
# 0 = Male, 1 = Female
gender_labels = {
    0: "Male",
    1: "Female"
}

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_age_gender_model():
    return tf.keras.models.load_model("model.h5", compile=False)

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(
        "emotion_model.h5",
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": tf.keras.applications.vgg16.preprocess_input
        }
    )

age_gender_model = load_age_gender_model()
emotion_model = load_emotion_model()

# -----------------------------
# Preprocessing Functions
# -----------------------------
def preprocess_for_age_gender(image):
    """
    Use this for age-gender model.
    Your multi-task model was trained using VGG16 preprocess_input outside the model.
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    return img_array


def preprocess_for_emotion(image):
    """
    Use this for emotion model.
    If your emotion model already has preprocess_input inside the model,
    do not apply preprocess_input here.
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# -----------------------------
# Prediction Functions
# -----------------------------
def predict_age_gender(image):
    processed_image = preprocess_for_age_gender(image)

    predictions = age_gender_model.predict(processed_image)

    # If model output is dictionary
    if isinstance(predictions, dict):
        age_pred = predictions["age_output"]
        gender_pred = predictions["gender_output"]

    # If model output is list
    else:
        # Your multi-task model output order was:
        # age_output, gender_output, emotion_output
        age_pred = predictions[0]
        gender_pred = predictions[1]

    predicted_age = age_pred[0][0] * AGE_SCALE

    gender_prob = gender_pred[0][0]
    gender_class = 1 if gender_prob >= 0.5 else 0
    gender = gender_labels[gender_class]

    gender_confidence = gender_prob if gender_class == 1 else 1 - gender_prob
    gender_confidence = gender_confidence * 100

    return predicted_age, gender, gender_confidence


def predict_emotion(image):
    processed_image = preprocess_for_emotion(image)

    emotion_pred = emotion_model.predict(processed_image)

    emotion_probs = emotion_pred[0]

    top_emotion_index = np.argmax(emotion_probs)
    top_emotion = emotion_labels[top_emotion_index]
    top_emotion_confidence = emotion_probs[top_emotion_index] * 100

    return top_emotion, top_emotion_confidence, emotion_probs


# -----------------------------
# UI
# -----------------------------
st.title("🙂 Age, Gender & Emotion Predictor")
st.caption("Upload a face image and the app will predict age, gender, and emotion.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("📷 Uploaded Image Preview")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    predict_btn = st.button("🔍 Predict", use_container_width=True)

    if predict_btn:

        with st.spinner("Analyzing image..."):
            age, gender, gender_conf = predict_age_gender(image)
            emotion, emotion_conf, emotion_probs = predict_emotion(image)

        st.divider()

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Age", f"{age:.0f} years")

        with col2:
            st.metric("Predicted Gender", gender)
            st.caption(f"Confidence: {gender_conf:.2f}%")

        with col3:
            st.metric("Top Emotion", emotion)
            st.caption(f"Confidence: {emotion_conf:.2f}%")

        st.divider()

        st.subheader("😊 Emotion Probabilities")

        # Show emotions sorted by probability
        sorted_indices = np.argsort(emotion_probs)[::-1]

        for idx in sorted_indices:
            emotion_name = emotion_labels[idx]
            prob = float(emotion_probs[idx])
            percentage = prob * 100

            st.markdown(f"**{emotion_name}**")
            st.progress(prob)
            st.caption(f"{percentage:.2f}%")
            st.write("")

else:
    st.info("Please upload an image to start prediction.")

st.divider()

st.caption(
    "This app uses separate CNN models for age-gender prediction and emotion classification."
)

st.divider()

st.markdown(
    "Developed by **Karan Gojiya** | [GitHub](https://github.com/KaranGojiya) | [LinkedIn](https://linkedin.com/in/karan-gojiya)"
)
