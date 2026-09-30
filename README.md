# Age, Gender & Emotion Predictor

A deep learning web app that predicts a person's **age**, **gender** and **facial emotion** from a single photo. Upload an image or use your camera and get all three predictions with confidence scores in about a second.

**[Live Demo](https://age-gender-emotion-predictor.streamlit.app/)**

## Features

- Predicts age (with an age group), gender and one of 7 emotions from one image
- Upload a JPG/PNG or take a photo with your camera
- Confidence score for gender and emotion, with a High / Medium / Low band
- Emotion probability chart showing how the model ranks all 7 emotions
- Prediction details: emotion margin (gap between the top two emotions), emotion uncertainty (normalized entropy), inference time and image size
- Fixes phone photo rotation automatically (EXIF orientation)
- Download the results as a JSON file
- Images are processed in memory and are not stored

## How it works

The app runs two separate CNN models on the same 224×224 image.

| Model | File | Task | Output |
|---|---|---|---|
| Age and gender | `model.h5` | Multi-task regression and classification | Age (scaled 0–1, multiplied by 120) and gender (sigmoid) |
| Emotion | `emotion_model.h5` | 7-class classification | Softmax over 7 emotions |

- Both models use a **VGG16-style convolutional backbone** with custom dense layers on top.
- The age/gender model was trained with a combined loss: MAE for age and binary cross-entropy for gender.
- The emotion model includes data augmentation layers (random flip and rotation) and VGG16 preprocessing inside the model.
- Emotion classes: Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise.

## Datasets

- **Age and gender:** UTKFace
- **Emotion:** [add dataset name]

The zipped training data is included in this repo (`Age.zip`, `Emotions.zip`).

## Model performance

Measured on the held-out test split.

| Task | Metric | Result |
|---|---|---|
| Age | MAE (years) | [add value] |
| Gender | Accuracy | [add value] |
| Emotion | Accuracy | [add value] |
| Emotion | Macro F1 | [add value] |

## Tech stack

Python, TensorFlow / Keras, Streamlit, NumPy, Pandas, Altair, Pillow

## Project structure

```
Age-Gender-Emotion-Predictor/
├── app.py               # Streamlit app
├── model.h5             # Age and gender model
├── emotion_model.h5     # Emotion model
├── Age.zip              # Age/gender dataset
├── Emotions.zip         # Emotion dataset
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/KaranGojiya/Age-Gender-Emotion-Predictor.git
cd Age-Gender-Emotion-Predictor
pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.11 or 3.12, since TensorFlow does not yet support the newest Python versions.

## Tips for better results

- Use a close-up, front-facing photo where the face fills most of the frame.
- Use good lighting and avoid heavy filters.
- Use one person per photo.

## Limitations

- Predictions are estimates. Age in particular can be off by several years.
- Accuracy drops on side profiles, low light, heavy makeup or filters, and small faces in wide photos.
- The models reflect the biases of their training datasets. Do not use this app to make decisions about people.

## Future improvements

- Automatic face detection and cropping
- Support for multiple faces in one photo
- Grad-CAM visualizations to show which face regions drive each prediction

## Author

**Karan Gojiya**, B.Tech CSE, IIIT Bhopal

[LinkedIn](https://www.linkedin.com/in/karan-gojiya) · [GitHub](https://github.com/KaranGojiya)
