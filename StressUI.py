import streamlit as st
import numpy as np
# import librosa
import cv2
from deepface import DeepFace
from tensorflow.keras.models import load_model
import threading

# Load the pre-trained CNN model
model = load_model("cnn_stress_classification_model.h5")

# Function to extract features from an audio file
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function for audio classification
def classify_audio(uploaded_file):
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_features("temp.wav")
    features = features.reshape(1, 13, 1, 1)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return "Non-Stress" if predicted_class == 0 else "Stress"

# Function for real-time emotion detection
def detect_emotion():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                emotion = "No Face Detected"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the current frame in Streamlit
        st.image(frame, channels="RGB", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit interface
st.title("Stress and Emotion Detection")

# Audio Stress Classification
st.write("### Audio Stress Classification")
uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])

if uploaded_file is not None:
    result = classify_audio(uploaded_file)
    st.write(f"Prediction: {result}")

# Emotion Detection
st.write("### Real-time Emotion Detection")
if st.button("Start Emotion Detection"):
    threading.Thread(target=detect_emotion, daemon=True).start()
