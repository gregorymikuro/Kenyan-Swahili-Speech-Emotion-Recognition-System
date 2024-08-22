import streamlit as st
import librosa
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sys

# st.write(f"Python version: {sys.version}")

# Load the models
rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
xgb_model = pickle.load(open("models/xgboost.pkl", "rb"))
cat_model = pickle.load(open("models/catboost.pkl", "rb"))

models = {
    "CatBoost (Recommended)": cat_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Get emotion labels from one of the models (assuming they all have the same classes)
emotion_map = {
    0: 'sad',
    1: 'happy',
    2: 'surprised',
    3: 'angry',
    4: 'calm'
}
emotion_labels = list(emotion_map.values())

# Comprehensive feedback messages
feedback_messages = {
    'sad': "The analysis indicates a predominance of sadness in the audio files. This suggests the customers may be feeling down or dissatisfied. It is advisable to offer empathy, understand their concerns, and provide solutions to improve their experience.",
    'happy': "The analysis reveals a predominance of happiness in the audio files. This indicates that customers are generally pleased with the service. Continue to maintain this positive interaction, and consider engaging customers further by asking for feedback or a testimonial.",
    'surprised': "The analysis shows a predominance of surprise in the audio files. This may suggest that customers encountered something unexpected. Ensure that any surprises are positive and clarify any points of confusion to maintain customer satisfaction.",
    'angry': "The analysis indicates a high prevalence of anger in the audio files. This suggests that customers are frustrated or upset. It is crucial to address their concerns with calmness and patience, listen to their grievances, and work diligently to resolve any issues they face.",
    'calm': "The analysis shows a predominance of calmness in the audio files. This suggests that customers are collected and composed. Maintain this positive atmosphere by providing clear, concise, and reassuring information."
}

def convert_to_wav(audio_file):
    if isinstance(audio_file, str):  # Check if audio_file is a file path
        if not audio_file.endswith('.wav'):
            data, samplerate = sf.read(audio_file)
            new_filepath = audio_file.rsplit('.', 1)[0] + '.wav'
            sf.write(new_filepath, data, samplerate)
            return new_filepath
        else:
            return audio_file
    else:  # Handle file-like object (e.g., Streamlit file uploader)
        if not audio_file.name.endswith('.wav'):
            data, samplerate = sf.read(audio_file)
            new_filepath = audio_file.name.rsplit('.', 1)[0] + '.wav'
            sf.write(new_filepath, data, samplerate)
            return new_filepath
        else:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            return "temp_audio.wav"

# Function to extract features
def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(librosa.feature.zero_crossing_rate(y=audio)),
        np.mean(librosa.feature.rms(y=audio)),
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)),
        np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)),
        np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
    ])
    return features.reshape(1, -1)

def analyze_single_audio(audio_file, model_name):
    audio_path = convert_to_wav(audio_file)
    audio, sample_rate = librosa.load(audio_path, sr=None)
    features = extract_features(audio, sample_rate)
    predictions = models[model_name].predict_proba(features)[0]

    predictions_with_labels = {emotion_map[i]: prob * 100 for i, prob in enumerate(predictions)}
    
    max_emotion = max(predictions_with_labels, key=predictions_with_labels.get)
    feedback = feedback_messages[max_emotion]

    return predictions_with_labels, feedback

def analyze_files(files, model_name):
    summary = {emotion: [] for emotion in emotion_labels}
    for file in files:
        wav_file_path = convert_to_wav(file)
        audio, sample_rate = librosa.load(wav_file_path, sr=None)
        features = extract_features(audio, sample_rate)
        predictions = models[model_name].predict_proba(features)[0]
        for emotion, prob in zip(emotion_labels, predictions):
            summary[emotion].append(prob)
    
    avg_probs = {emotion: np.mean(probs) * 100 for emotion, probs in summary.items()}
    max_emotion = max(avg_probs, key=avg_probs.get)
    feedback = f"Based on the analysis of the audio files, the predominant emotion detected is '{max_emotion}'. This suggests the following:\n\n{feedback_messages[max_emotion]}"
    
    return avg_probs, feedback

def analyze_long_audio(audio_file):
    audio_path = convert_to_wav(audio_file)
    audio, sample_rate = librosa.load(audio_path, sr=None)
    segment_duration = 5  # Segment duration in seconds
    times = []
    emotion_probs = {emotion: [] for emotion in emotion_labels}
    
    for start in range(0, len(audio), segment_duration * sample_rate):
        end = min(start + segment_duration * sample_rate, len(audio))
        segment = audio[start:end]
        
        if len(segment) < 1 * sample_rate:
            continue
        
        features = extract_features(segment, sample_rate)
        segment_pred = {model_name: model.predict_proba(features)[0] for model_name, model in models.items()}
        
        time_point = start / sample_rate
        times.append(time_point)
        
        for emotion, prob in zip(emotion_labels, np.mean([segment_pred[model_name] for model_name in models], axis=0)):
            emotion_probs[emotion].append(prob * 100)

    for emotion in emotion_labels:
        if len(times) != len(emotion_probs[emotion]):
            st.warning(f"Warning: Data length mismatch for emotion '{emotion}'.")
    
    plt.figure(figsize=(12, 8))
    for emotion in emotion_labels:
        if len(times) == len(emotion_probs[emotion]):
            plt.plot(times, emotion_probs[emotion], label=emotion)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability (%)')
    plt.title('Emotion Probabilities Over Time')
    plt.legend(loc='upper right')
    plt.grid(True)

    return plt

def main():
    st.title("Skiza-AI: Kenyan Swahili Emotion Analysis from Audio")
    st.markdown("""
        **Welcome to Skiza-AI!** This tool provides emotion analysis for audio files in Swahili. Choose from three types of analysis and see how emotions are represented in your audio files.
    """)
    
    analysis_type = st.radio("Select Analysis Type", ["Single Audio Analysis", "Folder Analysis", "Long Audio Analysis"])

    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=0,  
        format_func=lambda x: f"**{x}**" if x == "CatBoost (Recommended)" else x
    )

    if analysis_type == "Single Audio Analysis":
        st.header("Single Audio Analysis")
        audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "ogg", "flac"])
        
        if audio_file is not None:
            if st.button("Start Analysis"):
                predictions_with_labels, feedback = analyze_single_audio(audio_file, selected_model)

                st.subheader(f"Predicted Emotion Probabilities ({selected_model})")

                plt.figure(figsize=(8, 5))
                plt.bar(predictions_with_labels.keys(), predictions_with_labels.values())
                plt.xlabel("Emotion")
                plt.ylabel("Probability (%)")
                plt.title(f"Emotion Probabilities for {selected_model}")
                plt.xticks(rotation=45)
                st.pyplot(plt)
                
                st.subheader("Relevant Feedback")
                st.write(feedback)

    elif analysis_type == "Folder Analysis":
        st.header("Folder Analysis")
        uploaded_files = st.file_uploader("Upload Multiple Audio Files", type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Start Analysis"):
                avg_probs, feedback = analyze_files(uploaded_files, selected_model)
                st.subheader(f"Average Emotion Probabilities ({selected_model})")
                st.bar_chart(avg_probs)
                
                st.subheader("Relevant Feedback for Folder Analysis")
                st.write(feedback)

    elif analysis_type == "Long Audio Analysis":
        st.header("Long Audio Analysis")
        long_audio_file = st.file_uploader("Upload a Long Audio File", type=["wav", "mp3", "ogg", "flac"])
        if long_audio_file is not None:
            if st.button("Start Analysis"):
                plt_fig = analyze_long_audio(long_audio_file)
                st.subheader("Emotion Probabilities Over Time")
                st.pyplot(plt_fig)
                
                st.subheader("Relevant Feedback for Long Audio Analysis")
                st.write("The analysis shows how emotions evolve over time. Monitor sections with high probability emotions for insights on customer sentiment.")

if __name__ == "__main__":
    main()
