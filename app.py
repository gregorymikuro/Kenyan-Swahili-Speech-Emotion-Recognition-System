import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import soundfile as sf

# Load the meta model
meta_model = joblib.load("models/meta_model.pkl")

# Emotion mapping
emotion_map = {
    0: 'sad',
    1: 'happy',
    2: 'surprised',
    3: 'angry',
    4: 'calm'
}
emotion_labels = list(emotion_map.values())

# Feedback messages tailored for a call center
feedback_messages = {
    'sad': "The analysis indicates a predominance of sadness. Consider offering empathetic responses and possible solutions to uplift the caller's mood.",
    'happy': "The analysis reveals a predominance of happiness. Continue to engage positively and reinforce the caller's satisfaction.",
    'surprised': "The analysis shows a predominance of surprise. Clarify any points of confusion and ensure clear communication to maintain the caller's trust.",
    'angry': "The analysis indicates a high prevalence of anger. Address the caller's concerns promptly, and aim to resolve the issue with patience.",
    'calm': "The analysis shows a predominance of calmness. Maintain a steady and supportive tone to keep the conversation on track."
}

def convert_to_wav(audio_file):
    if isinstance(audio_file, str):
        if not audio_file.endswith('.wav'):
            data, samplerate = sf.read(audio_file)
            new_filepath = audio_file.rsplit('.', 1)[0] + '.wav'
            sf.write(new_filepath, data, samplerate)
            return new_filepath
        else:
            return audio_file
    else:
        if not audio_file.name.endswith('.wav'):
            data, samplerate = sf.read(audio_file)
            new_filepath = audio_file.name.rsplit('.', 1)[0] + '.wav'
            sf.write(new_filepath, data, samplerate)
            return new_filepath
        else:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            return "temp_audio.wav"

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

def analyze_single_audio(audio_file):
    audio_path = convert_to_wav(audio_file)
    audio, sample_rate = librosa.load(audio_path, sr=None)
    features = extract_features(audio, sample_rate)
    predictions = meta_model.predict_proba(features)[0]

    predictions_with_labels = {emotion_map[i]: prob * 100 for i, prob in enumerate(predictions)}
    
    max_emotion = max(predictions_with_labels, key=predictions_with_labels.get)
    feedback = feedback_messages[max_emotion]

    return predictions_with_labels, feedback

def analyze_files(files):
    file_analysis = []
    for file in files:
        wav_file_path = convert_to_wav(file)
        audio, sample_rate = librosa.load(wav_file_path, sr=None)
        features = extract_features(audio, sample_rate)
        predictions = meta_model.predict_proba(features)[0]
        
        predictions_with_labels = {emotion_map[i]: prob * 100 for i, prob in enumerate(predictions)}
        max_emotion = max(predictions_with_labels, key=predictions_with_labels.get)
        file_analysis.append((file.name, max_emotion))
    
    avg_probs = {emotion: np.mean([meta_model.predict_proba(extract_features(librosa.load(convert_to_wav(file), sr=None)[0], sample_rate))[0][i] * 100 for file in files]) for i, emotion in enumerate(emotion_labels)}
    max_emotion = max(avg_probs, key=avg_probs.get)
    feedback = f"Based on the analysis of the audio files, the predominant emotion detected is '{max_emotion}'. {feedback_messages[max_emotion]}"
    
    return avg_probs, feedback, file_analysis

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
        segment_pred = meta_model.predict_proba(features)[0]
        
        time_point = start / sample_rate
        times.append(time_point)
        
        for emotion, prob in zip(emotion_labels, segment_pred):
            emotion_probs[emotion].append(prob * 100)

    plt.figure(figsize=(12, 8))
    for emotion in emotion_labels:
        plt.plot(times, emotion_probs[emotion], label=emotion)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability (%)')
    plt.title('Emotion Probabilities Over Time')
    plt.legend(loc='upper right')
    plt.grid(True)

    return plt

def main():
    st.set_page_config(
        page_title="Skiza-AI: Swahili Emotion Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply a consistent color theme
    st.markdown(
        """
        <style>
        body {
            background-color: #F5F5F5;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
        .stRadio>div {
            background-color: #E0E0E0;
            padding: 10px;
            border-radius: 5px;
        }
        .stFileUploader>label {
            color: #333;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎙️ Skiza-AI: Kenyan Swahili Emotion Analysis")
    st.markdown("""
        **Welcome to Skiza-AI!** This tool provides detailed emotion analysis for Swahili audio files, tailored for call centers.
        Select an analysis type below to gain insights into customer emotions and improve service quality.
    """)

    analysis_type = st.radio("Select Analysis Type", ["Single Audio Analysis", "Folder Analysis", "Long Audio Analysis"], index=0)

    if analysis_type == "Single Audio Analysis":
        st.header("🔊 Single Audio Analysis")
        audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "ogg", "flac"])
        
        if audio_file is not None:
            if st.button("Start Analysis"):
                predictions_with_labels, feedback = analyze_single_audio(audio_file)

                st.subheader("🎯 Predicted Emotion Probabilities")
                plt.figure(figsize=(8, 5))
                plt.bar(predictions_with_labels.keys(), predictions_with_labels.values(), color='#5A9')
                plt.xlabel("Emotion")
                plt.ylabel("Probability (%)")
                plt.title("Emotion Probabilities")
                plt.xticks(rotation=45)
                st.pyplot(plt)
                
                st.subheader("📋 Relevant Feedback")
                st.write(feedback)

    elif analysis_type == "Folder Analysis":
        st.header("📁 Folder Analysis")
        uploaded_files = st.file_uploader("Upload Multiple Audio Files", type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Start Analysis"):
                avg_probs, feedback, file_analysis = analyze_files(uploaded_files)
                
                st.subheader("📊 Average Emotion Probabilities")
                st.bar_chart(avg_probs)
                
                st.subheader("📋 Relevant Feedback for Folder Analysis")
                st.write(feedback)

                st.subheader("🗂️ Files and Predominant Emotions")
                for file_name, emotion in file_analysis:
                    st.write(f"**File:** {file_name} | **Predominant Emotion:** {emotion}")

    elif analysis_type == "Long Audio Analysis":
        st.header("⏳ Long Audio Analysis")
        long_audio_file = st.file_uploader("Upload a Long Audio File", type=["wav", "mp3", "ogg", "flac"])
        
        if long_audio_file is not None:
            if st.button("Start Analysis"):
                plt_fig = analyze_long_audio(long_audio_file)
                st.subheader("📊 Emotion Probabilities Over Time")
                st.pyplot(plt_fig)
                
                st.subheader("📋 Relevant Feedback for Long Audio Analysis")
                st.write("This analysis reveals how emotions evolve over time. Pay attention to sections with high emotional intensity for insights into customer sentiment.")

if __name__ == "__main__":
    main()