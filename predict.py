import librosa
import numpy as np
import pickle
import soundfile as sf
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load models
rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
xgb_model = pickle.load(open("models/xgboost.pkl", "rb"))
cat_model = pickle.load(open("models/catboost.pkl", "rb"))

models = {
    "CatBoost (Recommended)": cat_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

emotion_map = {
    0: 'sad',
    1: 'happy',
    2: 'surprised',
    3: 'angry',
    4: 'calm'
}
emotion_labels = list(emotion_map.values())


def convert_to_wav(audio_file):
    data, samplerate = sf.read(audio_file)
    new_filepath = "temp_audio.wav"
    sf.write(new_filepath, data, samplerate)
    return new_filepath


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
    return predictions_with_labels


import logging

def analyze_folder(folder_path):
    summary = {emotion: [] for emotion in emotion_labels}
    
    # Log folder path and files
    logging.info(f"Analyzing folder: {folder_path}")
    
    try:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    except Exception as e:
        logging.error(f"Error listing files in folder: {e}")
        return {}
    
    for file_path in files:
        logging.info(f"Processing file: {file_path}")
        
        if os.path.isfile(file_path):
            try:
                wav_file_path = convert_to_wav(file_path)
                audio, sample_rate = librosa.load(wav_file_path, sr=None)
                features = extract_features(audio, sample_rate)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue

            try:
                for model_name, model in models.items():
                    predictions = model.predict_proba(features)[0]
                    for emotion, prob in zip(emotion_labels, predictions):
                        summary[emotion].append(prob)
            except Exception as e:
                logging.error(f"Error making predictions for file {file_path}: {e}")
                continue
    
    avg_probs = {emotion: np.mean(probs) * 100 for emotion, probs in summary.items()}
    return avg_probs



def analyze_long_audio(audio_file):
    import matplotlib.pyplot as plt
    audio_path = convert_to_wav(audio_file)
    audio, sample_rate = librosa.load(audio_path, sr=None)
    segment_duration = 5
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