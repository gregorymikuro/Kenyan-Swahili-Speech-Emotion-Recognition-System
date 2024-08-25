# Operating System and File Handling
import os

# Numerical Computation
import numpy as np
import ast

# Audio Processing
import librosa
import noisereduce as nr
import librosa.display
from scipy.io import wavfile
from scipy.signal import spectrogram

# Data Analysis and Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning (scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                             precision_score, recall_score, roc_curve, auc, accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV


# Machine Learning (XGBoost and CatBoost)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Utility
import pickle
from itertools import cycle
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Suppress only UserWarnings with specific message patterns
warnings.filterwarnings('ignore', category=UserWarning, message='.*use_label_encoder.*')

class DataLoader:
    def __init__(self, data_dir, emotions):
        self.data_dir = data_dir
        self.emotions = emotions
        self.X = []  # List to store audio data (as numpy arrays)
        self.y = []  # List to store corresponding labels

    def load_data(self):
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(self.data_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            print(f"Number of .wav files in {emotion} folder: {len(wav_files)}")

            for filename in wav_files:
                filepath = os.path.join(emotion_dir, filename)
                audio, sample_rate = librosa.load(filepath)
                self.X.append(audio)
                self.y.append(i)  # Use emotion index as label

        self.y = np.array(self.y)  # Convert labels to numpy array


class DataCleaner(DataLoader):
    def __init__(self, X, sample_rate):
        self.X = X
        self.sample_rate = sample_rate

    def clean_data(self):
        cleaned_X = []
        for audio in self.X:
            # Noise Reduction
            cleaned_audio = nr.reduce_noise(y=audio, sr=self.sample_rate)

            # Trim Silence
            cleaned_audio, _ = librosa.effects.trim(cleaned_audio)

            cleaned_X.append(cleaned_audio)
        return cleaned_X

class EDA(DataCleaner):
    def __init__(self, X, y, sample_rate, data_dir, emotions):
        super().__init__(X, sample_rate)
        self.y = y
        self.data_dir = data_dir
        self.emotions = emotions
        self.genders = {'f': 'Female', 'm': 'Male'}
        self.gender_counts = {'Female': 0, 'Male': 0, 'Unknown': 0}
        self.emotion_counts = {emotion: 0 for emotion in emotions}
        self.audio_lengths = {emotion: [] for emotion in emotions}

    def count_recordings_per_emotion(self):
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(self.data_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            self.emotion_counts[emotion] = len(wav_files)
            print(f"Number of recordings for emotion '{emotion}': {len(wav_files)}")

    def count_genders(self):
        for emotion in self.emotions:
            emotion_dir = os.path.join(self.data_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            for filename in wav_files:
                gender_code = filename[1].lower()
                if gender_code in self.genders:
                    self.gender_counts[self.genders[gender_code]] += 1
                else:
                    self.gender_counts['Unknown'] += 1
        print(f"Gender counts: {self.gender_counts}")

    def plot_waveplots_and_spectrograms(self):
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(self.data_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            if wav_files:
                example_file = wav_files[0]
                filepath = os.path.join(emotion_dir, example_file)
                audio, sr = librosa.load(filepath)
                duration = librosa.get_duration(y=audio, sr=sr)
                
                # Waveplot
                plt.figure(figsize=(14, 5))
                plt.subplot(1, 2, 1)
                librosa.display.waveshow(audio, sr=sr)
                plt.title(f'{emotion} Waveplot')
                
                # Spectrogram
                plt.subplot(1, 2, 2)
                Sxx, f, t, im = plt.specgram(audio, Fs=sr)
                plt.title(f'{emotion} Spectrogram')
                
                plt.tight_layout()
                plt.show()
                
                # Store audio lengths
                self.audio_lengths[emotion].append(duration)

    def compute_audio_length_statistics(self):
        audio_length_stats = {}
        for emotion, lengths in self.audio_lengths.items():
            if lengths:
                lengths = np.array(lengths)
                audio_length_stats[emotion] = {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths),
                    'median': np.median(lengths)
                }
        print("Audio length statistics per emotion:")
        for emotion, stats in audio_length_stats.items():
            print(f"{emotion}: {stats}")

    def visualize_emotion_distribution(self):
        emotion_labels = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=emotion_labels, y=counts, palette="viridis")
        plt.xlabel('Emotions')
        plt.ylabel('Number of Recordings')
        plt.title('Number of Recordings per Emotion')
        plt.xticks(rotation=45)
        plt.show()

    def visualize_gender_distribution(self):
        gender_labels = list(self.gender_counts.keys())
        counts = list(self.gender_counts.values())
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=gender_labels, y=counts, palette="viridis")
        plt.xlabel('Gender')
        plt.ylabel('Number of Recordings')
        plt.title('Number of Recordings by Gender')
        plt.show()




class AudioPreprocessor(DataCleaner):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, verbose=True):
        # Initialize with data directory, emotions, and target sample rate
        self.data_dir = data_dir
        self.emotions = emotions
        self.target_length = target_length  # Target length for padding/truncating audio files
        self.verbose = verbose  # Add a verbose flag

        # Initialize DataLoader to load and clean the data
        self.X = []
        self.y = []
        self.sample_rate = sample_rate

        self.load_data()
        self.X = self.clean_data()  # Clean the loaded data

    def load_data(self):
        """
        Load audio files and store them in X.
        """
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(self.data_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            
            if self.verbose:
                print(f"Processing {len(wav_files)} files for emotion: {emotion}")

            for filename in wav_files:
                filepath = os.path.join(emotion_dir, filename)
                try:
                    # Load the audio file
                    audio, _ = librosa.load(filepath, sr=self.sample_rate)
                    self.X.append(audio)
                    self.y.append(i)

                except Exception as e:
                    if self.verbose:
                        print(f"Error processing file {filepath}: {e}")

        self.y = np.array(self.y)  # Convert labels to numpy array

    def pad_audio(self):
        """
        Pads or truncates the audio array to the target length.
        """
        padded_X = []
        for audio in self.X:
            if len(audio) > self.target_length:
                padded_audio = audio[:self.target_length]
            else:
                padded_audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
            padded_X.append(padded_audio)

        return np.array(padded_X)

    def get_data(self):
        """
        Returns the processed and padded data and labels.
        """
        self.X = self.pad_audio()
        return np.array(self.X), np.array(self.y)

class FeatureExtractor(AudioPreprocessor):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, verbose=True):
        super().__init__(data_dir, emotions, sample_rate, target_length, verbose)
        self.n_mfcc = n_mfcc
        self.features = None  # To store the extracted features

    def extract_features(self):
        extracted_features = []
        for audio in self.X:
            features = []

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfccs_mean = np.mean(mfccs, axis=1)
            features.extend(mfccs_mean)

            # Extract Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)

            # Extract Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            features.extend(spectral_contrast_mean)

            # Extract Zero Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            features.append(zero_crossing_rate_mean)

            # Extract Root Mean Square Energy
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            features.append(rms_mean)

            # Extract Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_centroid_mean = np.mean(spectral_centroid)
            features.append(spectral_centroid_mean)

            # Extract Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            features.append(spectral_bandwidth_mean)

            # Extract Spectral Roll-off
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            features.append(spectral_rolloff_mean)

            extracted_features.append(features)

        self.features = np.array(extracted_features)
        return self.features

    def get_features_and_labels(self):
        """
        Returns extracted features and corresponding numerical labels.
        """
        self.extract_features()
        return self.features, self.y
    

class EmotionLabeler(FeatureExtractor):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, verbose=True):
        """
        Initializes the EmotionLabeler, inheriting from FeatureExtractor.

        Args:
            data_dir (str): Path to the directory containing audio files.
            emotions (list): List of emotion labels.
            sample_rate (int): Target sample rate for audio processing.
            target_length (int): Desired length for audio padding/truncation.
            n_mfcc (int): Number of MFCC coefficients to extract.
            verbose (bool): Whether to print progress messages.
        """
        super().__init__(data_dir, emotions, sample_rate, target_length, n_mfcc, verbose)

        # Define the explicit mapping of emotions to numerical labels
        self.emotion_map = {
            'sad': 0,
            'happy': 1,
            'surprised': 2,
            'angry': 3,
            'calm': 4
        }

    def label_emotions(self):
        """
        Converts numerical labels back to their corresponding emotion names.

        Returns:
            list: A list of emotion labels corresponding to the numerical labels in self.y.
        """
        return [self.emotion_map[label] for label in self.y]

    def get_numerical_labels(self):
        """
        Returns the numerical labels.

        Returns:
            numpy.ndarray: The numerical labels (self.y).
        """
        return self.y


class DataSaver(EmotionLabeler):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, save_path="processed_data.csv", verbose=True):
        super().__init__(data_dir, emotions, sample_rate, target_length, n_mfcc, verbose)
        self.save_path = save_path

    def save_to_csv(self):
        features, labels = self.get_features_and_labels()
        df = pd.DataFrame(features)
        df['emotion'] = labels  # Save numerical labels instead of words
        df.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

    def split_data(self):
        features, labels = self.get_features_and_labels()
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    



class FeaturesEDA(DataSaver):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, save_path="processed_data.csv", verbose=True):
        super().__init__(data_dir, emotions, sample_rate, target_length, n_mfcc, save_path, verbose)
        self.df = pd.read_csv(self.save_path)  # Load the CSV file

    def get_info(self):
        """
        Displays basic information about the DataFrame.
        """
        print("DataFrame Info:")
        print(self.df.info())

    def get_statistics(self):
        """
        Displays basic statistics of the features in the DataFrame.
        """
        print("DataFrame Statistics:")
        print(self.df.describe())

    def get_head_tail(self):
        """
        Displays the first and last few rows of the DataFrame.
        """
        print("DataFrame Head:")
        print(self.df.head())
        print("\nDataFrame Tail:")
        print(self.df.tail())

    def plot_correlation_matrix(self):
        """
        Computes and plots the correlation matrix.
        """
        # Compute the correlation matrix
        corr_matrix = self.df.corr()
        
        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

        # Print a summary of correlation indices
        print("Correlation Index Summary:")
        for column in corr_matrix.columns:
            print(f"\nColumn: {column}")
            print(corr_matrix[column].sort_values(ascending=False))

    def perform_pca(self):
        """
        Performs PCA on the feature set and plots the results.
        """
        features = self.df.drop(columns=['emotion'])
        labels = self.df['emotion']

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(features_scaled)
        
        # Create a DataFrame for the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['emotion'] = labels

        # Plot the PCA results
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='PC1', y='PC2', hue='emotion', palette='viridis', data=pca_df, alpha=0.7)
        plt.title('PCA of Audio Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        # Print PCA summary
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"PCA Explained Variance Ratio (Cumulative): {np.cumsum(pca.explained_variance_ratio_)}")


class Modeling:
    def __init__(self, file_path, target_column, test_size=0.3, random_state=42):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        # Load preprocessed data
        self.df = pd.read_csv(file_path)

        # Separate features and target variable
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

        # Binarize the output for multiclass ROC curves
        self.y_bin = label_binarize(self.y, classes=np.unique(self.y))
        self.n_classes = self.y_bin.shape[1]

        # Use StratifiedShuffleSplit for stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        for train_index, test_index in sss.split(self.X, self.y):
            self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]

        # Binarize the test labels for ROC computation
        self.y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y))

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Initialize classifiers including KNN as the baseline model
        self.classifiers = {
            'KNN': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Random Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
            'CatBoost': CatBoostClassifier(verbose=0),
            'Random Forest': RandomForestClassifier()
        }

    def get_base_learners(self):
        """Return a list of base learners as (name, model) tuples for stacking."""
        return [(name, clf) for name, clf in self.classifiers.items()]

    def tune_classifiers(self):
        param_grid = {
            'KNN': {'n_neighbors': [3, 5, 7]},
            'AdaBoost': {'n_estimators': [50, 100]},
            'Random Boosting': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
            'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'CatBoost': {'depth': [3, 5], 'learning_rate': [0.01, 0.1], 'iterations': [50, 100]},
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        }

        results = {}

        for name, clf in self.classifiers.items():
            print(f"Tuning {name}...")
            grid_search = GridSearchCV(
                clf,
                param_grid.get(name, {}),
                scoring='f1_weighted',
                cv=3,  # Reduced number of folds for computational efficiency
                n_jobs=1  
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            results[name] = {
                'Best Parameters': grid_search.best_params_,
                'Best Score': grid_search.best_score_
            }
            print(f"{name} Best Parameters: {grid_search.best_params_}")
            print(f"{name} Best Score: {grid_search.best_score_}")

        return results


class Evaluation(Modeling):
    def __init__(self, file_path, target_column, test_size=0.3, random_state=42):
        super().__init__(file_path, target_column, test_size, random_state)
        self.results = {}
        self.meta_model = None

    def evaluate_models(self):
        """Train and evaluate each model separately."""
        for name, clf in self.classifiers.items():
            print(f"\nTraining {name}...")
            clf.fit(self.X_train_scaled, self.y_train)
            
            # Predict on training and test data
            y_pred_train = clf.predict(self.X_train_scaled)
            y_pred_test = clf.predict(self.X_test_scaled)
            
            # Store results
            self.results[name] = {
                'train': classification_report(self.y_train, y_pred_train, output_dict=True),
                'test': classification_report(self.y_test, y_pred_test, output_dict=True)
            }
            
            # Print classification report
            print(f"\n{name} Classification Report on Test Data:")
            print(classification_report(self.y_test, y_pred_test, zero_division=1))

            # Plot ROC curve and confusion matrix for models that support probability prediction
            self.plot_confusion_matrix(name, y_pred_test)
            if hasattr(clf, "predict_proba"):
                self.plot_multiclass_roc_curve(name, clf)

            # Visualize the model's performance
            self.plot_evaluation(name, self.results[name])

    def train_stacking_model(self):
        """Train a stacking model with KNN as the meta-learner."""
        print("\nTraining Stacking Model...")
        base_learners = self.get_base_learners()
        self.meta_model = StackingClassifier(
            estimators=base_learners,
            final_estimator=KNeighborsClassifier(),
            cv=5
        )
        self.meta_model.fit(self.X_train_scaled, self.y_train)

        # Evaluate the stacking model
        y_pred_train = self.meta_model.predict(self.X_train_scaled)
        y_pred_test = self.meta_model.predict(self.X_test_scaled)

        # Store results
        self.results['Stacking Model'] = {
            'train': classification_report(self.y_train, y_pred_train, output_dict=True),
            'test': classification_report(self.y_test, y_pred_test, output_dict=True)
        }

        # Print classification report
        print("\nStacking Model Classification Report on Test Data:")
        print(classification_report(self.y_test, y_pred_test, zero_division=1))

        # Plot ROC curve and confusion matrix for the stacking model
        self.plot_confusion_matrix('Stacking Model', y_pred_test)
        self.plot_multiclass_roc_curve('Stacking Model', self.meta_model)

        # Visualize the stacking model's performance
        self.plot_evaluation('Stacking Model', self.results['Stacking Model'])

    def plot_confusion_matrix(self, model_name, y_pred):
        """Plot confusion matrix for each model."""
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

    def plot_multiclass_roc_curve(self, model_name, clf):
        """Plot ROC curve for multiclass classification using One-vs-Rest approach."""
        y_prob = clf.predict_proba(self.X_test_scaled)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute ROC curve and ROC area for each class
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

    def plot_evaluation(self, model_name, results):
        """Plot evaluation metrics like Precision, Recall, and F1-score."""
        categories = list(results['test'].keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precision = [results['test'][cat]['precision'] for cat in categories]
        recall = [results['test'][cat]['recall'] for cat in categories]
        f1_score = [results['test'][cat]['f1-score'] for cat in categories]

        x = np.arange(len(categories))  # Label locations
        width = 0.2  # Bar width

        fig, ax = plt.subplots()
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1_score, width, label='F1 Score')

        ax.set_xlabel('Categories')
        ax.set_title(f'Evaluation Metrics - {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        plt.xticks(rotation=45)
        plt.show()

class ModelSaver:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def save_model(self, model_name, model):
        """Save the trained model to disk."""
        filename = f"{model_name}_model.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved as {filename}")

    def save_results(self):
        """Save evaluation results to a CSV file."""
        results_df = pd.DataFrame(self.evaluation.results).T
        results_df.to_csv('evaluation_results.csv', index=True)
        print("Evaluation results saved to evaluation_results.csv")



# class ModelComparisonPlotter:
#     def __init__(self, csv_file):
#         self.df = pd.read_csv(csv_file)

#     def _parse_metrics(self, metrics_str):
#         # Convert JSON string to dictionary
#         return json.loads(metrics_str.replace("'", "\""))

#     def _extract_metrics(self, metrics_type):
#         metrics_df = pd.DataFrame()
        
#         for index, row in self.df.iterrows():
#             # Assuming your model names are actually in the index or need to be set manually
#             model_name = self.df.index[index]  # Adjust if model names are elsewhere
#             result = row[metrics_type]
            
#             # You might need to extract data and build DataFrame accordingly
#             # Example:
#             metrics = eval(result)  # Be cautious with eval; ensure data is safe
#             metrics_df = metrics_df.append({'model': model_name, 'metrics': metrics}, ignore_index=True)
        
#         return metrics_df

#     def plot_metrics(self):
#         metrics_types = ['accuracy']  # Adjust as needed
        
#         fig, axes = plt.subplots(len(metrics_types), 1, figsize=(10, 5 * len(metrics_types)))
        
#         for i, metrics_type in enumerate(metrics_types):
#             metrics_df = self._extract_metrics(metrics_type=metrics_type)
#             metrics_df.plot(kind='bar', x='model', y=[f'train_{metrics_type}', f'test_{metrics_type}'], ax=axes[i])
#             axes[i].set_title(f'{metrics_type.capitalize()} Metrics')
        
#         plt.tight_layout()
#         plt.show()