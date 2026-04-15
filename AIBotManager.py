import time
import numpy as np
import os
import sys

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from scipy.signal import find_peaks, stft
from PyQt5.QtCore import QThread, pyqtSignal

# Attempt to import Tensorflow for model logic
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import csv
    import librosa
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    HAS_TF = True
except ImportError:
    HAS_TF = False

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class AIAugmenter:
    """Provides acoustic data augmentation to increase training robustness."""
    @staticmethod
    def augment(audio, sr):
        """Returns a list of augmented versions of the input audio."""
        aug_list = [audio] # Original
        
        # 1. Gain variations
        aug_list.append(audio * 0.6)
        aug_list.append(audio * 1.4)
        
        # 2. Add subtle white noise
        noise = np.random.randn(len(audio))
        aug_list.append(audio + 0.005 * noise)
        
        # 3. Time shifting
        shift = int(0.1 * sr) # 100ms shift
        aug_list.append(np.roll(audio, shift))
        
        return aug_list

class CustomExpertModel:
    """Wrapper for the tailored scikit-learn classifier built on YAMNet embeddings."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.clf = None
        self.mean = None
        self.std = None
        self.label_encoder = LabelEncoder()
        self.classes_ = []
        self.last_trained = None

    def save(self):
        if self.clf:
            joblib.dump({
                'clf': self.clf,
                'mean': self.mean,
                'std': self.std,
                'le': self.label_encoder,
                'classes': self.classes_,
                'timestamp': time.time()
            }, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.clf = data['clf']
                self.mean = data.get('mean')
                self.std = data.get('std')
                self.label_encoder = data.get('le', LabelEncoder())
                self.classes_ = data['classes']
                self.last_trained = data.get('timestamp')
                return True
            except:
                return False
        return False

    def train(self, X, y, log_signal=None):
        """X: embeddings (n_samples, 1024), y: labels (n_samples)"""
        # v35.13: Label Encoding (The 'isnan' Slayer)
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_.tolist()
        
        if len(self.classes_) < 2:
            raise ValueError("Need at least 2 different classes to train.")
        
        # Manual Standardization
        X_num = np.ascontiguousarray(X, dtype=np.float64)
        self.mean = np.mean(X_num, axis=0)
        self.std = np.std(X_num, axis=0)
        self.std[self.std == 0] = 1.0
        X_scaled = (X_num - self.mean) / self.std
        
        if log_signal:
            log_signal.emit(f"   [DEBUG] Final Training Shape: {X_scaled.shape}, Dtype: {X_scaled.dtype}")
        
        # MLP Classifier
        self.clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True)
        self.clf.fit(X_scaled, y_encoded)
        self.save()

    def predict_proba(self, embeddings):
        if self.clf is None or self.mean is None:
            return None
        # Apply manual scaling
        X_num = np.ascontiguousarray(embeddings, dtype=np.float64)
        X_scaled = (X_num - self.mean) / self.std
        return self.clf.predict_proba(X_scaled)

class AIBotManager:
    """Manages the AI lifecycle: Discovery and Targeted Listening."""
    def __init__(self, main_app):
        self.main_app = main_app
        self.is_active = False

class AIBotInferenceThread(QThread):
    """
    Background thread for EXPERT LEARNING and Iterative Discovery.
    """
    progress = pyqtSignal(int, str)      
    log_signal = pyqtSignal(str)         
    discovery_event = pyqtSignal(float, float, str) 
    detection_found = pyqtSignal(float, float, str, float) 
    finished = pyqtSignal(str)           
    error = pyqtSignal(str)

    def __init__(self, mode, filepath, sr, n_channels, audio_data, spec_data, spec_settings, window_xlim):
        super().__init__()
        self.mode = mode # 'discovery', 'listening_all', 'learning'
        self.filepath = filepath
        self.sr = sr
        self.n_channels = n_channels
        self.audio_data = audio_data
        self.spec_data = spec_data 
        self.spec_settings = spec_settings
        self.window_xlim = window_xlim # (start_s, end_s)
        self.refine_nfft = 4096 
        self._is_cancelled = False
        
        # YAMNet resources
        self.yamnet_model = None
        self.yamnet_classes = []
        self.yamnet_class_map_path = get_resource_path('yamnet_class_map.csv')
        
        # Custom Learning Data
        self.training_annotations = [] # List of dicts from AnnotatorApp
        self.custom_model_path = get_resource_path('custom_expert.joblib')

    def run(self):
        try:
            if self.audio_data is None:
                self.error.emit("No audio data available.")
                return

            if self.mode == 'yamnet_discovery':
                self._run_yamnet_discovery()
            elif self.mode == 'learning':
                self._run_expert_learning()
            else:
                self.error.emit(f"Unsupported AI Mode: {self.mode}")

        except Exception as e:
            import traceback
            self.error.emit(f"AI Error: {str(e)}\n{traceback.format_exc()}")

    def cancel(self):
        self._is_cancelled = True

    def _load_yamnet_model(self):
        if not HAS_TF:
            raise ImportError("TensorFlow, TensorFlow Hub, or Librosa is not installed.")
        
        if self.yamnet_model is None:
            self.log_signal.emit(">>> [YAMNet] Loading model from TensorFlow Hub. This may take a moment on first run...")
            self.progress.emit(5, "Loading YAMNet model...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            self.yamnet_classes = []
            if os.path.exists(self.yamnet_class_map_path):
                with open(self.yamnet_class_map_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.yamnet_classes.append(row['display_name'])
            else:
                self.log_signal.emit(f">>> [WARNING] yamnet_class_map.csv not found at {self.yamnet_class_map_path}!")
                
            self.log_signal.emit(">>> [YAMNet] Model loaded successfully.")

    def _run_expert_learning(self):
        """
        Processes existing labels, generates augmented embeddings, and trains the custom classifier.
        """
        self.log_signal.emit(">>> [AI TRAINING] Starting Expert System Learning Phase...")
        if not self.training_annotations:
            self.error.emit("No labels found to train on.")
            return

        self._load_yamnet_model()
        
        X_list = []
        y_list = []
        
        total = len(self.training_annotations)
        for i, ann in enumerate(self.training_annotations):
            if self._is_cancelled: break
            
            label = ann['label']
            start_t = ann['start']
            end_t = ann['end']
            
            self.progress.emit(int(10 + (i/total)*70), f"Extracting features: {label}")
            
            # v35.9: Contextual Padding for short events
            # Instead of silence, we grab audio from around the label if available
            label_dur = end_t - start_t
            min_dur = 1.05 # YAMNet window
            
            if label_dur < min_dur:
                # Center the window on the label and expand outward
                needed = (min_dur - label_dur) / 2
                padded_start = max(0, start_t - needed)
                padded_end = min(self.audio_data.shape[0] / self.sr, end_t + needed)
                
                st_idx = int(padded_start * self.sr)
                en_idx = int(padded_end * self.sr)
                clip = self.audio_data[st_idx:en_idx, 0].astype(np.float32)
                
                # Final check if still too short (start/end of file)
                if len(clip) < int(min_dur * self.sr):
                    clip = np.pad(clip, (0, int(min_dur * self.sr) - len(clip)))
            else:
                st_idx = int(start_t * self.sr)
                en_idx = int(end_t * self.sr)
                clip = self.audio_data[st_idx:en_idx, 0].astype(np.float32)
            
            # Augmentation
            variations = AIAugmenter.augment(clip, self.sr)
            
            for v_audio in variations:
                # Resample for YAMNet
                audio_16k = librosa.resample(v_audio, orig_sr=self.sr, target_sr=16000)
                # Max scale
                m = np.max(np.abs(audio_16k))
                if m > 1e-6: audio_16k /= m
                
                # Get embeddings
                _, embeddings, _ = self.yamnet_model(audio_16k)
                emb_np = embeddings.numpy()
                
                if emb_np.shape[0] > 0:
                    # Average embeddings for the clip to get one 1024-D vector
                    avg_emb = np.mean(emb_np, axis=0)
                    
                    # v35.9: Strict shape and type validation
                    if avg_emb.shape == (1024,):
                        X_list.append(avg_emb.astype(np.float32))
                        y_list.append(str(label))
                    else:
                        self.log_signal.emit(f"   [SKIP] Unexpected embedding shape {avg_emb.shape} for {label}")

        if len(X_list) == 0:
            self.error.emit("Failed to extract any 1024-D features. Are your labels at least 1 second long?")
            return

        self.log_signal.emit(f">>> [AI TRAINING] Collected {len(X_list)} candidate vectors. Constructing numeric matrix...")
        
        try:
            # v35.10: Final 'Object Dtype' Killer - Pre-allocated float32 matrix
            # This is the most robust way to ensure we never get a non-numeric array
            num_samples = len(X_list)
            X = np.zeros((num_samples, 1024), dtype=np.float32)
            y = np.array(y_list)
            
            valid_indices = []
            for idx, sample in enumerate(X_list):
                try:
                    # Enforce numeric conversion and check for valid numbers per row
                    row_data = np.asarray(sample, dtype=np.float32)
                    if row_data.shape == (1024,):
                        # Manual finiteness check to avoid problematic numpy ufunc calls
                        if not np.any(np.isnan(row_data)) and not np.any(np.isinf(row_data)):
                             X[idx, :] = row_data
                             valid_indices.append(idx)
                except:
                    continue
            
            # Crop to only the successful rows
            if not valid_indices:
                 self.error.emit("No valid features extracted. Please ensure your labels are on clear audio sections.")
                 return

            X_final = X[valid_indices].copy()
            y_final = np.array([y[i] for i in valid_indices])
            
            self.log_signal.emit(f">>> [AI TRAINING] Data verified. {len(X_final)} valid samples for neural network training.")
            
            if len(np.unique(y_final)) < 2:
                self.error.emit("Not enough valid data. Training requires at least 2 DIFFERENT classes (e.g., 'Drone' and 'Background').")
                return

            self.progress.emit(85, "Optimizing Neural Network (MLP)...")
            model = CustomExpertModel(self.custom_model_path)
            model.train(X_final, y_final, log_signal=self.log_signal)
            self.log_signal.emit(f">>> [SUCCESS] Expert Model trained successfully.")
            self.progress.emit(100, "Training Complete.")
            self.finished.emit('learning')
        except Exception as e:
            self.error.emit(f"ML Processing Error: {str(e)}")

    def _run_yamnet_discovery(self):
        self.log_signal.emit(">>> [AI EXPERT] Initiating YAMNet Audio Analysis...")
        self._load_yamnet_model()
        
        start_s, end_s = self.window_xlim
        total_dur = end_s - start_s + 1e-10
        st_idx = int(start_s * self.sr)
        en_idx = int(end_s * self.sr)
        
        self.progress.emit(10, "Extracting and standardizing audio...")
        audio_slice = self.audio_data[st_idx:en_idx, 0].astype(np.float32)
        
        if self.sr != 16000:
            self.progress.emit(20, "Resampling to 16kHz for YAMNet...")
            audio_16k = librosa.resample(audio_slice, orig_sr=self.sr, target_sr=16000)
        else:
            audio_16k = audio_slice
            
        max_val = np.max(np.abs(audio_16k))
        if max_val > 0:
             audio_16k = audio_16k / max_val
        
        self.progress.emit(40, "Running YAMNet inference...")
        scores, embeddings, _ = self.yamnet_model(audio_16k)
        scores = scores.numpy() 
        embeddings = embeddings.numpy()
        
        # v35.0: Load Custom Expert if exists
        custom_model = CustomExpertModel(self.custom_model_path)
        has_custom = custom_model.load()
        if has_custom:
            self.log_signal.emit(f">>> [AI EXPERT] Custom model loaded (Trained: {time.ctime(custom_model.last_trained)})")
        
        frame_step = 0.48
        self.progress.emit(80, "Processing predictions...")
        
        final_events = [] 
        active_trackers = {} # label -> [start, end, max_score]
        threshold = 0.1      
        gap_tolerance = 0.1  
        
        for i, frame_scores in enumerate(scores):
            top_class_idx = np.argmax(frame_scores)
            top_score = frame_scores[top_class_idx]
            
            frame_start_t = start_s + (i * frame_step)
            frame_end_t = frame_start_t + 0.96 
            
            if top_score > threshold and self.yamnet_classes:
                label_name = self.yamnet_classes[top_class_idx]
                if "Silence" in label_name:
                    continue
                
                if label_name in active_trackers:
                    track = active_trackers[label_name]
                    if frame_start_t <= track[1] + gap_tolerance:
                        track[1] = max(track[1], frame_end_t)
                        track[2] = max(track[2], top_score)
                    else:
                        final_events.append((track[0], track[1], label_name, track[2]))
                        active_trackers[label_name] = [frame_start_t, frame_end_t, top_score]
                else:
                    active_trackers[label_name] = [frame_start_t, frame_end_t, top_score]

            # v35.1: Custom Expert Prediction (User-Taught AI)
            if has_custom:
                frame_emb = embeddings[i].reshape(1, -1)
                custom_probs = custom_model.predict_proba(frame_emb)[0]
                best_custom_idx = np.argmax(custom_probs)
                best_custom_score = custom_probs[best_custom_idx]
                
                if best_custom_score > 0.5: # User-defined priority threshold
                    c_label = custom_model.classes_[best_custom_idx]
                    # Tag as Expert to distinguish from standard YAMNet
                    expert_label = f"⭐ {c_label}" 
                    
                    if expert_label in active_trackers:
                        track = active_trackers[expert_label]
                        if frame_start_t <= track[1] + gap_tolerance:
                            track[1] = max(track[1], frame_end_t)
                            track[2] = max(track[2], best_custom_score)
                        else:
                            final_events.append((track[0], track[1], expert_label, track[2]))
                            active_trackers[expert_label] = [frame_start_t, frame_end_t, best_custom_score]
                    else:
                        active_trackers[expert_label] = [frame_start_t, frame_end_t, best_custom_score]
                     
        for label, track in active_trackers.items():
            final_events.append((track[0], track[1], label, track[2]))
             
        final_events.sort(key=lambda x: x[0])
             
        for start_t, end_t, label, score in final_events:
            self.discovery_event.emit(start_t, end_t, f"{label}|{score}")
            self.log_signal.emit(f"MATCH: Merged '{label}' ({score:.2f}) from {start_t:.2f}s to {end_t:.2f}s")
             
        self.progress.emit(100, "YAMNet Discovery Phase Finished.")
        self.finished.emit('yamnet_discovery')

    def _estimate_noise_floor(self):
        total_samples = self.audio_data.shape[0]
        sample_win = int(self.sr)
        samples = []
        for i in range(0, total_samples - sample_win, int(self.sr * 30)):
            chunk = self.audio_data[i : i + sample_win, 0]
            _, _, Sxx = stft(chunk / 32768.0, fs=self.sr, nperseg=self.refine_nfft)
            samples.append(np.median(np.abs(Sxx), axis=1))
        return np.median(samples, axis=0) if samples else np.zeros((self.refine_nfft // 2 + 1))

    def _classify_multi_label(self, audio_slice):
        if len(audio_slice) < 100: return []
        results = []
        dur = len(audio_slice) / self.sr
        audio_norm = audio_slice / (np.max(np.abs(audio_slice)) + 1e-6)
        fft_data = np.abs(np.fft.rfft(audio_norm))
        freqs = np.fft.rfftfreq(len(audio_norm), 1/self.sr)
        peaks, _ = find_peaks(fft_data, height=np.max(fft_data)*0.3)
        if len(peaks) >= 3:
            results.append(("Vehicle", 0.88, 0, dur))
        if dur > 1.0:
            results.append(("Human Speech", 0.75, 0.2, min(0.8, dur)))
        return results
