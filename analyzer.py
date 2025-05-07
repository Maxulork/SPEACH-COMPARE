import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import time
import warnings
from sklearn.decomposition import PCA
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from scipy.stats import pearsonr
from scipy.signal import correlate
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimized DTW with Sakoe-Chiba band constraint
def dtw_distance(x, y, distance_function=None):
    if distance_function is None:
        distance_function = lambda a, b: np.linalg.norm(a - b)

    n, m = len(x), len(y)
    band_width = max(abs(n - m) + 15, int(0.15 * max(n, m)))

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band_width)
        j_end = min(m + 1, i + band_width)
        for j in range(j_start, j_end):
            cost = distance_function(x[i - 1], y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        min_val = min(
            dtw_matrix[i - 1, j],
            dtw_matrix[i, j - 1],
            dtw_matrix[i - 1, j - 1]
        )
        if min_val == dtw_matrix[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif min_val == dtw_matrix[i - 1, j]:
            i = i - 1
        else:
            j = j - 1

    path.reverse()

    class DTWResult:
        def __init__(self, distance, path):
            self.distance = distance
            self.index1 = [p[0] for p in path]
            self.index2 = [p[1] for p in path]

    return DTWResult(dtw_matrix[n, m], path)

class SpeechSimilarityAnalyzer:
    def __init__(self):
        print("Initializing Speech Similarity Analyzer...")

        # Audio parameters
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        self.target_length = 250

        # Create cache directory
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Load models
        self._load_model(cache_dir)
        self._load_stt_model(cache_dir)
        # Removed _load_siamese_text_model

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor().to(device)

        # Segment parameters
        self.segment_duration = 0.5
        self.segment_samples = int(self.sample_rate * self.segment_duration)

    def _load_model(self, cache_dir):
        print("Loading pre-trained model...")
        models = [
            "infinitejoy/wav2vec2-large-xls-r-300m-slovak",
            "facebook/wav2vec2-base-960h"
        ]
        for model_name in models:
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
                self.model = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir).to(device)
                print(f"Successfully loaded {model_name}")
                return
            except Exception as e:
                print(f"Local load failed: {e}")
                try:
                    self.processor = Wav2Vec2Processor.from_pretrained(
                        model_name, cache_dir=cache_dir, local_files_only=False
                    )
                    self.model = Wav2Vec2Model.from_pretrained(
                        model_name, cache_dir=cache_dir, local_files_only=False
                    ).to(device)
                    print(f"Downloaded and loaded {model_name}")
                    return
                except Exception as e2:
                    print(f"Download failed: {e2}")
        raise RuntimeError("Could not load any model.")

    def _load_stt_model(self, cache_dir):
        print("Loading STT model specifically for Slovak language...")
        try:
            # Try to load Slovak-specific model first
            slovak_models = [
                "infinitejoy/wav2vec2-large-xls-r-300m-slovak"
            ]

            for model_name in slovak_models:
                try:
                    self.stt_processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
                    self.stt_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir).to(device)
                    print(f"Successfully loaded Slovak STT model: {model_name}")
                    return
                except Exception as e:
                    print(f"Failed loading local Slovak model {model_name}: {e}")
                    try:
                        self.stt_processor = Wav2Vec2Processor.from_pretrained(
                            model_name, cache_dir=cache_dir, local_files_only=False
                        )
                        self.stt_model = Wav2Vec2ForCTC.from_pretrained(
                            model_name, cache_dir=cache_dir, local_files_only=False
                        ).to(device)
                        print(f"Downloaded and loaded Slovak STT model: {model_name}")
                        return
                    except Exception as e2:
                        print(f"Download failed for {model_name}: {e2}")

            # Fallback to multilingual model if Slovak-specific models fail
            print("Slovak-specific models failed, trying multilingual model...")
            self.stt_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir
            )
            self.stt_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir
            ).to(device)
            print("Successfully loaded fallback multilingual XLSR model")

        except Exception as e:
            print(f"All STT model load attempts failed: {e}")
            # Last resort - fall back to English model
            try:
                print("Attempting to load English model as last resort...")
                self.stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h",
                                                                       cache_dir=cache_dir)
                self.stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir).to(
                    device)
                print("Successfully loaded English STT model as fallback")
            except Exception as e2:
                print(f"All fallbacks failed: {e2}")
                raise RuntimeError("Could not load any STT model.")

    def load_audio(self, file_path):
        print(f"Loading audio: {file_path}")
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f"Audio duration: {len(audio) / self.sample_rate:.2f} seconds")
            audio = self.preprocess_audio(audio)
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

    def preprocess_audio(self, audio):
        if len(audio) == 0 or np.isnan(audio).any():
            raise ValueError("Invalid audio data")

        audio = librosa.effects.preemphasis(audio, coef=0.97)
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp * 0.95

        intervals = librosa.effects.split(audio, top_db=30)
        if len(intervals) < 2:
            intervals = librosa.effects.split(audio, top_db=20)

        audio_out = np.zeros_like(audio)
        for interval in intervals:
            audio_out[interval[0]:interval[1]] = audio[interval[0]:interval[1]]

        if np.sum(np.abs(audio_out)) < 1e-6:
            print("Warning: No speech detected, using original")
            return audio

        return audio_out

    def create_spectrogram(self, audio):
        print("Creating spectrogram...")
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=8000
        )

        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_normalized = (spectrogram_db - spectrogram_db.min()) / (
                spectrogram_db.max() - spectrogram_db.min() + 1e-10)

        if spectrogram_normalized.shape[1] != self.target_length:
            resized_spec = np.zeros((self.n_mels, self.target_length))

            time_scale = spectrogram_normalized.shape[1] / self.target_length
            for t in range(self.target_length):
                src_pos = t * time_scale
                src_idx = int(src_pos)

                if src_idx >= spectrogram_normalized.shape[1] - 1:
                    src_idx = spectrogram_normalized.shape[1] - 2

                alpha = src_pos - src_idx
                resized_spec[:, t] = (1 - alpha) * spectrogram_normalized[:, src_idx] + \
                                     alpha * spectrogram_normalized[:, src_idx + 1]

            spectrogram_normalized = resized_spec

        return spectrogram_normalized

    def extract_hubert_features(self, audio):
        print("Extracting speech features...")
        try:
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            input_values = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_values
            input_values = input_values.to(device)

            with torch.no_grad():
                outputs = self.model(input_values)
                features = outputs.last_hidden_state.squeeze().cpu().numpy()

            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            features = (features - mean) / std

            if features.shape[0] > self.target_length:
                indices = np.linspace(0, features.shape[0] - 1, self.target_length, dtype=int)
                features = features[indices]
            elif features.shape[0] < self.target_length:
                resized_features = np.zeros((self.target_length, features.shape[1]))
                for i in range(features.shape[1]):
                    x_orig = np.linspace(0, 1, features.shape[0])
                    x_new = np.linspace(0, 1, self.target_length)
                    resized_features[:, i] = np.interp(x_new, x_orig, features[:, i])
                features = resized_features

            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros((self.target_length, self.model.config.hidden_size))

    def compute_context_dtw_similarity(self, features1, features2):
        print("Computing context-aware DTW similarity...")
        try:
            if features1.shape[1] > 64:
                pca = PCA(n_components=64)
                combined = np.nan_to_num(np.vstack([features1, features2]))
                pca.fit(combined)
                features1 = pca.transform(np.nan_to_num(features1))
                features2 = pca.transform(np.nan_to_num(features2))

            def context_distance(x, y):
                eps = 1e-8
                x_norm = x / (np.linalg.norm(x) + eps)
                y_norm = y / (np.linalg.norm(y) + eps)
                cos_sim = np.dot(x_norm, y_norm)
                eucl_dist = np.linalg.norm(x - y)
                return 0.7 * eucl_dist + 0.3 * (1 - cos_sim)

            alignment = dtw_distance(
                features1,
                features2,
                distance_function=context_distance
            )

            path_length = len(alignment.index1)
            if path_length == 0:
                return 20.0

            distance = alignment.distance / path_length
            similarity = 100 / (1 + np.exp(0.5 * distance - 2.5))
        except Exception as e:
            print(f"DTW calculation error: {e}")
            similarity = 20.0

        return min(100.0, max(0.0, similarity))

    def speech_to_text(self, audio):
        print("Converting speech to text with Slovak-optimized model...")
        try:
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            max_length = 30 * self.sample_rate

            if len(audio) > max_length:
                print("Long audio detected, processing in chunks...")
                chunks = [audio[i:i + max_length] for i in range(0, len(audio), max_length)]
                transcriptions = []

                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i + 1}/{len(chunks)}...")
                    input_values = self.stt_processor(chunk, sampling_rate=self.sample_rate,
                                                      return_tensors="pt").input_values
                    input_values = input_values.to(device)

                    with torch.no_grad():
                        logits = self.stt_model(input_values).logits

                    predicted_ids = torch.argmax(logits, dim=-1)
                    chunk_transcription = self.stt_processor.batch_decode(predicted_ids)[0]
                    transcriptions.append(chunk_transcription)

                transcription = " ".join(transcriptions)
            else:
                input_values = self.stt_processor(audio, sampling_rate=self.sample_rate,
                                                  return_tensors="pt").input_values
                input_values = input_values.to(device)

                with torch.no_grad():
                    logits = self.stt_model(input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.stt_processor.batch_decode(predicted_ids)[0]

            transcription = transcription.lower()
            transcription = self._fix_slovak_diacritics(transcription)

            print(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            print(f"STT error: {e}")
            return ""

    def _fix_slovak_diacritics(self, text):
        replacements = {
            "a´": "á", "e´": "é", "i´": "í", "o´": "ó", "u´": "ú",
            "c´": "č", "s´": "š", "z´": "ž", "l´": "ľ", "t´": "ť",
            "d´": "ď", "n´": "ň", "r´": "ŕ"
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def compute_text_similarity(self, text1, text2):
        print("Computing string similarity (Levenshtein)...")
        norm_text1 = text1.replace("_", "").replace(" ", "")
        norm_text2 = text2.replace("_", "").replace(" ", "")
        if norm_text1 == norm_text2:
            return 100.0
        max_len = max(len(norm_text1), len(norm_text2))
        if max_len == 0:
            return 100.0
        dist = levenshtein_distance(norm_text1, norm_text2)
        similarity = (1 - dist / max_len) * 100
        return max(0.0, min(100.0, similarity))

    def compute_phonetic_similarity(self, audio1, audio2):
        print("Computing phonetic similarity...")

        hubert_features1 = self.extract_hubert_features(audio1)
        hubert_features2 = self.extract_hubert_features(audio2)
        hubert_sim = self.compute_context_dtw_similarity(hubert_features1, hubert_features2)

        mfccs1 = librosa.feature.mfcc(
            y=audio1,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            htk=True
        )
        delta1_mfccs1 = librosa.feature.delta(mfccs1, order=1)
        delta2_mfccs1 = librosa.feature.delta(mfccs1, order=2)

        mfccs2 = librosa.feature.mfcc(
            y=audio2,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            htk=True
        )
        delta1_mfccs2 = librosa.feature.delta(mfccs2, order=1)
        delta2_mfccs2 = librosa.feature.delta(mfccs2, order=2)

        combined_deltas1 = np.vstack([mfccs1, delta1_mfccs1, delta2_mfccs1])
        combined_deltas2 = np.vstack([mfccs2, delta1_mfccs2, delta2_mfccs2])

        norm1 = np.sqrt(np.sum(combined_deltas1 ** 2, axis=0) + 1e-8)
        norm2 = np.sqrt(np.sum(combined_deltas2 ** 2, axis=0) + 1e-8)
        combined_deltas1 = combined_deltas1 / norm1
        combined_deltas2 = combined_deltas2 / norm2

        mfcc_sim = 40.0
        min_len = min(combined_deltas1.shape[1], combined_deltas2.shape[1])
        if min_len > 10:
            deltas1_trimmed = combined_deltas1[:, :min_len]
            deltas2_trimmed = combined_deltas2[:, :min_len]

            def mfcc_distance(x, y):
                weights = np.exp(-0.1 * np.arange(len(x)))
                return np.sqrt(np.sum(weights * (x - y) ** 2))

            mfcc_dist = dtw_distance(
                deltas1_trimmed.T,
                deltas2_trimmed.T,
                distance_function=mfcc_distance
            )
            mfcc_sim = 100 / (1 + np.exp(0.1 * mfcc_dist.distance / min_len))

        formants1 = librosa.effects.harmonic(audio1)
        formants2 = librosa.effects.harmonic(audio2)
        formant_sim = 50.0
        if len(formants1) > 0 and len(formants2) > 0:
            min_len = min(len(formants1), len(formants2))
            formants1 = formants1[:min_len]
            formants2 = formants2[:min_len]
            formant_corr, _ = pearsonr(formants1, formants2)
            formant_sim = 50.0 + 50.0 * max(0, formant_corr)

        pitch1, _ = librosa.piptrack(y=audio1, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length)
        pitch2, _ = librosa.piptrack(y=audio2, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length)
        pitch1 = np.max(pitch1, axis=0)
        pitch2 = np.max(pitch2, axis=0)
        min_len = min(len(pitch1), len(pitch2))
        pitch1 = pitch1[:min_len]
        pitch2 = pitch2[:min_len]
        pitch_sim = 50.0
        if min_len > 10:
            pitch_corr, _ = pearsonr(pitch1, pitch2)
            pitch_sim = 50.0 + 50.0 * max(0, pitch_corr)

        energy1 = librosa.feature.rms(y=audio1, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        energy2 = librosa.feature.rms(y=audio2, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        min_len = min(len(energy1), len(energy2))
        energy1 = energy1[:min_len]
        energy2 = energy2[:min_len]
        energy_sim = 50.0
        if min_len > 10:
            energy_corr, _ = pearsonr(energy1, energy2)
            energy_sim = 50.0 + 50.0 * max(0, energy_corr)

        text1 = self.speech_to_text(audio1)
        text2 = self.speech_to_text(audio2)
        print(f"Transcriptions - Audio 1: '{text1}', Audio 2: '{text2}'")
        text_sim = self.compute_text_similarity(text1, text2)

        return {
            "hubert_sim": hubert_sim,
            "mfcc_sim": mfcc_sim,
            "formant_sim": formant_sim,
            "pitch_sim": pitch_sim,
            "energy_sim": energy_sim,
            "text_sim": text_sim,
            "transcription1": text1,
            "transcription2": text2
        }, hubert_features1, hubert_features2

    def extract_spectral_features(self, spectrogram):
        if spectrogram is None or np.isnan(spectrogram).any():
            return np.zeros(256)

        spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            features = self.feature_extractor(spec_tensor)
            features = features.cpu().numpy().squeeze()

        if np.max(np.abs(features)) > 0:
            features = features / np.max(np.abs(features))

        return features

    def compute_spectral_similarity(self, audio1, audio2):
        print("Computing spectral similarity...")
        try:
            spec1 = self.create_spectrogram(audio1)
            spec2 = self.create_spectrogram(audio2)

            spec_features1 = self.extract_spectral_features(spec1)
            spec_features2 = self.extract_spectral_features(spec2)

            spectral_sim = (1 - cosine(spec_features1, spec_features2)) * 100
        except Exception as e:
            print(f"Error computing spectral similarity: {e}")
            spectral_sim = 50.0
            spec1 = np.zeros((self.n_mels, self.target_length))
            spec2 = np.zeros((self.n_mels, self.target_length))

        return max(0.0, min(100.0, spectral_sim)), spec1, spec2

    def visualize_comparison(self, spec1, spec2, features1, features2, similarity):
        try:
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.imshow(spec1, aspect='auto', origin='lower', cmap='viridis')
            plt.title('Spectrogram 1')
            plt.ylabel('Mel Frequency Bin')
            plt.colorbar(format='%.1f')

            plt.subplot(2, 2, 2)
            plt.imshow(spec2, aspect='auto', origin='lower', cmap='viridis')
            plt.title('Spectrogram 2')
            plt.colorbar(format='%.1f')

            pca = PCA(n_components=2)
            try:
                features1_clean = np.nan_to_num(features1)
                features2_clean = np.nan_to_num(features2)

                pca.fit(np.vstack([features1_clean, features2_clean]))
                features1_pca = pca.transform(features1_clean)
                features2_pca = pca.transform(features2_clean)

                plt.subplot(2, 2, 3)
                plt.scatter(features1_pca[:, 0], features1_pca[:, 1],
                            s=10, alpha=0.7, c=np.arange(len(features1_pca)), cmap='Blues')
                plt.colorbar(label='Time')
                plt.title('Feature Space Audio 1')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')

                plt.subplot(2, 2, 4)
                plt.scatter(features2_pca[:, 0], features2_pca[:, 1],
                            s=10, alpha=0.7, c=np.arange(len(features2_pca)), cmap='Oranges')
                plt.colorbar(label='Time')
                plt.title('Feature Space Audio 2')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
            except Exception as e:
                print(f"Error in visualization: {e}")
                plt.subplot(2, 2, 3)
                plt.text(0.5, 0.5, 'Feature visualization unavailable', ha='center')
                plt.subplot(2, 2, 4)
                plt.text(0.5, 0.5, 'Feature visualization unavailable', ha='center')

            plt.suptitle(f'Similarity (Word & Pronunciation): {similarity:.2f}%', fontsize=16)
            plt.tight_layout()

            plt.savefig('speech_comparison.png')
            print("Visualization saved as 'speech_comparison.png'")
            plt.close()
        except Exception as e:
            print(f"Visualization error: {e}")

    def color_code_transcription(self, text1, text2):
        """Compare two transcriptions and color-code the first based on an aligned match."""
        # Normalize for alignment (optional, depending on if you want to ignore spaces/underscores here too)
        norm_text1 = text1
        norm_text2 = text2

        # Build Levenshtein distance matrix
        m, n = len(norm_text1), len(norm_text2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if norm_text1[i - 1].lower() == norm_text2[j - 1].lower():
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) + 1

        # Backtrack to find alignment
        aligned1 = []
        aligned2 = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and norm_text1[i - 1].lower() == norm_text2[j - 1].lower():
                aligned1.append(norm_text1[i - 1])
                aligned2.append(norm_text2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i - 1, j] + 1 == dp[i, j]):
                aligned1.append(norm_text1[i - 1])
                aligned2.append('-')
                i -= 1
            elif j > 0 and (i == 0 or dp[i, j - 1] + 1 == dp[i, j]):
                aligned1.append('-')
                aligned2.append(norm_text2[j - 1])
                j -= 1
            else:
                aligned1.append(norm_text1[i - 1])
                aligned2.append(norm_text2[j - 1])
                i -= 1
                j -= 1
        aligned1.reverse()
        aligned2.reverse()

        # Color-code based on alignment
        ansi_colored = ""
        html_colored = "<span>"
        for c1, c2 in zip(aligned1, aligned2):
            if c1 == '-':  # Skip gaps in text1
                continue
            if c1.isspace():
                ansi_colored += c1
                html_colored += c1
            elif c2 != '-' and c1.lower() == c2.lower():
                ansi_colored += f"\033[92m{c1}\033[0m"  # Green
                html_colored += f'<span style="color:green">{c1}</span>'
            else:
                ansi_colored += f"\033[91m{c1}\033[0m"  # Red
                html_colored += f'<span style="color:red">{c1}</span>'
        html_colored += "</span>"

        return {
            "ansi": ansi_colored,
            "html": html_colored,
            "plain": text1,
            "transcription2": text2
        }

    def compute_similarity(self, audio1, audio2, visualize=True):
        print("\nComputing similarity...")
        start_time = time.time()

        metrics, features1, features2 = self.compute_phonetic_similarity(audio1, audio2)

        hubert_sim = metrics["hubert_sim"]
        mfcc_sim = metrics["mfcc_sim"]
        formant_sim = metrics["formant_sim"]
        pitch_sim = metrics["pitch_sim"]
        energy_sim = metrics["energy_sim"]
        text_sim = metrics["text_sim"]
        transcription1 = metrics["transcription1"]
        transcription2 = metrics["transcription2"]

        colored_transcription = self.color_code_transcription(transcription1, transcription2)

        spectral_sim, spec1, spec2 = self.compute_spectral_similarity(audio1, audio2)
        alignment_sim = self.compute_context_dtw_similarity(features1, features2)

        timing_sim = 50.0
        try:
            onset1 = librosa.onset.onset_strength(y=audio1, sr=self.sample_rate)
            onset2 = librosa.onset.onset_strength(y=audio2, sr=self.sample_rate)
            min_len = min(len(onset1), len(onset2))
            if min_len > 10:
                onset_corr, _ = pearsonr(onset1[:min_len], onset2[:min_len])
                timing_sim = 50.0 + 50.0 * max(0, onset_corr)
        except Exception as e:
            print(f"Timing similarity error: {e}")

        corr = correlate(audio1, audio2, mode='full')
        corr_sim = 50.0
        if len(corr) > 0:
            corr_max = np.max(np.abs(corr)) / (np.sqrt(np.sum(audio1 ** 2) * np.sum(audio2 ** 2)) + 1e-8)
            corr_sim = 100.0 * min(1.0, max(0.0, corr_max))

        print(f"Metrics:")
        print(f"  Hubert (Word): {hubert_sim:.2f}%")
        print(f"  MFCC (Pron): {mfcc_sim:.2f}%")
        print(f"  Formant (Pron): {formant_sim:.2f}%")
        print(f"  Pitch (Pron): {pitch_sim:.2f}%")
        print(f"  Energy (Pron): {energy_sim:.2f}%")
        print(f"  Text Similarity (Word): {text_sim:.2f}%")
        print(f"  Spectral: {spectral_sim:.2f}%")
        print(f"  Alignment: {alignment_sim:.2f}%")
        print(f"  Timing (Onset): {timing_sim:.2f}%")
        print(f"  Cross-Correlation: {corr_sim:.2f}%")

        # Adjusted weights: 20% Hubert, 80% Text Similarity
        word_score = 0.2 * hubert_sim + 0.8 * text_sim
        pron_score = (
                0.4 * mfcc_sim +
                0.2 * formant_sim +
                0.2 * pitch_sim +
                0.2 * energy_sim
        )
        linguistic_similarity = 0.5 * word_score + 0.5 * pron_score
        linguistic_similarity = 100 / (1 + np.exp(-0.1 * (linguistic_similarity - 50)))

        print(f"Word Score: {word_score:.2f}%")
        print(f"Pronunciation Score: {pron_score:.2f}%")
        print(f"Final Similarity (Same Word & Pronunciation): {linguistic_similarity:.2f}%")
        print(f"Computation time: {time.time() - start_time:.2f} seconds")

        if visualize:
            self.visualize_comparison(spec1, spec2, features1, features2, linguistic_similarity)

        return linguistic_similarity, colored_transcription

    def analyze_files(self, file1, file2, visualize=True):
        print(f"\nAnalyzing files: {file1} and {file2}")
        audio1 = self.load_audio(file1)
        audio2 = self.load_audio(file2)

        if audio1 is None or audio2 is None:
            return "Error: Could not load one or both audio files"

        similarity, colored_transcription = self.compute_similarity(audio1, audio2, visualize)

        print("\nTranscription Comparison:")
        print(f"First Audio Transcription (matches in green, mismatches in red): {colored_transcription['ansi']}")
        print(f"Second Audio Transcription: {colored_transcription['transcription2']}")
        with open("transcription_comparison.html", "w", encoding="utf-8") as f:
            f.write(f"<p>First Audio Transcription: {colored_transcription['html']}</p>")
            f.write(f"<p>Second Audio Transcription: {colored_transcription['transcription2']}</p>")
        print("HTML transcription comparison saved as 'transcription_comparison.html'")

        return similarity, colored_transcription

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)

def main():
    print("Slovak Speech Similarity Analyzer")
    print("=" * 50)

    try:
        analyzer = SpeechSimilarityAnalyzer()

        while True:
            file1 = input("\nEnter path to first audio file: ")
            if os.path.exists(file1):
                break
            print("File not found. Please try again.")

        while True:
            file2 = input("Enter path to second audio file: ")
            if os.path.exists(file2):
                break
            print("File not found. Please try again.")

        visualize = input("Visualize comparison? (yes/no, default: yes): ").lower() != "no"
        print("\nStarting analysis...")
        overall_start_time = time.time()

        result = analyzer.analyze_files(file1, file2, visualize)

        if isinstance(result, str):
            print(result)
        else:
            similarity, colored_transcription = result
            print("\n" + "=" * 50)
            print(f"Overall similarity (Same Word & Pronunciation): {similarity:.2f}%")
            print(f"Total processing time: {time.time() - overall_start_time:.2f} seconds")
            print("=" * 50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure required libraries are installed")

if __name__ == "__main__":
    main()
