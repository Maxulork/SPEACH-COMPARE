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
            except Exception:
                continue
        raise RuntimeError("Could not load any model.")

    def _load_stt_model(self, cache_dir):
        print("Loading STT model specifically for Slovak language...")
        slovak_models = [
            "infinitejoy/wav2vec2-large-xls-r-300m-slovak"
        ]
        for model_name in slovak_models:
            try:
                self.stt_processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
                self.stt_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir).to(device)
                print(f"Successfully loaded Slovak STT model: {model_name}")
                return
            except Exception:
                continue
        print("Slovak-specific models failed, trying multilingual model...")
        self.stt_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir
        )
        self.stt_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir
        ).to(device)
        print("Successfully loaded fallback multilingual XLSR model")

    def load_audio(self, file_path):
        print(f"Loading audio: {file_path}")
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        audio = self.preprocess_audio(audio)
        return audio

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
        for start, end in intervals:
            audio_out[start:end] = audio[start:end]

        if np.sum(np.abs(audio_out)) < 1e-6:
            return audio
        return audio_out

    def create_spectrogram(self, audio):
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
            spectrogram_db.max() - spectrogram_db.min() + 1e-10
        )
        if spectrogram_normalized.shape[1] != self.target_length:
            resized_spec = np.zeros((self.n_mels, self.target_length))
            time_scale = spectrogram_normalized.shape[1] / self.target_length
            for t in range(self.target_length):
                src_pos = t * time_scale
                src_idx = int(src_pos)
                if src_idx >= spectrogram_normalized.shape[1] - 1:
                    src_idx = spectrogram_normalized.shape[1] - 2
                alpha = src_pos - src_idx
                resized_spec[:, t] = (
                    (1 - alpha) * spectrogram_normalized[:, src_idx] +
                    alpha * spectrogram_normalized[:, src_idx + 1]
                )
            spectrogram_normalized = resized_spec
        return spectrogram_normalized

    def extract_hubert_features(self, audio):
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        input_values = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(device)
        with torch.no_grad():
            outputs = self.model(input_values)
            features = outputs.last_hidden_state.squeeze().cpu().numpy()
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        features = (features - mean) / std
        L = features.shape[0]
        if L > self.target_length:
            indices = np.linspace(0, L - 1, self.target_length, dtype=int)
            features = features[indices]
        elif L < self.target_length:
            resized = np.zeros((self.target_length, features.shape[1]))
            for i in range(features.shape[1]):
                resized[:, i] = np.interp(
                    np.linspace(0, 1, self.target_length),
                    np.linspace(0, 1, L),
                    features[:, i]
                )
            features = resized
        return features

    def compute_context_dtw_similarity(self, f1, f2):
        if f1.shape[1] > 64:
            pca = PCA(n_components=64)
            stacked = np.vstack([f1, f2])
            pca.fit(np.nan_to_num(stacked))
            f1 = pca.transform(np.nan_to_num(f1))
            f2 = pca.transform(np.nan_to_num(f2))
        def ctx_dist(a, b):
            eps = 1e-8
            an = a / (np.linalg.norm(a) + eps)
            bn = b / (np.linalg.norm(b) + eps)
            cos_sim = np.dot(an, bn)
            eucl = np.linalg.norm(a - b)
            return 0.7 * eucl + 0.3 * (1 - cos_sim)
        alignment = dtw_distance(f1, f2, distance_function=ctx_dist)
        length = len(alignment.index1)
        if length == 0:
            return 20.0
        dist = alignment.distance / length
        sim = 100 / (1 + np.exp(0.5 * dist - 2.5))
        return float(np.clip(sim, 0.0, 100.0))

    def speech_to_text(self, audio):
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        max_len = 30 * self.sample_rate
        if len(audio) > max_len:
            chunks = [audio[i:i + max_len] for i in range(0, len(audio), max_len)]
            transcriptions = []
            for chunk in chunks:
                input_vals = self.stt_processor(chunk, sampling_rate=self.sample_rate, return_tensors="pt").input_values
                input_vals = input_vals.to(device)
                with torch.no_grad():
                    logits = self.stt_model(input_vals).logits
                ids = torch.argmax(logits, dim=-1)
                transcriptions.append(self.stt_processor.batch_decode(ids)[0])
            text = " ".join(transcriptions)
        else:
            input_vals = self.stt_processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_values
            input_vals = input_vals.to(device)
            with torch.no_grad():
                logits = self.stt_model(input_vals).logits
            ids = torch.argmax(logits, dim=-1)
            text = self.stt_processor.batch_decode(ids)[0]
        text = text.lower()
        # fix diacritics
        diacritics = {"a´":"á","e´":"é","i´":"í","o´":"ó","u´":"ú",
                      "c´":"č","s´":"š","z´":"ž","l´":"ľ","t´":"ť",
                      "d´":"ď","n´":"ň","r´":"ŕ"}
        for old, new in diacritics.items():
            text = text.replace(old, new)
        return text

    def compute_text_similarity(self, t1, t2):
        n1 = t1.replace("_", "").replace(" ", "")
        n2 = t2.replace("_", "").replace(" ", "")
        if n1 == n2:
            return 100.0
        L = max(len(n1), len(n2))
        if L == 0:
            return 100.0
        dist = levenshtein_distance(n1, n2)
        return max(0.0, min(100.0, (1 - dist / L) * 100))

    def compute_phonetic_similarity(self, audio1, audio2):
        print("Computing phonetic similarity...")
        hub1 = self.extract_hubert_features(audio1)
        hub2 = self.extract_hubert_features(audio2)
        hub_sim = self.compute_context_dtw_similarity(hub1, hub2)

        # MFCC-based similarity
        mfccs1 = librosa.feature.mfcc(y=audio1, sr=self.sample_rate, n_mfcc=13,
                                       n_fft=self.n_fft, hop_length=self.hop_length, htk=True)
        mfccs2 = librosa.feature.mfcc(y=audio2, sr=self.sample_rate, n_mfcc=13,
                                       n_fft=self.n_fft, hop_length=self.hop_length, htk=True)
        d1 = librosa.feature.delta(mfccs1, order=1)
        dd1 = librosa.feature.delta(mfccs1, order=2)
        d2 = librosa.feature.delta(mfccs2, order=1)
        dd2 = librosa.feature.delta(mfccs2, order=2)
        comb1 = np.vstack([mfccs1, d1, dd1])
        comb2 = np.vstack([mfccs2, d2, dd2])
        norm1 = np.linalg.norm(comb1, axis=0) + 1e-8
        norm2 = np.linalg.norm(comb2, axis=0) + 1e-8
        comb1 /= norm1
        comb2 /= norm2
        min_len = min(comb1.shape[1], comb2.shape[1])
        mfcc_sim = 40.0
        if min_len > 10:
            def mfcc_dist(x, y):
                w = np.exp(-0.1 * np.arange(len(x)))
                return np.sqrt(np.sum(w * (x - y) ** 2))
            dist_res = dtw_distance(comb1[:, :min_len].T, comb2[:, :min_len].T,
                                     distance_function=mfcc_dist)
            mfcc_sim = 100 / (1 + np.exp(0.1 * dist_res.distance / min_len))

        # Formant similarity
        form1 = librosa.effects.harmonic(audio1)
        form2 = librosa.effects.harmonic(audio2)
        min_f = min(len(form1), len(form2))
        formant_sim = 50.0
        if min_f > 0:
            corr, _ = pearsonr(form1[:min_f], form2[:min_f])
            formant_sim = 50.0 + 50.0 * max(0, corr)

        # Pitch similarity
        p1, _ = librosa.piptrack(y=audio1, sr=self.sample_rate,
                                  n_fft=self.n_fft, hop_length=self.hop_length)
        p2, _ = librosa.piptrack(y=audio2, sr=self.sample_rate,
                                  n_fft=self.n_fft, hop_length=self.hop_length)
        p1 = np.max(p1, axis=0)
        p2 = np.max(p2, axis=0)
        min_p = min(len(p1), len(p2))
        pitch_sim = 50.0
        if min_p > 10:
            corr_p, _ = pearsonr(p1[:min_p], p2[:min_p])
            pitch_sim = 50.0 + 50.0 * max(0, corr_p)

        # Energy similarity
        e1 = librosa.feature.rms(y=audio1, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        e2 = librosa.feature.rms(y=audio2, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        min_e = min(len(e1), len(e2))
        energy_sim = 50.0
        if min_e > 10:
            corr_e, _ = pearsonr(e1[:min_e], e2[:min_e])
            energy_sim = 50.0 + 50.0 * max(0, corr_e)

        # Text similarity
        text1 = self.speech_to_text(audio1)
        text2 = self.speech_to_text(audio2)
        text_sim = self.compute_text_similarity(text1, text2)

        return {
            "hubert_sim": hub_sim,
            "mfcc_sim": mfcc_sim,
            "formant_sim": formant_sim,
            "pitch_sim": pitch_sim,
            "energy_sim": energy_sim,
            "text_sim": text_sim,
            "transcription1": text1,
            "transcription2": text2
        }, hub1, hub2

    def compute_spectral_similarity(self, audio1, audio2):
        try:
            spec1 = self.create_spectrogram(audio1)
            spec2 = self.create_spectrogram(audio2)
            f1 = self.extract_spectral_features(spec1)
            f2 = self.extract_spectral_features(spec2)
            spectral_sim = (1 - cosine(f1, f2)) * 100
        except Exception:
            spectral_sim = 50.0
            spec1 = np.zeros((self.n_mels, self.target_length))
            spec2 = np.zeros((self.n_mels, self.target_length))
        return max(0.0, min(100.0, spectral_sim)), spec1, spec2

    def visualize_comparison(self, spec1, spec2, f1, f2, sim):
        try:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(spec1, aspect='auto', origin='lower', cmap='viridis')
            plt.title('Spectrogram 1')
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.imshow(spec2, aspect='auto', origin='lower', cmap='viridis')
            plt.title('Spectrogram 2')
            plt.colorbar()
            pca = PCA(n_components=2)
            data_stack = np.vstack([np.nan_to_num(f1), np.nan_to_num(f2)])
            pca.fit(data_stack)
            p1 = pca.transform(np.nan_to_num(f1))
            p2 = pca.transform(np.nan_to_num(f2))
            plt.subplot(2, 2, 3)
            plt.scatter(p1[:, 0], p1[:, 1], s=10, alpha=0.7, c=np.arange(len(p1)), cmap='Blues')
            plt.title('Feature Space Audio 1')
            plt.subplot(2, 2, 4)
            plt.scatter(p2[:, 0], p2[:, 1], s=10, alpha=0.7, c=np.arange(len(p2)), cmap='Oranges')
            plt.title('Feature Space Audio 2')
            plt.suptitle(f'Similarity: {sim:.2f}%')
            plt.tight_layout()
            plt.savefig('speech_comparison.png')
            plt.close()
        except Exception:
            pass

    def color_code_transcription(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = np.zeros((m+1, n+1), dtype=int)
        for i in range(m+1): dp[i, 0] = i
        for j in range(n+1): dp[0, j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1].lower() == text2[j-1].lower():
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        aligned1, aligned2 = [], []
        i, j = m, n
        while i>0 or j>0:
            if i>0 and j>0 and text1[i-1].lower()==text2[j-1].lower():
                aligned1.append(text1[i-1]); aligned2.append(text2[j-1]); i-=1; j-=1
            elif i>0 and (j==0 or dp[i-1,j]+1==dp[i,j]):
                aligned1.append(text1[i-1]); aligned2.append('-'); i-=1
            elif j>0 and (i==0 or dp[i,j-1]+1==dp[i,j]):
                aligned1.append('-'); aligned2.append(text2[j-1]); j-=1
            else:
                aligned1.append(text1[i-1]); aligned2.append(text2[j-1]); i-=1; j-=1
        aligned1.reverse(); aligned2.reverse()
        html = ""
        for c1, c2 in zip(aligned1, aligned2):
            if c1=='-': continue
            if c1.lower()==c2.lower(): html += f"<span style='color:green'>{c1}</span>"
            else: html += f"<span style='color:red'>{c1}</span>"
        return {"html": html, "transcription2": text2}

    def compute_similarity(self, audio1, audio2, visualize=True):
        metrics, hub1, hub2 = self.compute_phonetic_similarity(audio1, audio2)
        sim_text = (0.2*metrics['hubert_sim'] + 0.8*metrics['text_sim'])
        final = 100/(1+np.exp(-0.1*(sim_text-50)))
        colored = self.color_code_transcription(metrics['transcription1'], metrics['transcription2'])
        if visualize:
            spec1 = self.create_spectrogram(audio1)
            spec2 = self.create_spectrogram(audio2)
            self.visualize_comparison(spec1, spec2, hub1, hub2, final)
        return final, colored
