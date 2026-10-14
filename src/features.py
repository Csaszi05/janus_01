import librosa
import numpy as np


def load_audio(path, sr=16000, max_sec=4.0):
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(sr * max_sec)
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def mel_db(y, sr=16000, n_fft=1024, hop=256, n_mels=80):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)
