import librosa
import numpy as np
from scipy.fftpack import dct


def load_audio(path, sr=16000, max_sec=4.0, trim_db=30):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    n = int(sr * max_sec)
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def mel_db(y, sr=16000, n_fft=1024, hop=256, n_mels=80):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


def linear_filterbank(sr, n_fft, n_filters):
    """n_filters darab haromszog-szuro, egyenletesen 0 es sr/2 kozott. Alak: (n_filters, n_fft//2+1)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)), dtype=np.float32)
    for i in range(n_filters):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


def lfcc(y, sr=16000, n_fft=512, win=320, hop=160, n_filters=20, n_lfcc=20, deltas=True):
    """LFCC: linearis szurobank -> log -> DCT. deltas=True eseten delta es delta-delta is,
    igy a kimenet (3*n_lfcc, T). Az ASVspoof 2019 hivatalos LFCC-baseline receptje."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win, hop_length=hop)) ** 2
    E = linear_filterbank(sr, n_fft, n_filters) @ S
    C = dct(np.log(E + 1e-10), type=2, axis=0, norm="ortho")[:n_lfcc]
    if deltas:
        C = np.concatenate([C, librosa.feature.delta(C), librosa.feature.delta(C, order=2)])
    return C.astype(np.float32)


def extract(y, cfg):
    """audio config alapjan valasztja a feature-t: cfg['feature'] = 'mel' (alapertelmezes) vagy 'lfcc'."""
    kind = cfg.get("feature", "mel")
    if kind == "mel":
        return mel_db(y, sr=cfg["sr"], n_fft=cfg["n_fft"], hop=cfg["hop"], n_mels=cfg["n_mels"])
    if kind == "lfcc":
        return lfcc(y, sr=cfg["sr"], **cfg["lfcc"])
    raise ValueError(f"ismeretlen feature: {kind}")
