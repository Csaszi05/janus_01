from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import load_audio, mel_db

PROTOCOL = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def load_protocol(root, split):
    root = Path(root)
    proto = root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL[split]
    df = pd.read_csv(proto, sep=" ", header=None, names=["speaker", "file", "-", "system", "label"])
    flac = root / f"ASVspoof2019_LA_{split}" / "flac"
    df["path"] = df["file"].map(lambda f: str(flac / f"{f}.flac"))
    df["y"] = (df["label"] == "spoof").astype(int)
    return df


class MelDataset(Dataset):
    """load_protocol DataFrame -> (mel (1, n_mels, T) float32, y float32).

    cache_dir megadasakor a mel-spektrogramot fajlonkent .npy-ba menti,
    igy a masodik epochtol nem szamol ujra.
    """

    def __init__(self, df, audio_cfg, cache_dir=None):
        self.paths = df["path"].tolist()
        self.files = df["file"].tolist()
        self.y = df["y"].to_numpy(dtype=np.float32)
        self.cfg = audio_cfg
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _mel(self, i):
        cache = self.cache_dir / f"{self.files[i]}.npy" if self.cache_dir else None
        if cache and cache.exists():
            return np.load(cache)
        c = self.cfg
        y = load_audio(self.paths[i], sr=c["sr"], max_sec=c["max_sec"], trim_db=c["trim_db"])
        m = mel_db(y, sr=c["sr"], n_fft=c["n_fft"], hop=c["hop"], n_mels=c["n_mels"])
        if cache:
            np.save(cache, m)
        return m

    def __getitem__(self, i):
        m = torch.from_numpy(self._mel(i)).unsqueeze(0)
        return m, torch.tensor(self.y[i])
