from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import extract, load_audio

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
    """load_protocol DataFrame -> (feature (1, F, T) float32, y float32).

    F = n_mels (mel) vagy 3*n_lfcc (lfcc), az audio config `feature` kulcsa szerint.
    cache_dir megadasakor a feature-t fajlonkent .npy-ba menti, a feature neve szerinti
    almappaba (cache/mel/..., cache/lfcc/...), igy a masodik epochtol nem szamol ujra,
    es a ket feature cache-e nem keveredik.
    """

    def __init__(self, df, audio_cfg, cache_dir=None, augment=None):
        self.paths = df["path"].tolist()
        self.files = df["file"].tolist()
        self.y = df["y"].to_numpy(dtype=np.float32)
        self.cfg = audio_cfg
        self.feature = audio_cfg.get("feature", "mel")
        self.augment = augment
        self.cache_dir = Path(cache_dir) / self.feature if cache_dir else None
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
        m = extract(y, c)
        if cache:
            np.save(cache, m)
        return m

    def _augment(self, m):
        """SpecAugment-szeru torzitas a dB-s mel-spektrogramon. torch veletlent hasznal,
        hogy a DataLoader worker-ek kulonbozo sorozatot kapjanak."""
        a = self.augment
        m = m.copy()
        if self.feature == "mel":
            # zajpadlo: teljesitmenyben osszeadjuk egy veletlen szintu egyenletes zajjal.
            # csak dB-s spektrogramon ertelmes; LFCC (cepstrum) eseten kimarad, ott csak maszkolas van.
            noise_db = a["noise_db_min"] + torch.rand(()).item() * (a["noise_db_max"] - a["noise_db_min"])
            m = 10 * np.log10(10 ** (m / 10) + 10 ** (noise_db / 10)).astype(np.float32)
        fill = m.mean()
        n_mels, T = m.shape
        for _ in range(a["n_freq_masks"]):
            w = int(torch.randint(0, a["freq_mask_max"] + 1, ()))
            f0 = int(torch.randint(0, n_mels - w + 1, ()))
            m[f0 : f0 + w, :] = fill
        for _ in range(a["n_time_masks"]):
            w = int(torch.randint(0, a["time_mask_max"] + 1, ()))
            t0 = int(torch.randint(0, T - w + 1, ()))
            m[:, t0 : t0 + w] = fill
        return m

    def __getitem__(self, i):
        m = self._mel(i)
        if self.augment:
            m = self._augment(m)
        return torch.from_numpy(m).unsqueeze(0), torch.tensor(self.y[i])
