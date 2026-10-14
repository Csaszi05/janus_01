from pathlib import Path

import pandas as pd

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
