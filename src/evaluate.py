import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MelDataset, load_protocol
from model import JanusCNN
from train import compute_eer

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    out = []
    for i, (x, _) in enumerate(loader, 1):
        out.append(model(x.to(device)).cpu())
        if i % 200 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="eval", choices=["dev", "eval"])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out", default=str(ROOT / "out"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = JanusCNN().to(device)
    model.load_state_dict(ck["model"])
    print(f"checkpoint epoch={ck['epoch']} device={device} split={args.split}")

    df = load_protocol(args.data_root, args.split)
    cache = Path(args.cache_dir) / args.split if args.cache_dir else None
    ds = MelDataset(df, cfg["audio"], cache)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    df["score"] = score(model, dl, device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df[["file", "speaker", "system", "label", "score"]].to_csv(out / f"{args.split}_scores.csv", index=False)

    bona = df[df.y == 0]
    spoof = df[df.y == 1]
    total = compute_eer(df.score.to_numpy(), df.y.to_numpy())
    print(f"\n{args.split}: {len(df)} fajl, bonafide {len(bona)}, spoof {len(spoof)}")
    print(f"osszesitett EER: {total * 100:.2f}%\n")
    print(f"{'tamadas':<8}{'n':>7}{'EER':>9}")
    rows = []
    for sysid, grp in sorted(spoof.groupby("system")):
        s = np.concatenate([bona.score.to_numpy(), grp.score.to_numpy()])
        y = np.concatenate([np.zeros(len(bona)), np.ones(len(grp))])
        eer = compute_eer(s, y)
        rows.append({"system": sysid, "n": len(grp), "eer": eer})
        print(f"{sysid:<8}{len(grp):>7}{eer * 100:>8.2f}%")
    pd.DataFrame(rows).to_csv(out / f"{args.split}_per_attack.csv", index=False)
    print(f"\nmentve: {out / f'{args.split}_scores.csv'}, {out / f'{args.split}_per_attack.csv'}")


if __name__ == "__main__":
    main()
