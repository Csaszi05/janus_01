import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from dataset import MelDataset, load_protocol
from model import JanusCNN

ROOT = Path(__file__).resolve().parent.parent


def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(model(x.to(device)).cpu())
        labels.append(y)
    return compute_eer(torch.cat(scores).numpy(), torch.cat(labels).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=str(ROOT / "out"))
    p.add_argument("--cache_dir", default=None, help="mel cache; alapertelmezes: <out>/cache")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=100, help="koztes loss kiirasa ennyi batch-enkent")
    p.add_argument("--augment", action="store_true", help="SpecAugment a train halmazon (config.yaml: augment)")
    p.add_argument("--sched", default="none", choices=["none", "cosine"], help="lr-utemezo")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else out / "cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = load_protocol(args.data_root, "train")
    dev_df = load_protocol(args.data_root, "dev")
    train_ds = MelDataset(train_df, cfg["audio"], cache / "train", augment=cfg["augment"] if args.augment else None)
    dev_ds = MelDataset(dev_df, cfg["audio"], cache / "dev")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    n_pos = train_df["y"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    model = JanusCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.sched == "cosine" else None

    print(f"device={device} train={len(train_ds)} dev={len(dev_ds)} pos_weight={pos_weight.item():.3f} "
          f"augment={args.augment} sched={args.sched}")
    best_eer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            if args.log_every and i % args.log_every == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_dl)} loss {total / n:.4f}", flush=True)
        train_loss = total / n
        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
        dev_eer = evaluate(model, dev_dl, device)
        flag = ""
        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": cfg},
                out / "janus_best.pt",
            )
            flag = " *"
        print(f"epoch {epoch:3d}  lr {lr:.2e}  train_loss {train_loss:.4f}  dev_eer {dev_eer * 100:.2f}%{flag}")
    print(f"best dev_eer {best_eer * 100:.2f}%")


if __name__ == "__main__":
    main()
