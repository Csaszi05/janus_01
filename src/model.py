import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class JanusCNN(nn.Module):
    """Bemenet: (B, 1, F, T) feature-kep (mel dB: F=80, LFCC: F=60). Az AdaptiveAvgPool miatt
    F es T tetszoleges. Kimenet: (B,) logit, pozitiv = spoof."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)
