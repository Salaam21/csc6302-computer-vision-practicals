"""
Task 1 — TinyCNN on CIFAR-10 (local CPU-friendly; Colab notebook uses more epochs).
Saves curves + confusion matrix under report/figures/
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Full CIFAR on CPU is slow — use 10 epochs; Colab notebook targets 12–15 on GPU.
EPOCHS = 10
BATCH = 128


class TinyCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    print("device", DEVICE, flush=True)
    tf_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    tf_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    root = Path(__file__).resolve().parents[1] / "images" / "cifar10"
    root.mkdir(parents=True, exist_ok=True)
    train_full = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tf_test)

    # 90/10 train/val from training set
    n = len(train_full)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    n_val = n // 10
    val_ds = Subset(train_full, idx[:n_val])
    train_ds = Subset(train_full, idx[n_val:])
    # CIFAR Subset keeps train transform — rebuild val with test transform
    val_base = datasets.CIFAR10(root=str(root), train=True, download=False, transform=tf_test)
    val_ds = Subset(val_base, idx[:n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH, num_workers=0)

    model = TinyCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.perf_counter()

    for ep in range(EPOCHS):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        tr_l, tr_a = loss_sum / total, correct / total

        model.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        va_l, va_a = loss_sum / total, correct / total
        hist["train_loss"].append(tr_l)
        hist["val_loss"].append(va_l)
        hist["train_acc"].append(tr_a)
        hist["val_acc"].append(va_a)
        print(f"epoch {ep+1}/{EPOCHS} train={tr_a:.3f} val={va_a:.3f}", flush=True)

    seconds = time.perf_counter() - t0
    classes = test_set.classes
    cm = np.zeros((10, 10), int)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu()
            for yt, yp in zip(y, pred):
                cm[int(yt), int(yp)] += 1
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("CIFAR-10 loss")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("CIFAR-10 accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "t1_curves_scratch.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "t1_confusion.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    np.savetxt(OUT / "t1_confusion.csv", cm, fmt="%d", delimiter=",")

    # simple pairwise confusion note
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    top = pairs[:5]
    overfit_ep = None
    for i in range(1, len(hist["val_loss"])):
        if hist["train_loss"][i] < hist["val_loss"][i] and hist["train_acc"][i] - hist["val_acc"][i] > 0.05:
            overfit_ep = i + 1
            break

    (OUT / "t1_cifar_summary.txt").write_text(
        f"test_acc={test_acc:.4f}\nseconds={seconds:.1f}\nepochs={EPOCHS}\n"
        f"overfit_hint_epoch={overfit_ep}\n"
        f"top_confusions={top}\n"
        "note=Cats/dogs and automobile/truck pairs often confuse because of shared shapes/colours.\n",
        encoding="utf-8",
    )
    print("test_acc", test_acc, "seconds", seconds, flush=True)


if __name__ == "__main__":
    main()
