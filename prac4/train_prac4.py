"""Train my tiny CNN and a MobileNet transfer model for Practical 4."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "shared" / "images" / "leaf_dataset"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["blight", "healthy", "rust"]  # ImageFolder sorts folders alphabetically


class TinyCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def loaders(img_size=128, batch=16):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(DATA / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA / "val", eval_tf)
    test_ds = datasets.ImageFolder(DATA / "test", eval_tf)
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True),
        DataLoader(val_ds, batch_size=batch),
        DataLoader(test_ds, batch_size=batch),
        train_ds.classes,
    )


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if train:
            loss.backward()
            opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, params=None):
    opt = torch.optim.Adam(params or model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.perf_counter()
    for _ in range(epochs):
        tr_l, tr_a = run_epoch(model, train_loader, opt)
        va_l, va_a = run_epoch(model, val_loader, None)
        hist["train_loss"].append(tr_l)
        hist["val_loss"].append(va_l)
        hist["train_acc"].append(tr_a)
        hist["val_acc"].append(va_a)
    hist["seconds"] = time.perf_counter() - t0
    return hist


def evaluate(model, loader, n_classes=3):
    model.eval()
    cm = np.zeros((n_classes, n_classes), int)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu()
            for yt, yp in zip(y, pred):
                cm[yt, yp] += 1
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total, cm


def plot_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def grad_cam(model, img_tensor, target_layer, class_idx=None):
    # Grad-CAM from the notes - hooks on a conv layer
    activations, gradients = [], []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    model.zero_grad()
    logits = model(img_tensor)
    if class_idx is None:
        class_idx = int(logits.argmax(1).item())
    logits[0, class_idx].backward()
    h1.remove()
    h2.remove()
    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * act).sum(0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.cpu().numpy(), class_idx, torch.softmax(logits, 1)[0].detach().cpu()


def main():
    train_loader, val_loader, test_loader, classes = loaders()
    comparison = []

    scratch = TinyCNN(len(classes)).to(DEVICE)
    hist_s = train_model(scratch, train_loader, val_loader, epochs=6, lr=1e-3)
    acc_s, cm_s = evaluate(scratch, test_loader, len(classes))
    plot_curves(hist_s, OUT / "t1_curves_scratch.png")
    np.savetxt(OUT / "t1_confusion.csv", cm_s, fmt="%d", delimiter=",")
    comparison.append({"method": "scratch_TinyCNN", "test_acc": acc_s, "seconds": hist_s["seconds"]})

    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, len(classes))
    m = m.to(DEVICE)
    hist_f = train_model(m, train_loader, val_loader, epochs=4, lr=1e-3, params=m.classifier.parameters())
    acc_f, _ = evaluate(m, test_loader, len(classes))
    comparison.append({"method": "transfer_frozen", "test_acc": acc_f, "seconds": hist_f["seconds"]})

    for p in m.features[-1].parameters():
        p.requires_grad = True
    params = list(m.classifier.parameters()) + list(m.features[-1].parameters())
    hist_ft = train_model(m, train_loader, val_loader, epochs=3, lr=1e-4, params=params)
    acc_ft, cm_ft = evaluate(m, test_loader, len(classes))
    plot_curves(hist_ft, OUT / "t2_curves_finetune.png")
    comparison.append({"method": "transfer_finetuned", "test_acc": acc_ft, "seconds": hist_ft["seconds"]})

    with open(OUT / "t2_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    model_path = Path(__file__).resolve().parent / "model.pt"
    torch.save({"state_dict": m.state_dict(), "classes": classes}, model_path)

    eval_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(DATA / "test", eval_tf)
    m.eval()
    target = m.features[-1]
    shown = 0
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        xt = x.unsqueeze(0).to(DEVICE)
        cam, pred, probs = grad_cam(m, xt, target)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (x.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(rgb)
        ax[0].set_title(f"true={classes[y]} pred={classes[pred]}")
        ax[0].axis("off")
        ax[1].imshow(rgb)
        ax[1].imshow(cam, cmap="jet", alpha=0.45)
        ax[1].set_title("Grad-CAM")
        ax[1].axis("off")
        tag = "correct" if pred == y else "miss"
        fig.savefig(OUT / f"t3_gradcam_{tag}_{shown}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        shown += 1
        if shown >= 4:
            break

    print("Practical 4 comparison:", comparison)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
