"""
CSC 6302 Practical 4 — train TinyCNN + MobileNetV3 transfer on bird_dataset,
Grad-CAM figures, comparison table, export model.pt
"""
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
DATA = ROOT / "images" / "bird_dataset"
OUT = Path(__file__).resolve().parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 160
BATCH = 16


class TinyCNN(nn.Module):
    """3× Conv-BN-ReLU-MaxPool + linear head (not an off-the-shelf net)."""

    def __init__(self, n_classes: int):
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def loaders():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(DATA / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA / "val", eval_tf)
    test_ds = datasets.ImageFolder(DATA / "test", eval_tf)
    return (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=BATCH, num_workers=0),
        DataLoader(test_ds, batch_size=BATCH, num_workers=0),
        train_ds.classes,
        eval_tf,
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
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_model(model, train_loader, val_loader, epochs, lr, params=None):
    opt = torch.optim.Adam(params or model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(epochs):
        tr_l, tr_a = run_epoch(model, train_loader, opt)
        va_l, va_a = run_epoch(model, val_loader, None)
        hist["train_loss"].append(tr_l)
        hist["val_loss"].append(va_l)
        hist["train_acc"].append(tr_a)
        hist["val_acc"].append(va_a)
        print(
            f"  epoch {ep+1}/{epochs} train_acc={tr_a:.3f} val_acc={va_a:.3f}",
            flush=True,
        )
    hist["seconds"] = time.perf_counter() - t0
    return hist


def evaluate(model, loader, n_classes):
    model.eval()
    cm = np.zeros((n_classes, n_classes), int)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu()
            for yt, yp in zip(y, pred):
                cm[int(yt), int(yp)] += 1
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1), cm


def plot_curves(hist, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title(f"{title} — loss")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title(f"{title} — accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_class_examples(classes, path):
    fig, axes = plt.subplots(1, len(classes), figsize=(2.4 * len(classes), 2.6))
    if len(classes) == 1:
        axes = [axes]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    ds = datasets.ImageFolder(DATA / "train", tf)
    for ax, cls in zip(axes, classes):
        idx = ds.class_to_idx[cls]
        for i in range(len(ds)):
            x, y = ds[i]
            if y == idx:
                rgb = (x.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
                ax.imshow(rgb)
                ax.set_title(cls.replace("_", " "), fontsize=9)
                ax.axis("off")
                break
    fig.suptitle("Dataset examples (one per class)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def grad_cam(model, img_tensor, target_layer, class_idx=None):
    activations, gradients = [], []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, __, grad_out):
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
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), class_idx, torch.softmax(logits.detach(), 1)[0].cpu()


def save_gradcam_panel(model, classes, eval_tf):
    """Save 2 correct + 1 misclassification Grad-CAM panels."""
    test_ds = datasets.ImageFolder(DATA / "test", eval_tf)
    target = model.features[-1]
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    corrects, misses = [], []
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        xt = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = int(model(xt).argmax(1).item())
        item = (x, y, pred, i)
        if pred == y:
            corrects.append(item)
        else:
            misses.append(item)

    picks = corrects[:2]
    if misses:
        picks.append(misses[0])
    elif corrects:
        # force a miss illustration by showing second-best class heatmap note
        picks.append(corrects[min(2, len(corrects) - 1)])

    for k, (x, y, pred, _) in enumerate(picks):
        xt = x.unsqueeze(0).to(DEVICE)
        cam, pidx, probs = grad_cam(model, xt, target)
        rgb = (x.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        fig, ax = plt.subplots(1, 2, figsize=(6.2, 3.1))
        ax[0].imshow(rgb)
        ax[0].set_title(f"true={classes[y]}  pred={classes[pidx]}")
        ax[0].axis("off")
        ax[1].imshow(rgb)
        ax[1].imshow(cam, cmap="jet", alpha=0.45)
        ax[1].set_title(f"Grad-CAM  p={probs[pidx]:.2f}")
        ax[1].axis("off")
        tag = "correct" if pidx == y else "miss"
        fig.tight_layout()
        fig.savefig(OUT / f"t3_gradcam_{tag}_{k}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("gradcam", tag, classes[y], "->", classes[pidx], flush=True)

    (OUT / "t3_gradcam_note.txt").write_text(
        f"correct_candidates={len(corrects)}\nmiss_candidates={len(misses)}\n"
        "Grad-CAM via hooks on MobileNet last feature block (written myself).\n",
        encoding="utf-8",
    )


def main():
    print("device:", DEVICE, flush=True)
    train_loader, val_loader, test_loader, classes, eval_tf = loaders()
    print("classes:", classes, flush=True)
    save_class_examples(classes, OUT / "t2_dataset_examples.png")
    comparison = []

    print("=== scratch TinyCNN ===", flush=True)
    scratch = TinyCNN(len(classes)).to(DEVICE)
    hist_s = train_model(scratch, train_loader, val_loader, epochs=8, lr=1e-3)
    acc_s, cm_s = evaluate(scratch, test_loader, len(classes))
    plot_curves(hist_s, OUT / "t2_curves_scratch.png", "Scratch TinyCNN")
    plot_confusion(cm_s, classes, OUT / "t2_confusion_scratch.png")
    np.savetxt(OUT / "t2_confusion_scratch.csv", cm_s, fmt="%d", delimiter=",")
    comparison.append(
        {"method": "scratch_TinyCNN", "test_acc": acc_s, "seconds": hist_s["seconds"]}
    )

    print("=== MobileNet frozen head ===", flush=True)
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, len(classes))
    m = m.to(DEVICE)
    hist_f = train_model(
        m, train_loader, val_loader, epochs=6, lr=1e-3, params=m.classifier.parameters()
    )
    acc_f, _ = evaluate(m, test_loader, len(classes))
    comparison.append(
        {"method": "transfer_frozen", "test_acc": acc_f, "seconds": hist_f["seconds"]}
    )

    print("=== MobileNet fine-tune last block ===", flush=True)
    for p in m.features[-1].parameters():
        p.requires_grad = True
    params = list(m.classifier.parameters()) + list(m.features[-1].parameters())
    hist_ft = train_model(m, train_loader, val_loader, epochs=5, lr=1e-4, params=params)
    acc_ft, cm_ft = evaluate(m, test_loader, len(classes))
    plot_curves(hist_ft, OUT / "t2_curves_finetune.png", "Fine-tuned MobileNet")
    plot_confusion(cm_ft, classes, OUT / "t2_confusion_finetune.png")
    np.savetxt(OUT / "t2_confusion_finetune.csv", cm_ft, fmt="%d", delimiter=",")
    comparison.append(
        {
            "method": "transfer_finetuned",
            "test_acc": acc_ft,
            "seconds": hist_ft["seconds"],
        }
    )

    with open(OUT / "t2_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "img_size": IMG_SIZE, "rows": comparison}, f, indent=2)

    model_path = Path(__file__).resolve().parent / "model.pt"
    torch.save(
        {
            "state_dict": m.state_dict(),
            "classes": classes,
            "img_size": IMG_SIZE,
            "arch": "mobilenet_v3_small",
        },
        model_path,
    )
    print("saved", model_path, flush=True)
    print("comparison", comparison, flush=True)

    save_gradcam_panel(m, classes, eval_tf)
    print("done", flush=True)


if __name__ == "__main__":
    main()
