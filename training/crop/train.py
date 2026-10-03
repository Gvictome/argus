"""
Fine-tune the ARGUS crop classifier on a folder of crops.

    python training/crop/train.py --data D:/argus-data/crops/make --out models/crop_make
    python training/crop/train.py --data D:/argus-data/crops/carrier --out models/crop_carrier

`--data` holds train/ val/ test/ subfolders, one folder per class
(torchvision ImageFolder layout). The backbone is MobileNetV3-Large,
ImageNet weights: small enough for a ~3 ms crop on the Hailo, and in the
Hailo model zoo, so the Dataflow Compiler knows every op in it.

Writes <out>/model.pt (state dict + class list), <out>/metrics.json, and
<out>/model.onnx for scripts/export_hailo.py's workstation step.
CPU-only works: about a minute or two an epoch on Stanford Cars with 12
threads.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import make_accuracies  # noqa: E402

IMG = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def loaders(root: Path, batch: int, workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.ImageFolder(root / "train", train_tf)
    val = datasets.ImageFolder(root / "val", eval_tf)
    test = datasets.ImageFolder(root / "test", eval_tf)
    for ds in (val, test):
        if ds.classes != train.classes:
            raise SystemExit(f"{ds.root}: class folders differ from train/")

    # Classes are badly imbalanced (Chevrolet 22 models, Tesla 1): sample
    # each class equally often rather than letting the big ones dominate.
    counts = torch.bincount(torch.tensor(train.targets), minlength=len(train.classes))
    weights = (1.0 / counts.float())[train.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    kw = {"batch_size": batch, "num_workers": workers, "persistent_workers": workers > 0}
    return (DataLoader(train, sampler=sampler, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw),
            train.classes)


def build(n_classes: int) -> nn.Module:
    net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_classes)
    return net


@torch.no_grad()
def predict(net: nn.Module, loader: DataLoader):
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(net(x).argmax(1))
        ys.append(y)
    return torch.cat(ys).tolist(), torch.cat(ps).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--task", choices=("make", "carrier", "size"), default="make")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    train_dl, val_dl, test_dl, classes = loaders(args.data, args.batch, args.workers)
    net = build(len(classes))

    head = list(net.classifier.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in net.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr * 0.3},
                             {"params": head, "lr": args.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * 0.3, args.lr],
        total_steps=args.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"  {len(classes)} classes, {len(train_dl.dataset)} train / "
          f"{len(val_dl.dataset)} val / {len(test_dl.dataset)} test, {args.threads} threads")
    args.out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, seen, running = time.perf_counter(), 0, 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        yv, pv = predict(net, val_dl)
        val_acc = sum(a == b for a, b in zip(yv, pv)) / len(yv)
        history.append({"epoch": epoch, "loss": round(running / seen, 4),
                        "val_acc": round(val_acc, 4),
                        "secs": round(time.perf_counter() - t0, 1)})
        print(f"  epoch {epoch:2d}  loss {running / seen:.3f}  val {val_acc:.3f}  "
              f"({history[-1]['secs']:.0f}s)", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "task": args.task, "img": IMG}, args.out / "model.pt")

    ckpt = torch.load(args.out / "model.pt", map_location="cpu")
    net.load_state_dict(ckpt["state_dict"])
    yt, pt = predict(net, test_dl)
    test_acc = sum(a == b for a, b in zip(yt, pt)) / len(yt)
    metrics = {"task": args.task, "classes": classes, "best_val_acc": round(best, 4),
               "test_acc": round(test_acc, 4), "test_n": len(yt), "history": history}
    if args.task == "make":
        _, slot_acc = make_accuracies(yt, pt, classes)
        metrics["test_acc_argus_slots"] = round(slot_acc, 4)
    confusion = [[0] * len(classes) for _ in classes]
    for a, b in zip(yt, pt):
        confusion[a][b] += 1
    metrics["confusion"] = confusion
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    net.eval()
    torch.onnx.export(net, torch.randn(1, 3, IMG, IMG), args.out / "model.onnx",
                      input_names=["image"], output_names=["logits"], opset_version=13,
                      dynamo=False)
    line = f"  test acc {test_acc:.3f}"
    if "test_acc_argus_slots" in metrics:
        line += f"  (ARGUS 7-slot {metrics['test_acc_argus_slots']:.3f})"
    print(line + f"  -> {args.out}")


if __name__ == "__main__":
    main()
