"""
CAME Assignment 02 - Colorectal Histopathology CNN Classifier (PyTorch).

Dataset : 1aurent/NCT-CRC-HE (split NCT_CRC_HE_100K), 9 tissue classes.
Model   : custom CNN trained from random initialisation (no pretrained weights).

This file is the single CMS submission. It is also imported by
``train_colab.py`` / ``train_colab.ipynb``, which run a Ray Tune search
(tracked in MLflow) on a Colab T4 GPU, train the final model with the best
configuration and then re-write this file with:

  * ``BEST_CONFIG``        - architecture picked by Ray Tune (used by build_model)
  * ``TRAINED_MODEL_B64``  - the Colab-trained weights (used by load_trained_model)

Importing this module never trains and never downloads data; all execution
happens in ``main()`` under ``if __name__ == "__main__":``.
"""

import base64
import copy
import io
import math
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 9
IMAGE_SIZE = 64
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
SPLIT_SEED = 42
VAL_FRACTION = 0.15
PARAM_MIN, PARAM_MAX = 25_000, 500_000

HF_DATASET_ID = "1aurent/NCT-CRC-HE"
HF_SPLIT = "NCT_CRC_HE_100K"
DEFAULT_PER_CLASS = 1000        # 9 x 1000 = 9,000 patches (balanced)
TAKE_PER_ROWGROUP = 50          # spread the sample over many parquet row groups

# ---------------------------------------------------------------------------
# Architecture selected by Ray Tune (re-written automatically by train_colab.py)
# ---------------------------------------------------------------------------
# >>> BEST_CONFIG >>>
BEST_CONFIG = {
    "width": 32,
    "n_blocks": 3,
    "convs_per_block": 2,
    "head_hidden": 128,
    "dropout": 0.3,
}
TUNED_TRAINING_CONFIG = {}  # informational: hyper-parameters used on Colab
# <<< BEST_CONFIG <<<

# TRAINED_MODEL_B64 (the Colab-trained weights) is defined near the end of
# this file because it is a long string.


# ===========================================================================
# Task 1 - Data preprocessing & loaders
# ===========================================================================
def official_classes() -> list[str]:
    return ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


class ArrayImageDataset(Dataset):
    """uint8 RGB patches (N, H, W, 3) + int labels -> (tensor_3x64x64, int)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.targets = self.labels.tolist()
        self.classes = official_classes()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


# ---- Hugging Face loading (Colab only; never used when CAME_DATA_DIR is set) ----
_RESIZE_64 = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def _decode_to_64(image_field) -> np.ndarray:
    """Decode one HF image cell (dict with bytes, or PIL image) to uint8 64x64x3."""
    if isinstance(image_field, dict):
        img = Image.open(io.BytesIO(image_field["bytes"]))
    else:
        img = image_field
    img = _RESIZE_64(img.convert("RGB"))       # same Resize as make_transforms()
    return np.asarray(img, dtype=np.uint8)


def _hf_parquet_files(split: str):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{HF_DATASET_ID}/data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split {split}")
    return fs, files


def _rowgroup_labels(fs, path: str):
    """[(row_group, min_label, max_label, num_rows), ...] from the parquet footer."""
    import pyarrow.parquet as pq
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        col = next(j for j in range(md.num_columns)
                   if md.schema.column(j).path == "label")
        out = []
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(col).statistics
            if st is None or not st.has_min_max:
                lab = pf.read_row_group(i, columns=["label"]).column("label").to_numpy()
                out.append((i, int(lab.min()), int(lab.max()), len(lab)))
            else:
                out.append((i, int(st.min), int(st.max), md.row_group(i).num_rows))
    return out


def _read_rowgroup(fs, path, rg, cls=None, take=None, seed=0):
    import pyarrow.parquet as pq
    with fs.open(path, "rb") as fh:
        table = pq.ParquetFile(fh).read_row_group(rg, columns=["image", "label"])
    labels = table.column("label").to_numpy().astype(np.int64)
    idx = np.arange(len(labels)) if cls is None else np.flatnonzero(labels == cls)
    if take is not None and len(idx) > take:
        idx = np.sort(np.random.default_rng(seed).choice(idx, take, replace=False))
    if len(idx) == 0:
        return np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8), labels[:0]
    cells = table.column("image").take(idx).to_pylist()
    return np.stack([_decode_to_64(c) for c in cells]), labels[idx]


def _run_rowgroup_jobs(fs, jobs, workers: int = 16):
    """jobs: [(path, rg, cls, take, seed)] -> concatenated (images, labels) in job order."""
    def work(job):
        return _read_rowgroup(fs, *job)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(work, jobs))
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return x, y


def _load_balanced_via_parquet(per_class: int, take: int):
    """Read only selected row groups, spread evenly over each class's block."""
    fs, files = _hf_parquet_files(HF_SPLIT)
    with ThreadPoolExecutor(max_workers=8) as ex:
        metas = list(ex.map(lambda p: _rowgroup_labels(fs, p), files))
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        cands = [(fi, rg) for fi, m in enumerate(metas)
                 for (rg, lo, hi, _) in m if lo == hi == c]
        if not cands:   # no pure row group -> accept mixed ones
            cands = [(fi, rg) for fi, m in enumerate(metas)
                     for (rg, lo, hi, _) in m if lo <= c <= hi]
        n_rg = min(len(cands), math.ceil(per_class / take))
        pick = np.unique(np.linspace(0, len(cands) - 1, n_rg).round().astype(int))
        k = math.ceil(per_class / len(pick))
        jobs = [(files[cands[p][0]], cands[p][1], c, k,
                 SPLIT_SEED * 100_000 + cands[p][0] * 1_000 + cands[p][1])
                for p in pick]
        x, y = _run_rowgroup_jobs(fs, jobs)
        xs.append(x[:per_class])
        ys.append(y[:per_class])
        print(f"  class {c} ({official_classes()[c]}): {len(ys[-1])} patches "
              f"from {len(jobs)} row groups", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


# One shard index per class whose rows are all that class (fallback only).
_PURE_SHARD = {0: 8, 1: 22, 2: 1, 3: 12, 4: 25, 5: 28, 6: 15, 7: 5, 8: 18}


def _load_balanced_via_datasets(per_class: int):
    """Fallback: stream one pure-class shard per class with HF `datasets`."""
    import datasets as hf_datasets
    xs, ys = [], []
    for c, shard in _PURE_SHARD.items():
        ds = hf_datasets.load_dataset(
            HF_DATASET_ID,
            data_files={HF_SPLIT: f"data/{HF_SPLIT}-{shard:05d}-of-*.parquet"},
            split=HF_SPLIT, streaming=True)
        n = 0
        for ex in ds:
            if int(ex["label"]) != c:
                continue
            xs.append(_decode_to_64(ex["image"]))
            ys.append(c)
            n += 1
            if n >= per_class:
                break
        print(f"  class {c} ({official_classes()[c]}): {n} patches (streamed)", flush=True)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def load_balanced_subset(per_class: int = None, cache_dir: str = None):
    """Deterministic balanced subset of NCT_CRC_HE_100K as uint8 64x64 arrays (cached)."""
    per_class = int(per_class or os.environ.get("CAME_PER_CLASS", DEFAULT_PER_CLASS))
    cache = Path(cache_dir or os.environ.get("CAME_CACHE_DIR", "came_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"{HF_SPLIT}_pc{per_class}_t{TAKE_PER_ROWGROUP}_{IMAGE_SIZE}px.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        return d["x"], d["y"]
    print(f"Loading {per_class}/class from {HF_DATASET_ID} [{HF_SPLIT}] ...", flush=True)
    t0 = time.time()
    try:
        x, y = _load_balanced_via_parquet(per_class, TAKE_PER_ROWGROUP)
    except Exception as exc:  # network / API differences -> HF datasets streaming
        print(f"  row-group loader failed ({exc!r}); falling back to datasets streaming")
        x, y = _load_balanced_via_datasets(per_class)
    np.savez(cache_file, x=x, y=y)
    print(f"  done: {len(y)} patches in {time.time() - t0:.0f}s -> {cache_file}", flush=True)
    return x, y


def stratified_split(labels: np.ndarray, val_fraction: float = VAL_FRACTION,
                     seed: int = SPLIT_SEED):
    """85/15 per-class split with a fixed seed -> (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def get_split_arrays(per_class: int = None, cache_dir: str = None):
    """(x_train, y_train, x_val, y_val) uint8 arrays shared by Colab training paths."""
    x, y = load_balanced_subset(per_class, cache_dir)
    tr, va = stratified_split(y)
    return x[tr], y[tr], x[va], y[va]


def build_datasets():
    data_dir = os.environ.get("CAME_DATA_DIR")
    if data_dir:  # CMS: local ImageFolder only, no HF / internet
        train_dataset = ImageFolder(os.path.join(data_dir, "train"),
                                    transform=make_transforms(True))
        val_dataset = ImageFolder(os.path.join(data_dir, "validation"),
                                  transform=make_transforms(False))
        return train_dataset, val_dataset
    x_tr, y_tr, x_va, y_va = get_split_arrays()
    train_dataset = ArrayImageDataset(x_tr, y_tr, transform=make_transforms(True))
    val_dataset = ArrayImageDataset(x_va, y_va, transform=make_transforms(False))
    return train_dataset, val_dataset


def make_loaders(train_dataset, val_dataset, batch_size: int):
    num_workers = int(os.environ.get("CAME_WORKERS", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


# ===========================================================================
# Task 2 - Custom CNN (from scratch)
# ===========================================================================
class HistoCNN(nn.Module):
    """Funnel CNN: [Conv3x3-BN-ReLU] x k -> MaxPool2d(2) per block, channels
    width -> 2*width -> 4*width ..., then GAP -> Dropout -> (Linear-ReLU-Dropout)
    -> Linear(num_classes). forward() returns raw logits (no softmax)."""

    def __init__(self, num_classes: int = 9, width: int = 32, n_blocks: int = 3,
                 convs_per_block: int = 2, head_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        layers, in_ch = [], 3
        for b in range(n_blocks):
            out_ch = width * (2 ** b)
            n_convs = 1 if b == 0 else convs_per_block   # keep 64x64 stage cheap
            for _ in range(n_convs):
                layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(out_ch),
                           nn.ReLU(inplace=True)]
                in_ch = out_ch
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)

        head = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout)]
        if head_hidden:
            head += [nn.Linear(in_ch, head_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_ch = head_hidden
        head.append(nn.Linear(in_ch, num_classes))
        self.classifier = nn.Sequential(*head)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model_from_config(num_classes: int, cfg: dict) -> torch.nn.Module:
    return HistoCNN(num_classes=num_classes,
                    width=int(cfg["width"]),
                    n_blocks=int(cfg["n_blocks"]),
                    convs_per_block=int(cfg["convs_per_block"]),
                    head_hidden=int(cfg["head_hidden"]),
                    dropout=float(cfg["dropout"]))


def build_model(num_classes: int = 9) -> torch.nn.Module:
    return build_model_from_config(num_classes, BEST_CONFIG)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Task 3 - Training loop, validation & best checkpoint restoration
# ===========================================================================
def train_model(train_loader, val_loader, epochs: int, device: torch.device):
    device = torch.device(device)
    torch.manual_seed(42)
    model = build_model(9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc, best_val_loss = -1.0, float("inf")
    history = []

    for epoch in range(epochs):
        # ---- training phase ----
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += n
        train_loss = loss_sum / max(seen, 1)
        train_acc = correct / max(seen, 1)

        # ---- validation phase ----
        model.eval()
        loss_sum, correct, seen = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                n = labels.size(0)
                loss_sum += loss.item() * n
                correct += (logits.argmax(dim=1) == labels).sum().item()
                seen += n
        val_loss = loss_sum / max(seen, 1)
        val_acc = correct / max(seen, 1)

        # ---- best checkpoint tracking ----
        if val_acc > best_val_acc + 1e-12 or (
                abs(val_acc - best_val_acc) <= 1e-12 and val_loss < best_val_loss):
            best_val_acc, best_val_loss = val_acc, val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })
        print(f"epoch {epoch + 1:02d}/{epochs} | train_loss {train_loss:.4f} "
              f"acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, history


# ===========================================================================
# Task 4 - Inference & metrics
# ===========================================================================
def predict(model: torch.nn.Module, loader, device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(images.to(device))
            preds.append(logits.argmax(dim=1).cpu())
    if not preds:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(preds).long()


def classification_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_true = torch.as_tensor(y_true).detach().cpu().long().view(-1)
    y_pred = torch.as_tensor(y_pred).detach().cpu().long().view(-1)
    if y_true.numel() != y_pred.numel():
        raise ValueError("y_true and y_pred must have the same length")
    n = NUM_CLASSES
    confusion = torch.zeros((n, n), dtype=torch.long)
    if y_true.numel() > 0:
        confusion = torch.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n).long()
    accuracy = float((y_true == y_pred).float().mean().item()) if y_true.numel() else 0.0
    support = confusion.sum(dim=1)
    recalls = confusion.diag().double() / support.clamp(min=1).double()
    present = support > 0
    macro_recall = float(recalls[present].mean().item()) if present.any() else 0.0
    return {
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "confusion": confusion,
        "per_class_recall": {c: float(recalls[i]) for i, c in enumerate(official_classes())
                             if present[i]},
    }


# ===========================================================================
# Task 5 - Embedding the Colab-trained weights
# ===========================================================================
def encode_model_to_b64(model: torch.nn.Module) -> str:
    """Run this on Colab after training and paste the output into TRAINED_MODEL_B64."""
    state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state, buf)
    compressed = zlib.compress(buf.getvalue(), level=9)
    return base64.b64encode(compressed).decode("ascii")


def load_trained_model(device: torch.device) -> torch.nn.Module:
    """Rebuilds build_model(9) and loads TRAINED_MODEL_B64 on CMS."""
    if not TRAINED_MODEL_B64.strip():
        raise RuntimeError("TRAINED_MODEL_B64 is empty - train on Colab and embed the weights.")
    model = build_model(9)
    raw = zlib.decompress(base64.b64decode(TRAINED_MODEL_B64.strip()))
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ===========================================================================
# main() - standard Colab run with the official training protocol
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available(), "| Device:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    batch_size = int(os.environ.get("CAME_BATCH_SIZE", 64 if device.type == "cuda" else 32))
    epochs = int(os.environ.get("CAME_EPOCHS", 12))

    model_probe = build_model(9)
    print(f"Architecture BEST_CONFIG={BEST_CONFIG} | "
          f"trainable params={count_parameters(model_probe):,}")

    train_ds, val_ds = build_datasets()
    print(f"train={len(train_ds)} val={len(val_ds)} batch_size={batch_size} epochs={epochs}")
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size)

    t0 = time.time()
    model, history = train_model(train_loader, val_loader, epochs, device)
    print(f"Training finished in {time.time() - t0:.1f}s")

    y_true = torch.cat([labels for _, labels in val_loader]).long()
    report = classification_report(y_true, predict(model, val_loader, device))
    print(f"Validation (best checkpoint): accuracy={report['accuracy']:.4f} "
          f"macro_recall={report['macro_recall']:.4f}")
    print(report["confusion"])

    b64 = encode_model_to_b64(model)
    Path("trained_model_b64.txt").write_text(b64)
    print(f"Wrote trained_model_b64.txt ({len(b64):,} chars). Paste it into TRAINED_MODEL_B64 "
          f"(or use train_colab.py, which embeds it automatically).")

    if TRAINED_MODEL_B64.strip():
        embedded = load_trained_model(torch.device("cpu"))
        rep = classification_report(y_true, predict(embedded, val_loader, torch.device("cpu")))
        print(f"Embedded TRAINED_MODEL_B64 on val: accuracy={rep['accuracy']:.4f} "
              f"macro_recall={rep['macro_recall']:.4f}")


# ---------------------------------------------------------------------------
# Colab-trained weights (re-written automatically by train_colab.py)
# ---------------------------------------------------------------------------
# >>> TRAINED_MODEL_B64 >>>
TRAINED_MODEL_B64 = """"""
# <<< TRAINED_MODEL_B64 <<<


def _run_on_cms_if_needed() -> bool:
    zpath = Path("cms_eval_data.zip")
    if not zpath.exists():
        return False
    import sys, runpy, zipfile
    sys.modules["a02_cnn"] = sys.modules["__main__"]
    with zipfile.ZipFile(zpath) as archive:
        archive.extract("cms_grader.py", ".")
    runpy.run_path("cms_grader.py", run_name="__main__")
    return True


if __name__ == "__main__":
    if not _run_on_cms_if_needed():
        main()
