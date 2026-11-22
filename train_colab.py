# %% [markdown]
# # CAME Assignment 02 - Colab GPU pipeline: Ray Tune + MLflow
#
# **Runtime -> Change runtime type -> T4 GPU**, then run every cell (or
# `!python train_colab.py`). `assignment02.py` must sit next to this file / notebook.
#
# Pipeline:
# 1. Load a deterministic balanced subset of `1aurent/NCT-CRC-HE` (`NCT_CRC_HE_100K`),
#    resized to 64x64, split 85/15 with seed 42 (the same split `build_datasets()` uses).
# 2. **Ray Tune** (Optuna TPE search + ASHA early stopping) over architecture and training
#    hyper-parameters, 4 trials at a time on one T4. Objective: validation macro recall.
# 3. Every trial plus the final run is tracked in **MLflow**.
# 4. Train the final model with the best config, then write
#    `submission/assignment02.py` with `BEST_CONFIG` and the `TRAINED_MODEL_B64` weights.
# 5. Verify the generated file: `load_trained_model()`, the official PIL preprocessing,
#    an unseen-patient check on `CRC_VAL_HE_7K`, and a simulated CMS Stage A on CPU.

# %%
# --- Setup: install packages and check the GPU ------------------------------
import importlib.util
import subprocess
import sys


def ensure_packages(spec_to_module: dict) -> None:
    missing = [spec for spec, mod in spec_to_module.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print("pip install", *missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages({
    "torch": "torch", "torchvision": "torchvision", "datasets": "datasets",
    "pillow": "PIL", "huggingface_hub": "huggingface_hub", "pyarrow": "pyarrow",
    "ray[tune]>=2.40": "ray", "optuna": "optuna", "mlflow": "mlflow",
    "matplotlib": "matplotlib", "pandas": "pandas",
})

import torch

print("CUDA available:", torch.cuda.is_available(), "| Device:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %%
# --- Configuration -----------------------------------------------------------
import json
import math
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch.nn as nn

IN_COLAB = Path("/content").exists()
try:
    get_ipython  # noqa: F821  (defined inside Jupyter / Colab)
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False

CODE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
WORKDIR = Path(os.environ.get("CAME_WORKDIR",
                              "/content/came_a02" if IN_COLAB else CODE_DIR / "came_a02_out"))
WORKDIR.mkdir(parents=True, exist_ok=True)

PER_CLASS = int(os.environ.get("CAME_PER_CLASS", 1000))   # 9 x 1000 balanced patches
os.environ["CAME_PER_CLASS"] = str(PER_CLASS)
os.environ.setdefault("CAME_CACHE_DIR", str(WORKDIR / "cache"))

# Ray Tune budget (about 10-15 min on a T4)
TUNE_NUM_SAMPLES = int(os.environ.get("CAME_TUNE_SAMPLES", 40))
TUNE_MAX_EPOCHS = int(os.environ.get("CAME_TUNE_EPOCHS", 30))
TUNE_GRACE_EPOCHS = 8
GPU_PER_TRIAL = 0.25            # 4 concurrent trials share the T4
CPU_PER_TRIAL = 0.5
FINAL_EPOCHS = int(os.environ.get("CAME_FINAL_EPOCHS", 40))
TUNE_METRIC, TUNE_MODE = "val_macro_recall", "max"

RUN_BASELINE = True             # official train_model() (AdamW 1e-3, 12 epochs) for reference
RUN_EXTERNAL_EVAL = True        # CRC_VAL_HE_7K = 50 patients not in the 100K split

MLFLOW_EXPERIMENT = "came-a02-histopathology-cnn"
MLFLOW_TRACKING_URI = f"sqlite:///{(WORKDIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (WORKDIR / "mlartifacts").resolve()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"WORKDIR={WORKDIR}  CODE_DIR={CODE_DIR}  DEVICE={DEVICE}")

# %%
# --- Import the submission module (same code CMS grades) ---------------------
sys.path.insert(0, str(CODE_DIR))
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in [str(CODE_DIR), os.environ.get("PYTHONPATH", "")] if p)
import assignment02 as a02  # noqa: E402

CLASSES = a02.official_classes()
print("classes:", CLASSES)

# %%
# --- Data: balanced subset of NCT_CRC_HE_100K -> 85/15 split (seed 42) ---------
x_tr, y_tr, x_va, y_va = a02.get_split_arrays(PER_CLASS)
print(f"train {x_tr.shape}  val {x_va.shape}")
print("train per class:", np.bincount(y_tr, minlength=9).tolist())
print("val   per class:", np.bincount(y_va, minlength=9).tolist())

import matplotlib
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS = WORKDIR / "plots"
PLOTS.mkdir(exist_ok=True)


def show(fig, name):
    fig.savefig(PLOTS / name, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


fig, axes = plt.subplots(3, 9, figsize=(13, 4.6))
for c in range(9):
    idx = np.flatnonzero(y_tr == c)[:3]
    for r, i in enumerate(idx):
        axes[r, c].imshow(x_tr[i])
        axes[r, c].axis("off")
    axes[0, c].set_title(CLASSES[c])
fig.suptitle("Training patches (64x64)")
show(fig, "samples.png")

# %%
# --- GPU training utilities shared by Ray Tune trials and the final run ------
# The whole uint8 dataset lives on the GPU and augmentation runs batched on the GPU,
# so a trial epoch takes about 1-2 s. Normalisation matches make_transforms():
# ToTensor() + Normalize(0.5, 0.5).

def normalize_uint8(xb: torch.Tensor) -> torch.Tensor:
    return (xb.float() / 255.0 - 0.5) / 0.5


def gpu_augment(xb: torch.Tensor, cfg: dict) -> torch.Tensor:
    """xb: uint8 (N,3,H,W) on device -> augmented, normalised float tensor."""
    x = xb.float() / 255.0
    n, dev = x.size(0), x.device
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)        # horizontal flip
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(2), x)        # vertical flip
    if cfg.get("rot90"):                                            # tissue has no "up"
        k = torch.randint(0, 2, (n,), device=dev).bool()
        x = torch.where(k[:, None, None, None], x.transpose(2, 3), x)
    s = float(cfg.get("color_jitter", 0.0))
    if s > 0:                                                       # stain / scanner shift
        def u(*shape):
            return 1.0 + (torch.rand(*shape, device=dev) * 2 - 1) * s
        x = x * u(n, 1, 1, 1)                                       # brightness
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * u(n, 1, 1, 1) + m                             # contrast
        g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = (x - g) * u(n, 1, 1, 1) + g                             # saturation
        x = x * (1.0 + (torch.rand(n, 3, 1, 1, device=dev) * 2 - 1) * s * 0.5)  # channel gain
        x = x.clamp(0.0, 1.0)
    return (x - 0.5) / 0.5


@torch.no_grad()
def evaluate_arrays(model, X, Y, batch_size=1024):
    """X uint8 (N,3,H,W) and Y on the model's device -> (val_loss, classification_report)."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, preds = 0.0, []
    for i in range(0, X.size(0), batch_size):
        logits = model(normalize_uint8(X[i:i + batch_size])).float()
        loss_sum += crit(logits, Y[i:i + batch_size]).item()
        preds.append(logits.argmax(dim=1))
    report = a02.classification_report(Y.cpu(), torch.cat(preds).cpu())
    return loss_sum / max(X.size(0), 1), report


def to_device_arrays(data, device):
    xt, yt, xv, yv = data
    with warnings.catch_warnings():      # Ray hands over read-only numpy arrays
        warnings.simplefilter("ignore")
        Xtr = torch.from_numpy(np.asarray(xt)).to(device).permute(0, 3, 1, 2).contiguous()
        Xva = torch.from_numpy(np.asarray(xv)).to(device).permute(0, 3, 1, 2).contiguous()
        Ytr = torch.from_numpy(np.asarray(yt)).long().to(device)
        Yva = torch.from_numpy(np.asarray(yv)).long().to(device)
    return Xtr, Ytr, Xva, Yva


def fit(cfg: dict, data, device, epochs: int, on_epoch=None, seed: int = 42):
    """AdamW + OneCycle + AMP. Keeps the epoch with the best val macro recall."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = a02.build_model_from_config(9, cfg).to(device)
    Xtr, Ytr, Xva, Yva = to_device_arrays(data, device)
    n, bs = Xtr.size(0), int(cfg["batch_size"])
    steps = math.ceil(n / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=epochs,
                                                steps_per_epoch=steps, pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"val_macro_recall": -1.0, "val_loss": float("inf")}
    best_state, history = None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb, yb = gpu_augment(Xtr[idx], cfg), Ytr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach().float() * idx.numel()
            correct += (logits.argmax(dim=1) == yb).sum()
        val_loss, rep = evaluate_arrays(model, Xva, Yva)
        m = {
            "epoch": epoch + 1,
            "train_loss": float(loss_sum.item() / n),
            "train_acc": float(correct.item() / n),
            "val_loss": float(val_loss),
            "val_acc": rep["accuracy"],
            "val_macro_recall": rep["macro_recall"],
            "lr": float(sched.get_last_lr()[0]),
        }
        if (m["val_macro_recall"] > best["val_macro_recall"] + 1e-12 or
                (abs(m["val_macro_recall"] - best["val_macro_recall"]) <= 1e-12
                 and m["val_loss"] < best["val_loss"])):
            best = dict(m)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        m["best_val_macro_recall"] = best["val_macro_recall"]
        history.append(m)
        if on_epoch is not None:
            on_epoch(m)
    model.load_state_dict(best_state)
    return model, history, best


# Architectures are "width-n_blocks-convs_per_block". All fit the 25k-500k
# parameter budget and stay under ~120 MMACs/image, so CMS CPU training stays fast.
ARCH_CHOICES = ["16-4-2", "24-3-2", "24-4-1", "32-3-1", "32-3-2", "32-4-1", "48-3-1"]
ARCH_KEYS = ("width", "n_blocks", "convs_per_block", "head_hidden", "dropout")


def expand_config(config: dict) -> dict:
    cfg = dict(config)
    w, b, k = (int(v) for v in str(cfg.pop("arch")).split("-"))
    cfg.update(width=w, n_blocks=b, convs_per_block=k)
    return cfg


for arch in ARCH_CHOICES:
    probe = a02.build_model_from_config(9, expand_config({"arch": arch, "head_hidden": 128,
                                                          "dropout": 0.3}))
    print(f"{arch:>7}: {a02.count_parameters(probe):>8,} params")

# %%
# --- MLflow tracking ---------------------------------------------------------
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
    mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.as_uri())
mlflow.set_experiment(MLFLOW_EXPERIMENT)
print("MLflow tracking URI:", MLFLOW_TRACKING_URI)

# %%
# --- (Optional) Baseline: the official train_model() exactly as CMS runs it --
# Runs build_datasets() -> make_loaders() -> train_model() (AdamW lr=1e-3, wd=1e-4,
# 12 epochs, batch 64) with the default BEST_CONFIG, for comparison with the tuned model.
baseline = None
if RUN_BASELINE:
    os.environ.setdefault("CAME_WORKERS", "2")
    t0 = time.time()
    tr_ds, va_ds = a02.build_datasets()
    tr_ld, va_ld = a02.make_loaders(tr_ds, va_ds, batch_size=64)
    with mlflow.start_run(run_name="baseline-official-train_model"):
        mlflow.log_params({**a02.BEST_CONFIG, "optimizer": "AdamW", "lr": 1e-3,
                           "weight_decay": 1e-4, "epochs": 12, "batch_size": 64,
                           "per_class": PER_CLASS, "pipeline": "official"})
        base_model, base_hist = a02.train_model(tr_ld, va_ld, 12, DEVICE)
        for step, h in enumerate(base_hist, start=1):
            mlflow.log_metrics(h, step=step)
        y_true = torch.tensor(va_ds.labels)
        baseline = a02.classification_report(y_true, a02.predict(base_model, va_ld, DEVICE))
        mlflow.log_metrics({"final_val_acc": baseline["accuracy"],
                            "final_val_macro_recall": baseline["macro_recall"],
                            "train_seconds": time.time() - t0})
    print(f"Baseline: acc={baseline['accuracy']:.4f} macro_recall={baseline['macro_recall']:.4f}"
          f" ({time.time() - t0:.0f}s)")
    os.environ.pop("CAME_WORKERS", None)

# %%
# --- Ray Tune: hyper-parameter search ----------------------------------------
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

try:
    from ray.tune import RunConfig
except ImportError:  # older Ray
    from ray.train import RunConfig
try:
    from ray.air.integrations.mlflow import MLflowLoggerCallback
except ImportError:
    from ray.tune.integration.mlflow import MLflowLoggerCallback


def tune_report(metrics: dict) -> None:
    try:
        tune.report(metrics)
    except (AttributeError, TypeError):
        from ray import train as ray_train
        ray_train.report(metrics)


def tune_trainable(config, data=None):
    # Do not touch torch.backends.cudnn here: Ray pickles this function and
    # cloudpickle cannot serialise torch's CudnnModule ("cannot pickle 'CudnnModule'").
    torch.set_num_threads(1)
    cfg = expand_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_params = a02.count_parameters(a02.build_model_from_config(9, cfg))
    if not (a02.PARAM_MIN <= n_params <= a02.PARAM_MAX):
        tune_report({"val_macro_recall": 0.0, "val_acc": 0.0, "val_loss": 99.0,
                     "params": n_params, "epoch": 1})
        return
    fit(cfg, data, device, epochs=int(config["epochs"]),
        on_epoch=lambda m: tune_report({**m, "params": n_params}))


search_space = {
    "arch": tune.choice(ARCH_CHOICES),
    "head_hidden": tune.choice([0, 64, 128]),
    "dropout": tune.uniform(0.0, 0.5),
    "lr": tune.loguniform(3e-4, 6e-3),
    "weight_decay": tune.loguniform(1e-5, 5e-3),
    "batch_size": tune.choice([32, 64, 128]),
    "label_smoothing": tune.choice([0.0, 0.05, 0.1]),
    "color_jitter": tune.uniform(0.0, 0.3),
    "rot90": tune.choice([True, False]),
    "epochs": TUNE_MAX_EPOCHS,
}

try:
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    max_conc = max(1, int(round(1 / GPU_PER_TRIAL)))
    search_alg = ConcurrencyLimiter(OptunaSearch(seed=42), max_concurrent=max_conc)
    print("Search algorithm: Optuna TPE")
except Exception as exc:  # optuna missing -> random search
    print("Optuna unavailable, using random search:", exc)
    search_alg = None

scheduler = ASHAScheduler(time_attr="training_iteration", max_t=TUNE_MAX_EPOCHS,
                          grace_period=TUNE_GRACE_EPOCHS, reduction_factor=3)

ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False,
         runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})

trainable = tune.with_resources(
    tune.with_parameters(tune_trainable, data=(x_tr, y_tr, x_va, y_va)),
    {"cpu": CPU_PER_TRIAL, "gpu": GPU_PER_TRIAL if torch.cuda.is_available() else 0})

tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    tune_config=tune.TuneConfig(metric=TUNE_METRIC, mode=TUNE_MODE, scheduler=scheduler,
                                search_alg=search_alg, num_samples=TUNE_NUM_SAMPLES),
    run_config=RunConfig(
        name="came_a02_tune",
        storage_path=str((WORKDIR / "ray_results").resolve()),
        callbacks=[MLflowLoggerCallback(tracking_uri=MLFLOW_TRACKING_URI,
                                        experiment_name=MLFLOW_EXPERIMENT,
                                        tags={"stage": "ray-tune"},
                                        save_artifact=False)],
    ),
)
t0 = time.time()
results = tuner.fit()
print(f"Ray Tune finished in {(time.time() - t0) / 60:.1f} min")

# %%
# --- Ray Tune leaderboard and best configuration -----------------------------
import pandas as pd

df = results.get_dataframe(filter_metric=TUNE_METRIC, filter_mode=TUNE_MODE)
cols = [c for c in df.columns if c.startswith("config/") and c != "config/epochs"]
board = (df[[TUNE_METRIC, "val_acc", "val_loss", "training_iteration", "params", *cols]]
         .sort_values(TUNE_METRIC, ascending=False).reset_index(drop=True))
board.columns = [c.replace("config/", "") for c in board.columns]
board.to_csv(WORKDIR / "ray_tune_leaderboard.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(board.head(10).to_string())

best_result = results.get_best_result(metric=TUNE_METRIC, mode=TUNE_MODE, scope="all")
best_cfg = expand_config({k: v for k, v in best_result.config.items() if k != "epochs"})
print("\nBest trial config:", json.dumps(best_cfg, indent=2))
ray.shutdown()

# %%
# --- Final training with the best configuration (tracked in MLflow) ----------
while mlflow.active_run() is not None:
    mlflow.end_run()
with mlflow.start_run(run_name="final-best-config") as final_run:
    mlflow.log_params({**best_cfg, "epochs": FINAL_EPOCHS, "per_class": PER_CLASS,
                       "train_size": len(y_tr), "val_size": len(y_va)})
    mlflow.log_artifact(str(WORKDIR / "ray_tune_leaderboard.csv"))
    t0 = time.time()

    def log_epoch(m):
        mlflow.log_metrics({k: v for k, v in m.items() if k != "epoch"}, step=m["epoch"])
        print(f"epoch {m['epoch']:02d} | train_loss {m['train_loss']:.4f} acc {m['train_acc']:.4f}"
              f" | val_loss {m['val_loss']:.4f} acc {m['val_acc']:.4f}"
              f" macro_recall {m['val_macro_recall']:.4f}", flush=True)

    final_model, final_hist, final_best = fit(best_cfg, (x_tr, y_tr, x_va, y_va), DEVICE,
                                              epochs=FINAL_EPOCHS, on_epoch=log_epoch)
    train_seconds = time.time() - t0
    n_params = a02.count_parameters(final_model)
    mlflow.log_metrics({"train_seconds": train_seconds, "params": n_params,
                        "best_val_macro_recall": final_best["val_macro_recall"],
                        "best_val_acc": final_best["val_acc"],
                        "best_epoch": final_best["epoch"]})
    torch.save(final_model.state_dict(), WORKDIR / "best_model.pt")
    mlflow.log_artifact(str(WORKDIR / "best_model.pt"))
    (WORKDIR / "final_history.json").write_text(json.dumps(final_hist, indent=2))
    mlflow.log_artifact(str(WORKDIR / "final_history.json"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = [h["epoch"] for h in final_hist]
    ax[0].plot(ep, [h["train_loss"] for h in final_hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in final_hist], label="val")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in final_hist], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in final_hist], label="val acc")
    ax[1].plot(ep, [h["val_macro_recall"] for h in final_hist], label="val macro recall")
    ax[1].set_title("accuracy / recall"); ax[1].set_xlabel("epoch"); ax[1].legend()
    show(fig, "final_curves.png")
    mlflow.log_artifact(str(PLOTS / "final_curves.png"))

print(f"\nFinal model: {n_params:,} params, best epoch {final_best['epoch']}, "
      f"val acc {final_best['val_acc']:.4f}, val macro recall "
      f"{final_best['val_macro_recall']:.4f} ({train_seconds:.0f}s)")

# %%
# --- Write submission/assignment02.py with BEST_CONFIG + TRAINED_MODEL_B64 ----
import re
import textwrap

ARCH_ONLY = {k: best_cfg[k] for k in ARCH_KEYS}
TRAINING_INFO = {k: v for k, v in best_cfg.items() if k not in ARCH_KEYS}
TRAINING_INFO.update(epochs=FINAL_EPOCHS, optimizer="AdamW", scheduler="OneCycleLR",
                     amp=True, per_class=PER_CLASS, tuned_with="Ray Tune (Optuna TPE + ASHA)")


def py_dict(d: dict) -> str:
    body = "".join(f"    {k!r}: {v!r},\n" for k, v in d.items())
    return "{\n" + body + "}"


b64 = a02.encode_model_to_b64(final_model)
src = (CODE_DIR / "assignment02.py").read_text(encoding="utf-8")
src = re.sub(r"# >>> BEST_CONFIG >>>.*?# <<< BEST_CONFIG <<<",
             lambda _: ("# >>> BEST_CONFIG >>>\n"
                        f"BEST_CONFIG = {py_dict(ARCH_ONLY)}\n"
                        f"TUNED_TRAINING_CONFIG = {py_dict(TRAINING_INFO)}  # informational\n"
                        "# <<< BEST_CONFIG <<<"),
             src, flags=re.S)
wrapped = "\n".join(textwrap.wrap(b64, 100))
src = re.sub(r"# >>> TRAINED_MODEL_B64 >>>.*?# <<< TRAINED_MODEL_B64 <<<",
             lambda _: ("# >>> TRAINED_MODEL_B64 >>>\n"
                        f'TRAINED_MODEL_B64 = """\n{wrapped}\n"""\n'
                        "# <<< TRAINED_MODEL_B64 <<<"),
             src, flags=re.S)
SUBMISSION_DIR = WORKDIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)
SUBMISSION = SUBMISSION_DIR / "assignment02.py"
SUBMISSION.write_text(src, encoding="utf-8")
print(f"Wrote {SUBMISSION} ({SUBMISSION.stat().st_size / 1e6:.2f} MB, "
      f"b64 {len(b64):,} chars)")

# %%
# --- Verify the generated submission like CMS Stage B ------------------------
import importlib.util as ilu


def import_submission(path: Path, name: str = "a02_submission"):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


os.environ.pop("CAME_DATA_DIR", None)
sub = import_submission(SUBMISSION)
cpu = torch.device("cpu")
loaded = sub.load_trained_model(cpu)
n_conv = sum(isinstance(m, nn.Conv2d) for m in loaded.modules())
allowed = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
           nn.Flatten, nn.Dropout, nn.Linear, nn.Sequential)
leaf_ok = all(isinstance(m, allowed) for m in loaded.modules() if not list(m.children()))
out = loaded(torch.randn(4, 3, 64, 64))
n_params = sub.count_parameters(sub.build_model(9))
print(f"Conv2d layers={n_conv} | leaf layers allowed={leaf_ok} | logits={tuple(out.shape)} | "
      f"params={n_params:,} (budget ok: {sub.PARAM_MIN <= n_params <= sub.PARAM_MAX})")
assert n_conv >= 3 and leaf_ok and out.shape == (4, 9)
assert sub.PARAM_MIN <= n_params <= sub.PARAM_MAX

# Official PIL preprocessing (make_transforms(False)) on the validation split
val_ds_official = sub.ArrayImageDataset(x_va, y_va, transform=sub.make_transforms(False))
_, val_ld_official = sub.make_loaders(val_ds_official, val_ds_official, batch_size=256)
val_rep = sub.classification_report(torch.tensor(y_va),
                                    sub.predict(loaded, val_ld_official, cpu))
print(f"Embedded fp16 model on validation (official pipeline): "
      f"acc={val_rep['accuracy']:.4f} macro_recall={val_rep['macro_recall']:.4f}")


def plot_confusion(cm: torch.Tensor, title: str, fname: str):
    cm = cm.numpy()
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASSES, rotation=45)
    ax.set_yticks(range(9), CLASSES)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    show(fig, fname)


plot_confusion(val_rep["confusion"], "Validation (NCT-CRC-HE-100K subset)", "cm_val.png")

# %%
# --- Unseen-patient check: CRC_VAL_HE_7K (50 patients not in the 100K split) ---
# Report only. It is never used for tuning or model selection, because the hidden
# CMS test set may come from the same source.
ext_rep = None
if RUN_EXTERNAL_EVAL:
    ext_cache = Path(os.environ["CAME_CACHE_DIR"]) / f"CRC_VAL_HE_7K_{a02.IMAGE_SIZE}px.npz"
    if ext_cache.exists():
        d = np.load(ext_cache)
        x_ext, y_ext = d["x"], d["y"]
    else:
        t0 = time.time()
        fs, files = a02._hf_parquet_files("CRC_VAL_HE_7K")
        jobs = [(p, rg, None, None, 0) for p in files
                for (rg, _, _, _) in a02._rowgroup_labels(fs, p)]
        x_ext, y_ext = a02._run_rowgroup_jobs(fs, jobs)
        np.savez(ext_cache, x=x_ext, y=y_ext)
        print(f"Loaded CRC_VAL_HE_7K: {len(y_ext)} patches in {time.time() - t0:.0f}s")
    ext_ds = sub.ArrayImageDataset(x_ext, y_ext, transform=sub.make_transforms(False))
    _, ext_ld = sub.make_loaders(ext_ds, ext_ds, batch_size=256)
    ext_rep = sub.classification_report(torch.tensor(y_ext), sub.predict(loaded, ext_ld, cpu))
    print(f"CRC_VAL_HE_7K: acc={ext_rep['accuracy']:.4f} "
          f"macro_recall={ext_rep['macro_recall']:.4f}")
    print("per-class recall:", {k: round(v, 3) for k, v in ext_rep["per_class_recall"].items()})
    plot_confusion(ext_rep["confusion"], "Unseen patients (CRC-VAL-HE-7K)", "cm_external.png")

# %%
# --- Simulated CMS Stage A: ImageFolder + 2 CPU epochs with the official functions ---
from PIL import Image

cms_dir = WORKDIR / "cms_dryrun"
shutil.rmtree(cms_dir, ignore_errors=True)
rng = np.random.default_rng(0)
for split, n_per in (("train", 40), ("validation", 15)):
    for c, name in enumerate(CLASSES):
        (cms_dir / split / name).mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(rng.choice(np.flatnonzero(y_va == c), n_per, replace=False)):
            Image.fromarray(x_va[i]).save(cms_dir / split / name / f"{name}_{k}.png")

os.environ["CAME_DATA_DIR"] = str(cms_dir)
try:
    t0 = time.time()
    tr_ds, va_ds = sub.build_datasets()
    tr_ld, va_ld = sub.make_loaders(tr_ds, va_ds, batch_size=32)
    m_a, hist_a = sub.train_model(tr_ld, va_ld, 2, cpu)
    y_true = torch.tensor(va_ds.targets)
    rep_a = sub.classification_report(y_true, sub.predict(m_a, va_ld, cpu))
    stage_a_seconds = time.time() - t0
    assert tr_ds.classes == CLASSES and len(hist_a) == 2
    assert rep_a["confusion"].shape == (9, 9) and rep_a["confusion"].dtype == torch.long
    print(f"Stage A dry run OK in {stage_a_seconds:.1f}s on CPU "
          f"({len(tr_ds)} train / {len(va_ds)} val images): {hist_a}")
finally:
    os.environ.pop("CAME_DATA_DIR", None)

# %%
# --- Log verification results to MLflow and package outputs ------------------
with mlflow.start_run(run_id=final_run.info.run_id):
    mlflow.log_metrics({"embedded_val_acc": val_rep["accuracy"],
                        "embedded_val_macro_recall": val_rep["macro_recall"],
                        "cms_stage_a_cpu_seconds": stage_a_seconds})
    if ext_rep is not None:
        mlflow.log_metrics({"external_7k_acc": ext_rep["accuracy"],
                            "external_7k_macro_recall": ext_rep["macro_recall"],
                            **{f"external_7k_recall_{k}": v
                               for k, v in ext_rep["per_class_recall"].items()}})
    for png in PLOTS.glob("*.png"):
        mlflow.log_artifact(str(png), artifact_path="plots")
    mlflow.log_artifact(str(SUBMISSION), artifact_path="submission")

summary = {
    "best_config": best_cfg,
    "params": n_params,
    "val_acc": val_rep["accuracy"],
    "val_macro_recall": val_rep["macro_recall"],
    "external_7k_acc": None if ext_rep is None else ext_rep["accuracy"],
    "external_7k_macro_recall": None if ext_rep is None else ext_rep["macro_recall"],
    "baseline_official_val_macro_recall": None if baseline is None else baseline["macro_recall"],
}
(WORKDIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

import zipfile

archive = WORKDIR.parent / "came_a02_outputs.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in WORKDIR.rglob("*"):
        rel = f.relative_to(WORKDIR)
        if f.is_file() and rel.parts[0] not in ("cache", "cms_dryrun"):
            zf.write(f, rel)
print("Outputs (MLflow db + artifacts, Ray results, plots, submission) zipped to", archive)
print("Submit:", SUBMISSION)

if IN_COLAB and IN_NOTEBOOK:
    from google.colab import files
    files.download(str(SUBMISSION))

# %% [markdown]
# ## (Optional) Browse MLflow inside Colab
# Run the next cell, then click the link it prints. Locally, unzip `came_a02_outputs.zip`
# and run `mlflow ui --backend-store-uri sqlite:///mlflow.db` inside that folder.

# %%
if IN_COLAB and IN_NOTEBOOK:
    subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI,
                      "--default-artifact-root", MLFLOW_ARTIFACTS.as_uri(),
                      "--host", "127.0.0.1", "--port", "5000"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    from google.colab import output
    output.serve_kernel_port_as_window(5000)
