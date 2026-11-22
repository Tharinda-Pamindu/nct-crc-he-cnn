# histopath-cnn-raytune

A custom CNN, trained from scratch in PyTorch, that classifies 9 colorectal tissue types in
[NCT-CRC-HE-100K](https://huggingface.co/datasets/1aurent/NCT-CRC-HE) H&E histopathology patches.
Hyper-parameters are searched with **Ray Tune** (Optuna TPE + ASHA) and every run is tracked in **MLflow**.
Training runs on a Google Colab T4 GPU. Built for CAME Assignment 02.

Classes: `ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`. Input: RGB 64x64, normalised with mean/std 0.5.

## Results

From the Colab T4 run in [`output/CAME_Assignment_02.ipynb`](output/CAME_Assignment_02.ipynb):

| Model | Validation macro recall | Unseen patients (CRC-VAL-HE-7K) macro recall |
| --- | --- | --- |
| Baseline: official `train_model()`, 12 epochs | 0.899 | not measured |
| **Ray Tune best config, 40 epochs** | **0.981** | **0.909** (accuracy 0.935) |

- Validation is 15% of a balanced 9,000-patch subset of NCT-CRC-HE-100K (seed 42). The validation
  figure is for the embedded fp16 weights with the official preprocessing.
- CRC-VAL-HE-7K is 7,180 patches from 50 patients who do not appear in the 100K split. It was used only
  for this report, never for tuning or model selection.
- Simulated CMS Stage A (2 epochs on CPU with the official functions): 8.8 s.

Per-class recall on unseen patients:

| ADI  | BACK | DEB  | LYM  | MUC  | MUS  | NORM | STR      | TUM  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | ---- |
| 0.96 | 1.00 | 0.96 | 0.99 | 0.96 | 0.88 | 0.95 | **0.52** | 0.97 |

Stroma (STR) is the weakest class: the model mostly confuses it with smooth muscle (MUS), a known
difficulty when moving from the 100K set to new patients.

## Best configuration (Ray Tune)

40 trials, about 26 minutes on one T4. All of the top 10 trials chose the same architecture.

| Parameter | Value | Range searched |
| --- | --- | --- |
| Architecture (width-blocks-convs per block) | `24-3-2` (channels 24 → 48 → 96) | 7 preset options |
| `head_hidden` | 0 (no hidden dense layer) | 0 / 64 / 128 |
| `dropout` | 0.163 | 0.0 – 0.5 |
| `lr` (AdamW + OneCycle) | 5.16e-3 | 3e-4 – 6e-3 |
| `weight_decay` | 2.01e-3 | 1e-5 – 5e-3 |
| `batch_size` | 32 | 32 / 64 / 128 |
| `label_smoothing` | 0.05 | 0 / 0.05 / 0.1 |
| `color_jitter` | 0.008 | 0 – 0.3 |
| `rot90` | True | True / False |

The final model has 157,665 trainable parameters and 5 `Conv2d` layers.

## Files

| File | Purpose |
| --- | --- |
| [`output/assignment02.py`](output/assignment02.py) | **Trained submission**: tuned `BEST_CONFIG` plus the embedded `TRAINED_MODEL_B64` weights. Upload this file to CMS. |
| [`output/CAME_Assignment_02.ipynb`](output/CAME_Assignment_02.ipynb) | The executed Colab notebook, with all outputs, from the run above. |
| [`output/colab_backup/content/came_a02/`](output/colab_backup/content/came_a02/) | Artifacts from that run: MLflow artifacts (`mlartifacts/`), `ray_tune_leaderboard.csv` (all 40 Ray Tune trials), `summary.json`, `final_history.json`, `best_model.pt` and `plots/`. The MLflow database and raw Ray Tune trial folders are not in the repo: GitHub secret scanning flags random IDs in the SQLite file, and the trial folder names are too long for Windows. |
| [`assignment02.py`](assignment02.py) | Template submission with all required functions; `TRAINED_MODEL_B64` is empty. The notebook imports it and writes the trained version. |
| [`train_colab.ipynb`](train_colab.ipynb) | Colab notebook: Ray Tune search, MLflow tracking, final training, writing the submission, verification. |
| [`train_colab.py`](train_colab.py) | The same pipeline as a script (`!python train_colab.py`). |
| [`tools/build_notebook.py`](tools/build_notebook.py) | Optional: regenerates a self-contained notebook that embeds `assignment02.py` via `%%writefile`. |

## Reproduce on Colab

1. Runtime -> Change runtime type -> **T4 GPU**.
2. Open `train_colab.ipynb`, run all cells, and upload the template `assignment02.py` when asked.
3. When the run finishes, `/content/came_a02/submission/assignment02.py` is downloaded automatically.
   That file holds the tuned config and weights.
4. Optional: the last cell opens the MLflow UI inside Colab. `came_a02_outputs.zip` holds the MLflow
   database, Ray results, plots and the submission.

Optional environment variables: `CAME_PER_CLASS` (default 1000), `CAME_TUNE_SAMPLES` (40),
`CAME_TUNE_EPOCHS` (30), `CAME_FINAL_EPOCHS` (40).

## Design notes

- **Data:** the 100K parquet shards are sorted by class, so the loader reads only selected row groups
  spread across each class block. That gives a balanced, patient-diverse subset without downloading 15 GB.
  The subset is split 85/15 per class with seed 42.
- **Model:** funnel CNN, `[Conv3x3-BN-ReLU] x k -> MaxPool2d` per block, then GAP, Dropout and Linear.
  It uses only the allowed layer types and stays within the 25k–500k parameter budget.
- **Search:** Optuna TPE proposes configurations, ASHA stops weak trials early (grace period 8 epochs,
  max 30), and 4 trials share the T4. Objective: validation macro recall.
- **Graded code:** `train_model()` follows the official protocol exactly (AdamW lr=1e-3, wd=1e-4, seed 42,
  best-val-acc checkpoint). The tuned recipe (OneCycle, AMP, GPU augmentation) is used only to produce
  the Colab weights.

