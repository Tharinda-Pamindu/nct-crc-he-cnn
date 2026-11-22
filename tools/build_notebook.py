"""Build train_colab.ipynb from train_colab.py (# %% cells) + assignment02.py.

Usage:  python tools/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": to_source(text)}


def parse_percent_script(path: Path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is not None and "".join(buf).strip():
            body = "\n".join(buf)
            if kind == "markdown":
                body = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                                 for ln in buf)
                cells.append(md(body))
            else:
                cells.append(code(body))

    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    script_cells = parse_percent_script(ROOT / "train_colab.py")
    submission = (ROOT / "assignment02.py").read_text(encoding="utf-8")

    cells = [script_cells[0]]  # title / overview
    cells.append(code(
        "# Run this in the first cell of your Google Colab notebook:\n"
        "!pip install -q torch torchvision datasets pillow\n"
        "import torch\n"
        "print('CUDA available:', torch.cuda.is_available(), '| Device:',\n"
        "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"))
    cells.append(md(
        "## Submission module `assignment02.py`\n"
        "This cell writes the template the pipeline imports. After training, the pipeline\n"
        "writes the final file with the tuned `BEST_CONFIG` and the embedded\n"
        "`TRAINED_MODEL_B64` to `/content/came_a02/submission/assignment02.py`. Upload\n"
        "that file to CMS."))
    cells.append(code("%%writefile assignment02.py\n" + submission))
    cells.extend(script_cells[1:])

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = ROOT / "train_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
