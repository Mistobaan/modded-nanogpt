#!/usr/bin/env python3
"""Build a training notebook from repository assets and ship log archives upstream."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("requests package is required to run this script") from exc


@dataclass
class NotebookAssets:
    config_cells: List[dict]
    download_cell: dict


def load_setup_assets(setup_path: Path) -> NotebookAssets:
    raw = json.loads(setup_path.read_text())
    config_cells: List[dict] = []
    download_cell: dict | None = None
    torch_cell: dict | None = None

    for cell in raw.get("cells", []):
        text = "".join(cell.get("source", []))
        if "# download fine web" in text:
            download_cell = _sanitize_cell(cell)
        elif "PyTorch version:" in text:
            torch_cell = _sanitize_cell(cell)
        elif download_cell is None:
            config_cells.append(_sanitize_cell(cell))

    if download_cell is None:
        raise ValueError("Unable to locate the fineweb download cell in setup notebook")

    if torch_cell is not None:
        config_cells.append(torch_cell)

    return NotebookAssets(config_cells=config_cells, download_cell=download_cell)


def _sanitize_cell(cell: dict) -> dict:
    base = {
        "cell_type": cell["cell_type"],
        "metadata": cell.get("metadata", {}),
        "source": list(cell.get("source", [])),
    }
    if cell["cell_type"] == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def _as_source(text: str) -> List[str]:
    clean = text.replace("\r\n", "\n")
    if not clean.endswith("\n"):
        clean += "\n"
    return clean.splitlines(keepends=True)


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _as_source(source),
    }


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _as_source(source),
    }


def make_writefile_cell(filename: str, content: str) -> dict:
    return make_code_cell(f"%%writefile {filename}\n{content}")


def build_library_versions_cell(distributions: Sequence[str]) -> dict:
    header = "from importlib.metadata import PackageNotFoundError, version\n\n"
    listing = (
        "libraries = [\n"
        + "".join(f'    "{name}",\n' for name in distributions)
        + "]\n\n"
    )
    body = (
        "for dist in libraries:\n"
        "    try:\n"
        '        print(f"{dist}: {version(dist)}")\n'
        "    except PackageNotFoundError:\n"
        '        print(f"{dist}: <not installed>")\n'
    )
    return make_code_cell(header + listing + body)


def create_notebook(
    assets: NotebookAssets,
    train_py: Path,
    train_sh: Path,
    torchrun_sh: Path,
    output_path: Path,
    extra_libraries: Sequence[str],
) -> None:
    cells: List[dict] = []
    cells.extend(assets.config_cells)
    cells.append(build_library_versions_cell(extra_libraries))
    cells.append(assets.download_cell)
    cells.append(_build_writefile_cell(train_py, "train_gpy.py"))
    cells.append(_build_bash_cell_inline("nvidia-smi topo -m"))
    cells.append(_build_writefile_cell(train_sh, "train_gpt.sh"))
    cells.append(_build_bash_cell(torchrun_sh))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": ".".join(map(str, sys.version_info[:3])),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2))


def _build_writefile_cell(source_path: Path, target_name: str) -> dict:
    content = source_path.read_text()
    return make_writefile_cell(target_name, content)


def _build_bash_cell(source_path: Path) -> dict:
    body = source_path.read_text()
    comment = f"%%bash\n# {source_path.name}\n"
    return make_code_cell(comment + body)


def _build_bash_cell_inline(body: str) -> dict:
    comment = "%%bash\n"
    return make_code_cell(comment + body)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-output",
        default="notebooks/generated_training_bundle.ipynb",
        type=Path,
        help="Path for the generated notebook",
    )
    parser.add_argument(
        "--setup-notebook",
        default=Path("notebooks/setup-8xH100.ipynb"),
        type=Path,
        help="Path to the source setup notebook",
    )
    parser.add_argument(
        "--train-script",
        default=Path("train_gpt.py"),
        type=Path,
        help="Python training script to embed",
    )
    parser.add_argument(
        "--train-shell",
        default=Path("train_gpt.sh"),
        type=Path,
        help="Shell training script to embed",
    )
    parser.add_argument(
        "--torchrun-shell",
        default=Path("torchrun_train_gpt.sh"),
        type=Path,
        help="Torchrun shell script to embed",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Check git status to ensure everything is committed
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit("Failed to check git status")
    if result.stdout.strip():
        raise SystemExit(
            "There are uncommitted changes. Please commit all changes before creating the notebook."
        )

    # Get short git hash
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit("Failed to get git hash")
    hash_short = result.stdout.strip()

    # Modify output path to include hash as prefix
    original_path = args.notebook_output
    new_name = f"{hash_short}_{original_path.name}"
    output_path = original_path.parent / new_name

    assets = load_setup_assets(args.setup_notebook)

    libraries = []
    base_libraries = ["torch", "huggingface-hub", "requests"]
    extra_libraries = list(dict.fromkeys(base_libraries + libraries))

    create_notebook(
        assets=assets,
        train_py=args.train_script,
        train_sh=args.train_shell,
        torchrun_sh=args.torchrun_shell,
        output_path=output_path,
        extra_libraries=extra_libraries,
    )

    print(f"created {output_path}")


if __name__ == "__main__":
    main()
