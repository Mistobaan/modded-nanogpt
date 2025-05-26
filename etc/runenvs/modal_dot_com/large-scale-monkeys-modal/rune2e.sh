#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

command -v modal >/dev/null 2>&1 || { echo >&2 "modal command not found. Install modal first! Aborting."; exit 1; }

echo 'downloading LLaMA 3.1 8B'
echo 'make sure to create a Secret called huggingface on Modal and accept the LLaMA 3.1 license'
modal run download_llama.py
echo 'deploying vLLM inference server'
modal deploy inference.py
echo 'running HumanEval generation'
modal run client.py --data-dir test --no-dry-run --n 1000 --subsample 100
echo 'running HumanEval evaluation'
modal run eval.py::find_missing_files
echo 'run "modal launch jupyter --volume humaneval" and upload the notebook to run the analysis'
