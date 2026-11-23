"""Writes vectors/now.json: the thresholds of should_use_noisy_gemm from config.yaml on Pearl's master branch.

    python3 vectors/now.py <pearl checkout> <commit>
"""
import json
import subprocess
import sys

repo, commit = sys.argv[1], sys.argv[2]
full = subprocess.run(["git", "-C", repo, "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()
text = subprocess.run(["git", "-C", repo, "show", f"{commit}:miner/vllm-miner/src/vllm_miner/config.yaml"], capture_output=True, text=True, check=True).stdout
values = {}
for line in text.splitlines():
    key, _, value = line.strip().partition(":")
    if key in ("min_m", "min_n", "min_k"):
        values[key] = int(value)
json.dump({"source": f"pearl-research-labs/pearl {full}", "thresholds": values}, sys.stdout, indent=2)
