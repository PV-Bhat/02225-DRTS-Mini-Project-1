#!/usr/bin/env python3
"""Build Overleaf submission artifacts locally.

Steps:
1) Generate harmonic dataset (if missing).
2) Run analysis to emit figures into ./figures.
3) Build Group1_MiniProject1.zip via bundle_artifact.py.
"""
import subprocess
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
fig_dir = root / "figures"
fig_dir.mkdir(exist_ok=True)

cmds = [
    [sys.executable, "generate_harmonic_tasksets.py", "--out", "Task-sets-harmonic", "--sets-per-util", "20", "--tasks", "10", "--seed", "42"],
    [sys.executable, "analysis.py", "--task-sets-dir", "Task-sets-harmonic", "--save-plots", "figures", "--runs", "50"],
    [sys.executable, "bundle_artifact.py"],
]

for cmd in cmds:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)

print("\nDone. Created:")
print("- figures/*.png")
print("- Group1_MiniProject1.zip")
