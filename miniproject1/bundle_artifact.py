#!/usr/bin/env python3
import shutil, zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
zip_path = root / "Group1_MiniProject1.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(root / "overleaf_master.tex", arcname="main.tex")
    z.write(root / "SUBMISSION_README.md", arcname="SUBMISSION_README.md")
    for p in (root / "figures").rglob("*.png"):
        z.write(p, arcname=str(Path("figures") / p.name))
    code_files = ["DM-logic.py","EDF-logic.py","Task-model.py","analysis.py","simulation.py","generate_harmonic_tasksets.py","harmonic_analysis.log"]
    for name in code_files:
        p = root / name
        if p.exists():
            z.write(p, arcname=str(Path("code") / name))
    for p in (root / "Task-sets").rglob("*.csv"):
        z.write(p, arcname=str(Path("code") / "Task-sets" / p.relative_to(root / "Task-sets")))
    for p in (root / "Task-sets-harmonic").rglob("*.csv"):
        z.write(p, arcname=str(Path("code") / "Task-sets-harmonic" / p.relative_to(root / "Task-sets-harmonic")))

print(f"Created {zip_path}")
