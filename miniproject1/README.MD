# 02225 DRTS Mini-Project 1 — Real-Time Schedulability Analysis

**Group 1** | Danmarks Tekniske Universitet | Spring 2026

Worst-Case Response Time (WCRT) analysis and comparison of **Deadline Monotonic (DM)** and **Earliest Deadline First (EDF)** scheduling on single-core periodic real-time task sets.

---

## Repository Structure

```
miniproject1/
├── Task-model.py              Core data structures: Task class, CSV loader, helpers
├── DM-logic.py                DM WCRT via fixed-point RTA (Buttazzo §4.5.2)
├── EDF-logic.py               EDF schedulability (PDC) + exact hyperperiod WCRT simulation
├── simulation.py              Stochastic Monte Carlo EDF simulation (ET ∈ [BCET, WCET])
├── analysis.py                Main entry-point: batch analysis + all comparison plots
├── generate_harmonic_tasksets.py   Generates harmonic task sets (tractable hyperperiods)
├── report.tex                 LaTeX source for the project report (paste into Overleaf)
├── figures/                   Generated plot PNGs (referenced by report.tex)
│   ├── schedulability_vs_utilization.png
│   ├── wcrt_u{30,50,70,90}_taskset-*.png
│   └── cdf_u{30,50,70,90}_taskset-*.png
├── Task-sets/                 Original (non-harmonic) task sets used for the report
│   ├── u30/   (20 sets, U ≈ 0.30)
│   ├── u50/   (20 sets, U ≈ 0.50)
│   ├── u70/   (20 sets, U ≈ 0.70)
│   └── u90/   (20 sets, U ≈ 0.90)
└── Task-sets-harmonic/        Harmonic task sets (tractable hyperperiods, seed=42)
    ├── u30/ u50/ u70/ u90/
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install matplotlib numpy
```

### 2. Run the full analysis (all utilisation levels)

```bash
python analysis.py
```

Processes all 80 task sets in `Task-sets/u{30,50,70,90}/`, prints a summary table, and shows comparison plots.

### 3. Save figures for the report

```bash
python analysis.py --save-plots figures/
```

### 4. Analyse a single task set

```bash
python analysis.py --csv Task-sets/u90/taskset-12.csv --verbose
```

### 5. One utilisation level only

```bash
python analysis.py --util u90
```

### 6. More simulation runs

```bash
python analysis.py --runs 1000 --save-plots figures/
```

### 7. Run on the harmonic task sets

```bash
python analysis.py --task-sets-dir Task-sets-harmonic --save-plots figures/
```

---

## Module Reference

### `Task-model.py`
Defines `Task` (BCET, WCET, Period, Deadline) and utilities:
- `load_taskset(csv_path)` — parse a CSV into a list of Task objects
- `hyperperiod(tasks)` — LCM of all task periods
- `total_utilization(tasks)` — Σ Cᵢ/Tᵢ

### `DM-logic.py`
Fixed-point Response-Time Analysis (RTA). Priority order: shortest deadline first (ties broken by shortest period). For each task τᵢ:
```
R⁽⁰⁾ = Cᵢ
R⁽ᵏ⁺¹⁾ = Cᵢ + Σⱼ∈hp(i) ⌈R⁽ᵏ⁾/Tⱼ⌉ · Cⱼ    (iterate to convergence or R > Dᵢ)
```
Output: per-task WCRT with PASS/FAIL vs deadline.

### `EDF-logic.py`
Two-part EDF analysis:

**Part A — Processor Demand Criterion (schedulability):**
```
U ≤ 1  AND  dbf(0,t) ≤ t  for all t ∈ (0, H]
dbf(0,t) = Σᵢ max(0, ⌊(t + Tᵢ − Dᵢ)/Tᵢ⌋) · Cᵢ
```

**Part B — Exact WCRT via hyperperiod simulation:**
Generates all jobs over H = lcm(T₁,…,Tₙ), simulates preemptive EDF with every job executing for exactly its WCET, records per-task max response time. Skipped when total job count exceeds 200,000 (hyperperiod explosion).

### `simulation.py`
Monte Carlo stochastic simulation. Each of `--runs` iterations draws job execution times uniformly from [BCET, WCET] and simulates preemptive EDF. Reports per-task max, mean, P95, P99 and deadline miss rate.

### `analysis.py`
Orchestrates everything. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--util u30 u50 u70 u90` | all four | Utilisation levels to process |
| `--task-sets-dir DIR` | `Task-sets` | Root folder for task set subdirs |
| `--csv file.csv …` | — | Analyse specific CSV files directly |
| `--runs N` | 300 | Monte Carlo runs per task set |
| `--save-plots DIR` | interactive | Save PNGs instead of showing them |
| `--no-plots` | off | Suppress all plot output |
| `--verbose / -v` | off | Print per-task WCRT tables |

### `generate_harmonic_tasksets.py`
Generates task sets with harmonic periods (small LCMs → tractable hyperperiods). Useful when you need exact EDF WCRTs and Monte Carlo validation for all sets.

```bash
python generate_harmonic_tasksets.py --out Task-sets-harmonic --sets-per-util 20 --tasks 10 --seed 42
```

---

## Results Summary (from the report)

Across 80 task sets (20 per utilisation level, 10 tasks each, 300 Monte Carlo runs):

| Level | Target U | DM pass rate | EDF pass rate |
|-------|----------|-------------|--------------|
| u30   | 0.30     | 20/20 (100%) | 20/20 (100%) |
| u50   | 0.50     | 20/20 (100%) | 20/20 (100%) |
| u70   | 0.70     | 20/20 (100%) | 20/20 (100%) |
| u90   | 0.90     | 18/20 (90%)  | 20/20 (100%) |
| **Total** | — | **78/80 (97.5%)** | **80/80 (100%)** |

Key findings:
- EDF is provably schedulable for any U ≤ 1; DM can fail above the hyperbolic bound (≈71.8% for n=10).
- 66/80 task sets have non-harmonic periods causing hyperperiod explosion (>200,000 jobs); exact EDF WCRTs were only computed for the remaining 14.
- In all 14 fully simulated sets: zero deadline misses, observed response times strictly below WCRT, confirming bounds are correct but pessimistic (mean ≈ 30–60% of WCRT).

---

## Task Set CSV Format

| Column | Meaning |
|--------|---------|
| `Name` | Task identifier (T0, T1, …) |
| `Jitter` | Release jitter (0 = synchronous) |
| `BCET` | Best-case execution time |
| `WCET` | Worst-case execution time (Cᵢ) |
| `Period` | Period (Tᵢ) |
| `Deadline` | Relative deadline (Dᵢ ≤ Tᵢ) |
| `PE` | Processing element (0 = single core) |

---

## References

| Topic | Reference |
|-------|-----------|
| DM / RTA | Buttazzo, *Hard Real-Time Computing Systems*, 4th ed. Springer, 2024 — §4.5, §4.5.2 |
| EDF + PDC | Buttazzo 2024 — §4.6, §4.6.1 |
| DM vs EDF | Buttazzo (2005) "Rate Monotonic vs. EDF: Judgment Day", *Real-Time Systems* 29(1) |
| EDF WCRT | Stankovic et al. (1998) "Response Times under EDF Scheduling", Springer |
| Simulation | Buttazzo 2024 — §13.5.3 |
