# 02225 DRTS Mini-Project 1 — Real-Time Schedulability Analysis

Worst-Case Response Time (WCRT) analysis and comparison of **Deadline Monotonic
(DM)** and **Earliest Deadline First (EDF)** scheduling algorithms for single-core
periodic real-time task sets.

---

## Repository Structure

```
02225-DRTS-Mini-Project-1/
├── Task-model.py       Core data structures: Task class, CSV loader, helpers
├── DM-logic.py         DM WCRT analysis (fixed-point RTA, Buttazzo §4.5.2)
├── EDF-logic.py        EDF schedulability (processor demand) + WCRT simulation
├── simulation.py       Stochastic EDF simulation (random ET ∈ [BCET, WCET])
├── analysis.py         Main entry-point: batch analysis + comparison plots
├── GUIDE.MD            Step-by-step usage guide for task set generation
└── Task-sets/
    ├── taskset-0.csv   Sample task set (task_generator default output)
    ├── taskset-1.csv   Border set: U≈0.85, DM likely fails, EDF passes
    ├── task-set-example.csv
    ├── u30/            20 task sets at ~30% CPU utilisation
    ├── u50/            20 task sets at ~50% CPU utilisation
    ├── u70/            20 task sets at ~70% CPU utilisation
    └── u90/            20 task sets at ~90% CPU utilisation
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

This processes every task set in `Task-sets/u30`, `u50`, `u70`, `u90`,
prints a summary table, and displays four types of comparison plots.

### 3. Save plots to a directory

```bash
python analysis.py --save-plots plots/
```

### 4. Analyse a single CSV file

```bash
python analysis.py --csv Task-sets/taskset-1.csv
```

### 5. One utilisation level only

```bash
python analysis.py --util u90
```

### 6. More simulation runs (higher statistical confidence)

```bash
python analysis.py --runs 1000 --save-plots plots/
```

---

## Module Reference

### `Task-model.py` — Core data model

```
python Task-model.py <taskset.csv>
```

Defines `Task` (BCET, WCET, Period, Deadline, …) and utility functions:
- `load_taskset(csv_path)` — read a CSV into a list of Task objects
- `hyperperiod(tasks)` — LCM of all task periods
- `total_utilization(tasks)` — Σ Cᵢ/Tᵢ

---

### `DM-logic.py` — Deadline Monotonic WCRT Analysis

```
python DM-logic.py Task-sets/u70/taskset-0.csv
python DM-logic.py Task-sets/u90/*.csv
```

**Algorithm (Buttazzo §4.5.2 — Response-Time Analysis):**

Priority assignment: shorter deadline = higher priority.

For each task τᵢ in priority order:

```
R⁽⁰⁾ = Cᵢ
R⁽ᵏ⁺¹⁾ = Cᵢ + Σⱼ∈hp(i) ⌈R⁽ᵏ⁾/Tⱼ⌉ · Cⱼ
```

Iterate until convergence.  Task is **not schedulable** if R > Dᵢ.

**Why DM?**
- Optimal among all fixed-priority algorithms for constrained-deadline tasks.
- Simple, static priorities — good for low to medium utilisation.
- Fails at high utilisation (>≈0.85) where EDF still succeeds.

**Output:** Per-task WCRT table with PASS/FAIL status.

---

### `EDF-logic.py` — EDF Schedulability + Analytical WCRT

```
python EDF-logic.py Task-sets/u70/taskset-0.csv
python EDF-logic.py Task-sets/u90/*.csv
```

**Part A — Processor Demand Criterion (Buttazzo §4.6.1):**

Necessary condition: U = Σ Cᵢ/Tᵢ ≤ 1

Sufficient + necessary: for all scheduling points t ∈ (0, H]:
```
dbf(0,t) = Σᵢ max(0, ⌊(t + Tᵢ − Dᵢ)/Tᵢ⌋) · Cᵢ  ≤  t
```

**Part B — Analytical WCRT via hyperperiod simulation:**

1. Compute H = lcm(T₁,…,Tₙ)
2. Generate all jobs in [0, H): release rᵢ,ₖ = k·Tᵢ, deadline dᵢ,ₖ = rᵢ,ₖ + Dᵢ
3. Simulate preemptive EDF with all jobs executing for exactly WCET
4. WCRTᵢ = max over all jobs of τᵢ of (finish − release)

**Why EDF?**
- Theoretically optimal: schedulable iff U ≤ 1.
- Dominates DM at high utilisation (see "Judgment Day" paper).
- Trade-off: dynamic priorities require runtime scheduling decisions.

**Output:** Schedulability verdict + per-task WCRT table.

---

### `simulation.py` — Stochastic EDF Simulation

```
python simulation.py Task-sets/u70/taskset-0.csv
python simulation.py Task-sets/u90/taskset-3.csv --runs 500 --plot
python simulation.py Task-sets/u70/taskset-0.csv --runs 1000 --seed 42
```

**What it does:**
- Runs the EDF schedule `--runs` times (default 500).
- Each run: draws each job's execution time uniformly from [BCET, WCET].
- Collects observed response times across all runs.
- Reports: max, mean, P95, P99 per task vs the analytical EDF WCRT.

**What to look for:**
- The analytical WCRT must always ≥ observed maximum (it is a proven bound).
- The typical response time (mean/P95) is usually well below the WCRT.
- A non-zero deadline miss rate indicates a schedulability problem.

**Output:** Per-task statistics table + optional histogram plots.

---

### `analysis.py` — Full Batch Analysis

The main script.  Ties everything together:

| Flag | Default | Description |
|------|---------|-------------|
| `--util u30 u50 u70 u90` | all four | Utilisation levels to process |
| `--csv file.csv …` | — | Specific CSV file(s) instead of folders |
| `--runs N` | 300 | Stochastic simulation runs per task set |
| `--save-plots DIR` | interactive | Save plots as PNGs |
| `--no-plots` | off | Suppress all matplotlib output |
| `--verbose` / `-v` | off | Print per-task WCRT tables |

**Plots produced:**
1. **Schedulability rate vs utilisation** — bar chart comparing DM and EDF pass rates.
2. **WCRT comparison** — grouped bar chart, DM vs EDF per task, with deadline line.
3. **Analytical vs simulation CDF** — per-task empirical CDF vs the WCRT bound.

---

## Interpreting Results

### DM vs EDF schedulability

```
u30: DM=100%  EDF=100%   ← low utilisation, both trivially pass
u50: DM= 95%  EDF=100%
u70: DM= 75%  EDF=100%   ← DM starts failing
u90: DM= 30%  EDF= 95%   ← DM fails most sets; EDF still passes
```

### WCRT comparison

Under DM:
- High-priority tasks (short deadline) → very short WCRT (little interference).
- Low-priority tasks → large WCRT due to accumulated preemption by all higher-priority tasks.

Under EDF:
- WCRTs are more evenly distributed: any task can be preempted when a
  task with an earlier absolute deadline arrives, regardless of static rank.
- Generally, EDF WCRTs for low-priority tasks are shorter than DM WCRTs.

### Simulation vs analytical bound

- The simulated CDF line should always end before (to the left of) the red WCRT line.
- A large gap between the mean/P99 line and the WCRT shows the bound is pessimistic
  (WCRT requires a specific worst-case phase alignment unlikely in practice).

---

## Task Set CSV Format

| Column | Meaning |
|--------|---------|
| `Name` | Task identifier (T0, T1, …) |
| `Jitter` | Release jitter (0 = synchronous, used for generation only) |
| `BCET` | Best-case execution time |
| `WCET` | Worst-case execution time  Cᵢ |
| `Period` | Period  Tᵢ |
| `Deadline` | Relative deadline  Dᵢ  (≤ Tᵢ for constrained deadlines) |
| `PE` | Processing element (0 = single core) |

---

## Theoretical Background

| Topic | Reference |
|-------|-----------|
| DM Scheduling | Buttazzo §4.5, §4.5.2 |
| EDF with Constrained Deadlines | Buttazzo §4.6, §4.6.1 |
| DM vs EDF comparison | Buttazzo (2005) "Rate Monotonic vs. EDF: Judgment Day" |
| EDF WCRT for arbitrary deadlines | Stankovic et al. (1998) ch. "Response Times under EDF" |
| Simulation methodology | Buttazzo §13.5.3 |
