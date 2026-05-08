## How to Generate Task Sets for 02225 DRTS Mini-project 1

### 1. Setup

```bash
pip install docopt scipy numpy csv graphviz
```

---

### 2. Generating Task Sets (Single Core, DM/EDF Compatible)

Since the project is **single-core** with **deadlines ≤ periods**, use these flags:

```bash
# Generate 10 task sets, 5 tasks each, 60% utilization, UUniFast algorithm, single core
python task_generator.py -n 5 -s 10 -u 60 -g 1 -p 1 --npe 1 -r
```

| Flag | Meaning | Recommended |
|------|---------|-------------|
| `-n N` | Tasks per set | 5–10 |
| `-s N` | Number of sets | 10–50 |
| `-u N` | Total utilization (%) | Try 30, 50, 70, 90 |
| `-g N` | Generator: `1`=UUniFast, `2`=Emberson | `1` (UUniFast gives exact utilization) |
| `--npe 1` | **Single core** | **Required for your project** |
| `-r` | Round WCET/BCET to integers | Recommended |

> **Important:** By default, `Deadline = Period`, so the constraint Di ≤ Ti is already satisfied.

The output CSV format is:
```
Name, Jitter, BCET, WCET, Period, Deadline, PE
T0,   0,      31,   44,   1000,   1000,     0
T1,   0,      87,   123,  2000,   2000,     0
...
```

---

### 3. Generating Jobsets (with DM or EDF Priorities)

Once you have `taskset-0.csv`:

```bash
# DM priority assignment (for DM analysis)
python priority_generator.py -t taskset-0.csv -m 1

# EDF priority assignment (for EDF jobset)
python priority_generator.py -t taskset-0.csv -m 2
```

This expands tasks over the full **hyperperiod** and assigns priorities. The output `jobset-taskset-0.csv` has:
```
Name, Arrival min., Arrival max., BCET, WCET, Abs. deadline, PE, Priority
T0,0, 0,  0, 31, 44, 1000, 0, 3
T0,1, 1000, 1000, 31, 44, 2000, 0, 7
...
```

---

### 4. What You Need to Implement Yourself

The tools give you **task sets** and **job lists**. You still need to write the analysis yourself:

#### A) DM Worst-Case Response Time (Buttazzo §4.5.2)

Iterative fixed-point calculation for each task τᵢ (sorted by deadline):

```
R_i^(0) = C_i
R_i^(k+1) = C_i + Σ_{j∈hp(i)} ⌈R_i^(k) / T_j⌉ · C_j

Stop when R_i^(k+1) == R_i^(k)  →  WCRT_i = R_i
If R_i > D_i  →  task is NOT schedulable
```

where `hp(i)` = set of tasks with higher priority than τᵢ (shorter deadline under DM).

#### B) EDF WCRT — Analytical Hyperperiod Method (from your assignment doc)

```
1. H = lcm(T1, T2, ..., Tn)
2. Generate all jobs: τᵢ releases job k at time r_{i,k} = k·Tᵢ, for k·Tᵢ < H
3. Simulate EDF from t=0 to t=H:
   - Every job executes exactly WCET (Cᵢ)
   - At each instant, schedule the ready job with earliest absolute deadline d_{i,k} = r_{i,k} + Dᵢ
4. Record finish time f_{i,k} for each job
5. R_{i,k} = f_{i,k} − r_{i,k}
6. WCRT_i = max_k { R_{i,k} }
```

#### C) EDF Schedulability Check (Processor Demand, Buttazzo §4.6.1)

For constrained deadlines (Di ≤ Ti), check:

```
U = Σ Cᵢ/Tᵢ ≤ 1    (necessary condition)

For all scheduling points t in [0, H]:
  dbf(0,t) = Σᵢ ⌊(t + Tᵢ − Dᵢ) / Tᵢ⌋ · Cᵢ  ≤ t
```

---

### 5. Comparison Strategy

**DM vs EDF:**
- Generate task sets at increasing utilizations (e.g., 50%, 60%, 70%, 80%, 90%, 95%)
- For each set, check schedulability under both algorithms
- Plot: % of schedulable sets vs utilization → EDF should dominate at high utilization
- Compare WCRTs: under DM, high-priority tasks get short WCRTs; under EDF the distribution differs

**Analytical vs Simulation:**
- In your simulator, draw each job's execution time uniformly from `[BCET, WCET]`
- Run the simulation many times (e.g., 1000 iterations × hyperperiod)
- Plot the distribution of observed response times vs the analytical WCRT
- The WCRT should be an **upper bound** — observed RTs should never exceed it

---

### 6. Batch Generation (Multiple Utilizations)

```bash
for u in 30 50 70 80 90; do
  python task_generator.py -n 8 -s 50 -u $u -g 1 --npe 1 -r
  # move/rename outputs for each utilization level
done
```

Or use the provided `group_generator.sh`:

```bash
bash group_generator.sh -s 50 -n 8 -u 70 -g 1 -p 1 -c 1
```

This auto-generates tasksets, jobsets (DM), and segment sets into organized subdirectories.

---

### 7. Task Set CSV Format Quick Reference

| Column | Meaning |
|--------|---------|
| `Name` | Task ID (T0, T1, ...) |
| `Jitter` | Release jitter (0 = synchronous release) |
| `BCET` | Best-case execution time |
| `WCET` | Worst-case execution time Cᵢ |
| `Period` | Task period Tᵢ |
| `Deadline` | Relative deadline Dᵢ (= Tᵢ by default) |
| `PE` | Processing element (0 for single core) |

For your analysis:
- Use **WCET** as `Cᵢ` for analytical WCRTs
- Use random value in **[BCET, WCET]** for simulation

---

### 8. Recommended Test Cases

| Scenario | Purpose |
|----------|---------|
| 3–5 tasks, U ≈ 0.7, simple periods (e.g., 2,4,8) | Easy hand-verification |
| Task set where DM fails but EDF succeeds | Demonstrates EDF superiority |
| U = 1.0 exactly | Boundary case |
| U > 1.0 | Both should fail |
| Harmonic periods (T, 2T, 4T) | Known analytical results |
