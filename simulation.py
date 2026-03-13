"""
Stochastic EDF Simulation for Real-Time Task Sets
==================================================

Theory
------
Analytical WCRT analysis (DM and EDF) assumes every job runs for exactly its
Worst-Case Execution Time (WCET).  In practice the actual execution time lies
somewhere in [BCET, WCET].  This module simulates the EDF schedule many times,
each time drawing job execution times randomly from a uniform distribution over
[BCET, WCET], and records the resulting response times.

Why simulate?
-------------
* Verifies analytical results: the observed response time of every job in every
  run must NEVER exceed the analytical WCRT (the WCRT is a provable upper bound).
* Reveals the typical / expected response time distribution, which is usually
  well below the WCRT.
* Demonstrates that the WCRT is tight (or nearly so) only under the specific
  worst-case load pattern.
* Satisfies the project requirement: "compare calculated WCRTs with response
  times observed during simulations".

Simulation procedure (per run):
    1. For each job τᵢ,k released in [0, H):
           execution_time ~ Uniform(BCET_i, WCET_i)
    2. Simulate preemptive EDF over [0, H) with those execution times.
    3. Record response time  R_{i,k} = finish_{i,k} − release_{i,k}.

Statistical outputs:
    - Per-task observed WCRT (max over all runs and all jobs of that task)
    - Per-task mean response time
    - Per-task 95th and 99th percentile response times
    - Deadline miss rate (should be 0 if analytical WCRT ≤ deadline)

How to run
----------
Single task set with default 500 runs:
    python simulation.py Task-sets/u70/taskset-0.csv

Override number of runs:
    python simulation.py Task-sets/u70/taskset-0.csv --runs 200

Show a histogram plot:
    python simulation.py Task-sets/u70/taskset-0.csv --plot

Use analysis.py for full comparison plots across all utilisation levels.

Dependencies
------------
* EDF-logic.py  (the preemptive EDF simulation engine)
* Task-model.py (Task class and CSV loader)
* Both loaded via importlib (hyphen in filenames prevents normal import).
* numpy and matplotlib are required for statistics and plots.
"""

import importlib.util
import os
import sys
import random
import argparse


# ---------------------------------------------------------------------------
# Load sibling modules via importlib (hyphen filenames)
# ---------------------------------------------------------------------------

def _load_module(name, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tm  = _load_module("task_model", "Task-model.py")
_edf = _load_module("edf_logic",  "EDF-logic.py")

Task              = _tm.Task
load_taskset      = _tm.load_taskset
hyperperiod       = _tm.hyperperiod
total_utilization = _tm.total_utilization

_run_edf_simulation = _edf._run_edf_simulation


# ---------------------------------------------------------------------------
# Tuneable defaults
# ---------------------------------------------------------------------------

DEFAULT_RUNS       = 500    # number of Monte-Carlo iterations per task set
MAX_JOBS_PER_RUN   = 200_000     # skip if a single run would process more jobs


# ---------------------------------------------------------------------------
# Core stochastic simulation
# ---------------------------------------------------------------------------

def run_stochastic_simulation(tasks, n_runs=DEFAULT_RUNS, seed=None):
    """
    Run the EDF scheduler n_runs times with randomised execution times.

    Each run samples execution times uniformly from [BCET, WCET] per job.
    Response times for every job instance in every run are accumulated for
    statistical analysis.

    Parameters
    ----------
    tasks  : list[Task]
    n_runs : int          — number of independent Monte-Carlo runs
    seed   : int or None  — RNG seed for reproducibility

    Returns
    -------
    results : dict
        task_id → {
            'all_rts'        : list[float]  — all response times across all runs
            'max_rt'         : float        — observed WCRT (max over all runs)
            'mean_rt'        : float
            'p95_rt'         : float        — 95th percentile
            'p99_rt'         : float        — 99th percentile
            'n_samples'      : int          — total job instances observed
        }
    miss_rate : float
        Fraction of runs in which at least one deadline was missed.
    skipped : bool
        True if the simulation was skipped due to excessive hyperperiod size.

    Notes
    -----
    If BCET == WCET for all tasks, execution times are deterministic and every
    run produces identical results (which should match the analytical WCRT).
    """
    H      = hyperperiod(tasks)
    n_jobs = sum(H // t.period for t in tasks)

    if n_jobs > MAX_JOBS_PER_RUN:
        print(f"  [Simulation] Skipping: {n_jobs:,} jobs/run exceeds limit {MAX_JOBS_PER_RUN:,}")
        return None, 0.0, True

    rng = random.Random(seed)

    # Accumulate response times per task
    all_rts = {t.task_id: [] for t in tasks}
    n_misses = 0

    for _ in range(n_runs):
        jobs, missed = _run_edf_simulation(tasks, H, random_et=True, rng=rng)
        if missed:
            n_misses += 1
        for job in jobs:
            if job['finish'] is not None:
                rt = job['finish'] - job['release']
                all_rts[job['task_id']].append(rt)

    # Compute per-task statistics
    results = {}
    for task in tasks:
        tid = task.task_id
        rts = all_rts[tid]
        if not rts:
            results[tid] = {
                'all_rts': [], 'max_rt': 0, 'mean_rt': 0,
                'p95_rt': 0, 'p99_rt': 0, 'n_samples': 0,
            }
            continue
        rts_sorted = sorted(rts)
        n          = len(rts_sorted)
        p95_idx    = min(int(0.95 * n), n - 1)
        p99_idx    = min(int(0.99 * n), n - 1)
        results[tid] = {
            'all_rts'  : rts_sorted,
            'max_rt'   : rts_sorted[-1],
            'mean_rt'  : sum(rts_sorted) / n,
            'p95_rt'   : rts_sorted[p95_idx],
            'p99_rt'   : rts_sorted[p99_idx],
            'n_samples': n,
        }

    miss_rate = n_misses / n_runs if n_runs > 0 else 0.0
    return results, miss_rate, False


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def print_simulation_results(sim_results, analytical_wcrt, tasks, miss_rate,
                              csv_path=""):
    """
    Print a table comparing simulated statistics with analytical WCRTs.

    Parameters
    ----------
    sim_results      : dict (from run_stochastic_simulation)
    analytical_wcrt  : dict (from compute_dm_wcrt or compute_edf_wcrt), or None
    tasks            : list[Task]
    miss_rate        : float
    csv_path         : str
    """
    if sim_results is None:
        print("  [Simulation] No results available.")
        return

    label = os.path.basename(csv_path) if csv_path else "task set"
    print(f"\n{'='*76}")
    print(f"  Stochastic Simulation — {label}")
    print(f"  Deadline miss rate: {miss_rate*100:.2f}%")
    print(f"{'='*76}")
    hdr = f"{'Task':<8} {'Analyt.WCRT':>11} {'Obs.max':>9} {'Mean':>9} "
    hdr += f"{'P95':>9} {'P99':>9}  Bound OK?"
    print(hdr)
    print("-" * 76)

    for task in tasks:
        tid = task.task_id
        s   = sim_results.get(tid, {})
        if not s or s['n_samples'] == 0:
            print(f"{tid:<8}  (no data)")
            continue

        an_wcrt = analytical_wcrt[tid]['wcrt'] if analytical_wcrt and tid in analytical_wcrt else None
        obs_max = s['max_rt']
        bound_ok = "YES" if an_wcrt is None or obs_max <= an_wcrt else "NO ← violated!"
        an_str = f"{an_wcrt:>11}" if an_wcrt is not None else f"{'N/A':>11}"
        print(f"{tid:<8} {an_str} {obs_max:>9.1f} {s['mean_rt']:>9.1f} "
              f"{s['p95_rt']:>9.1f} {s['p99_rt']:>9.1f}  {bound_ok}")

    print(f"{'='*76}")


# ---------------------------------------------------------------------------
# Histogram plot (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_response_time_distribution(sim_results, analytical_wcrt, tasks,
                                    title="Response Time Distribution",
                                    save_path=None):
    """
    Plot per-task histograms of observed response times vs the analytical WCRT.

    The analytical WCRT (vertical red line) should always be to the right of
    every observed bar — if any bar exceeds it, the bound is violated.

    Parameters
    ----------
    sim_results     : dict (from run_stochastic_simulation)
    analytical_wcrt : dict or None
    tasks           : list[Task]
    title           : str
    save_path       : str or None — if given, saves the figure to disk
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [plot] matplotlib not installed — skipping plot.")
        return

    n_tasks = len(tasks)
    cols    = min(3, n_tasks)
    rows    = math.ceil(n_tasks / cols) if n_tasks > 0 else 1

    import math as _math

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    fig.suptitle(title, fontsize=13, y=1.01)

    # Flatten axes for easy indexing
    if n_tasks == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, task in enumerate(tasks):
        ax  = axes[idx]
        tid = task.task_id
        s   = sim_results.get(tid, {})

        if not s or s['n_samples'] == 0:
            ax.set_title(f"{tid} — no data")
            ax.axis('off')
            continue

        rts = s['all_rts']
        ax.hist(rts, bins=40, color='steelblue', edgecolor='white',
                alpha=0.8, density=True)

        if analytical_wcrt and tid in analytical_wcrt:
            wcrt = analytical_wcrt[tid]['wcrt']
            ax.axvline(wcrt, color='red', linewidth=1.5,
                       linestyle='--', label=f"WCRT={wcrt}")
            ax.legend(fontsize=8)

        ax.axvline(s['max_rt'], color='orange', linewidth=1.2,
                   linestyle=':', label=f"obs.max={s['max_rt']:.0f}")
        ax.set_title(f"{tid}  (D={task.deadline})", fontsize=10)
        ax.set_xlabel("Response time", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(tasks), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stochastic EDF simulation for real-time task sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulation.py Task-sets/u70/taskset-0.csv
  python simulation.py Task-sets/u90/taskset-3.csv --runs 200 --plot
  python simulation.py Task-sets/u90/taskset-3.csv --runs 1000 --seed 42
""",
    )
    parser.add_argument("csv", nargs="+", help="Task set CSV file(s)")
    parser.add_argument("--runs",  type=int, default=DEFAULT_RUNS,
                        help=f"Number of simulation runs (default {DEFAULT_RUNS})")
    parser.add_argument("--seed",  type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--plot",  action="store_true",
                        help="Show response-time histogram plots")
    args = parser.parse_args()

    for csv_path in args.csv:
        try:
            tasks = load_taskset(csv_path)
        except FileNotFoundError:
            print(f"ERROR: file not found: {csv_path}")
            continue

        if not tasks:
            print(f"WARNING: no tasks loaded from {csv_path}")
            continue

        print(f"\nRunning {args.runs} simulation(s) for: {csv_path}")
        print(f"  U = {total_utilization(tasks):.4f}  |  "
              f"tasks = {len(tasks)}  |  H = {hyperperiod(tasks)}")

        # Also compute analytical EDF WCRT for comparison
        try:
            an_wcrt = _edf.compute_edf_wcrt(tasks)
        except Exception:
            an_wcrt = None

        sim_results, miss_rate, skipped = run_stochastic_simulation(
            tasks, n_runs=args.runs, seed=args.seed
        )

        if skipped:
            print("  Simulation skipped (hyperperiod too large).")
            continue

        print_simulation_results(sim_results, an_wcrt, tasks, miss_rate,
                                  csv_path=csv_path)

        if args.plot:
            plot_response_time_distribution(
                sim_results, an_wcrt, tasks,
                title=f"EDF Response Times — {os.path.basename(csv_path)}"
            )
