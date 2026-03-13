"""
02225 DRTS Mini-Project 1 — Main Analysis Script
=================================================

Overview
--------
This is the single entry-point for the full schedulability and WCRT analysis
required by the mini-project.  It:

  1. Loads every task set under Task-sets/{u30,u50,u70,u90}/ (or a custom path).
  2. Runs DM Worst-Case Response Time analysis (Buttazzo §4.5.2).
  3. Runs EDF Processor Demand schedulability check (Buttazzo §4.6.1).
  4. Runs EDF analytical WCRT simulation over the hyperperiod.
  5. Runs a stochastic EDF simulation (random ET ∈ [BCET, WCET]) to compare
     observed response times against the analytical upper bounds.
  6. Produces four types of comparison plots:
       (a) Schedulability rate vs utilisation — DM vs EDF
       (b) WCRT comparison: DM vs EDF for each task in a set
       (c) Analytical vs simulated response time CDFs
       (d) WCRT box-plots across all task sets at each utilisation level

Why this structure?
-------------------
The three-module design mirrors the project's analytical decomposition:
  * DM-logic.py   — fixed-priority (offline) analysis
  * EDF-logic.py  — dynamic-priority (online) analysis
  * simulation.py — stochastic validation

analysis.py orchestrates these modules, handles I/O (CSV loading, plot saving)
and provides command-line controls so the analysis is reproducible.

How to run
----------
Run everything with defaults (processes all u30/u50/u70/u90 sets, 300 sim runs):
    python analysis.py

Target a single utilisation level:
    python analysis.py --util u70

Analyse a specific CSV file:
    python analysis.py --csv Task-sets/taskset-1.csv

Override simulation run count:
    python analysis.py --runs 1000

Save plots to a directory instead of showing interactively:
    python analysis.py --save-plots plots/

Suppress plots entirely:
    python analysis.py --no-plots

Analyse a custom CSV:
    python analysis.py --csv my_taskset.csv --runs 500 --save-plots out/

Interpreting the results
------------------------
DM WCRT table:
  * Lower-priority tasks (longer deadlines) accumulate more interference and
    typically have larger WCRTs.
  * FAIL means the task's WCRT exceeds its deadline — the set is unschedulable
    under DM.

EDF WCRT table:
  * WCRTs are generally more evenly distributed because EDF adapts dynamically.
  * EDF should PASS all sets where U ≤ 1.

Schedulability rate plot:
  * At low U both algorithms pass most sets.
  * As U rises, DM failure rate increases first.
  * EDF remains schedulable right up to U = 1.

Simulation vs analytical:
  * All bars in the histogram must lie to the LEFT of the red WCRT line.
  * The mean and percentile lines (P95, P99) show the typical operating range.
  * A large gap between mean and WCRT confirms the WCRT is a pessimistic bound.

Dependencies
------------
Standard library:  os, sys, argparse, glob, importlib
Third-party:       matplotlib, numpy (both installed via pip)
Sibling modules:   Task-model.py, DM-logic.py, EDF-logic.py, simulation.py
"""

import os
import sys
import glob
import argparse
import importlib.util

# ---------------------------------------------------------------------------
# Load sibling modules (hyphen filenames need importlib)
# ---------------------------------------------------------------------------

def _load_module(name, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tm  = _load_module("task_model",  "Task-model.py")
_dm  = _load_module("dm_logic",    "DM-logic.py")
_edf = _load_module("edf_logic",   "EDF-logic.py")
_sim = _load_module("simulation",  "simulation.py")

load_taskset            = _tm.load_taskset
print_taskset           = _tm.print_taskset
hyperperiod             = _tm.hyperperiod
total_utilization       = _tm.total_utilization

compute_dm_wcrt         = _dm.compute_dm_wcrt
is_schedulable_dm       = _dm.is_schedulable_dm

edf_schedulability_check = _edf.edf_schedulability_check
compute_edf_wcrt        = _edf.compute_edf_wcrt
is_schedulable_edf      = _edf.is_schedulable_edf

run_stochastic_simulation = _sim.run_stochastic_simulation
plot_response_time_distribution = _sim.plot_response_time_distribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTIL_FOLDERS = ["u30", "u50", "u70", "u90"]
TASK_SETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Task-sets")


def _csvs_in(directory):
    """Return sorted list of CSV paths in a directory."""
    return sorted(glob.glob(os.path.join(directory, "*.csv")))


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _safe_load(csv_path):
    """Load a task set, returning None and a warning on failure."""
    try:
        tasks = load_taskset(csv_path)
        if not tasks:
            print(f"  WARNING: no tasks in {csv_path}")
            return None
        return tasks
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-set analysis
# ---------------------------------------------------------------------------

def analyse_one(tasks, csv_path, n_runs, verbose=False):
    """
    Run DM + EDF + simulation on a single task set.

    Returns
    -------
    dict with keys:
        csv_path, utilization,
        dm_schedulable, edf_schedulable,
        dm_results, edf_results,
        sim_results, miss_rate
    """
    U = total_utilization(tasks)
    label = os.path.basename(csv_path)

    if verbose:
        print_taskset(tasks, title=f"\n  {label}  (U={U:.3f})")

    # DM analysis
    dm_results = compute_dm_wcrt(tasks)
    dm_ok      = all(r['schedulable'] for r in dm_results.values())

    # EDF schedulability check
    edf_ok, edf_details = edf_schedulability_check(tasks)

    # EDF WCRT (analytical simulation)
    edf_results = compute_edf_wcrt(tasks)

    # Stochastic simulation
    sim_results, miss_rate, skipped = run_stochastic_simulation(
        tasks, n_runs=n_runs, seed=0
    )
    if skipped:
        sim_results = None
        miss_rate   = float('nan')

    return {
        'csv_path'       : csv_path,
        'label'          : label,
        'utilization'    : U,
        'dm_schedulable' : dm_ok,
        'edf_schedulable': edf_ok,
        'dm_results'     : dm_results,
        'edf_results'    : edf_results,
        'sim_results'    : sim_results,
        'miss_rate'      : miss_rate,
        'tasks'          : tasks,
    }


# ---------------------------------------------------------------------------
# Batch analysis across utilisation levels
# ---------------------------------------------------------------------------

def batch_analysis(util_dirs, n_runs, verbose=False):
    """
    Run the full analysis for every task set in each utilisation directory.

    Parameters
    ----------
    util_dirs : list[str]   — paths to utilisation directories
    n_runs    : int         — simulation runs per task set

    Returns
    -------
    list[dict]   — one entry per task set (output of analyse_one)
    """
    all_results = []

    for d in util_dirs:
        csvs = _csvs_in(d)
        if not csvs:
            print(f"  [batch] No CSVs found in {d}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Processing: {d}  ({len(csvs)} task sets)")
        print(f"{'─'*60}")

        for csv_path in csvs:
            tasks = _safe_load(csv_path)
            if tasks is None:
                continue
            res = analyse_one(tasks, csv_path, n_runs=n_runs, verbose=verbose)
            all_results.append(res)

            status = ("DM=PASS" if res['dm_schedulable'] else "DM=FAIL",
                      "EDF=PASS" if res['edf_schedulable'] else "EDF=FAIL")
            print(f"  {res['label']:<30}  U={res['utilization']:.3f}  "
                  f"{status[0]}  {status[1]}")

    return all_results


# ---------------------------------------------------------------------------
# Plot 1 — Schedulability rate vs utilisation
# ---------------------------------------------------------------------------

def plot_schedulability_vs_utilization(all_results, save_path=None):
    """
    Bar chart: percentage of task sets that pass DM / EDF at each utilisation.

    This plot directly demonstrates EDF's superiority at high utilisation.
    Under DM, once the set's utilisation approaches the hyperbolic bound
    (n(2^(1/n)−1)), many sets start failing.  EDF remains schedulable as long
    as U ≤ 1.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib/numpy not installed.")
        return

    # Group by utilisation folder name
    groups = {}
    for r in all_results:
        # Infer group from path  (…/u70/taskset-0.csv → "u70")
        parts = r['csv_path'].replace("\\", "/").split("/")
        grp   = next((p for p in parts if p.startswith("u") and p[1:].isdigit()), "other")
        groups.setdefault(grp, []).append(r)

    sorted_keys = sorted(groups.keys(), key=lambda k: int(k[1:]) if k[1:].isdigit() else 0)

    labels    = []
    dm_rates  = []
    edf_rates = []

    for grp in sorted_keys:
        recs = groups[grp]
        n    = len(recs)
        dm_rate  = sum(1 for r in recs if r['dm_schedulable'])  / n * 100
        edf_rate = sum(1 for r in recs if r['edf_schedulable']) / n * 100
        labels.append(grp)
        dm_rates.append(dm_rate)
        edf_rates.append(edf_rate)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_dm  = ax.bar(x - width/2, dm_rates,  width, label='DM',  color='steelblue',  alpha=0.85)
    bars_edf = ax.bar(x + width/2, edf_rates, width, label='EDF', color='darkorange', alpha=0.85)

    ax.set_xlabel("Utilisation level", fontsize=12)
    ax.set_ylabel("Schedulable task sets (%)", fontsize=12)
    ax.set_title("DM vs EDF — Schedulability Rate by Utilisation", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.axhline(100, linestyle='--', linewidth=0.8, color='gray')
    ax.legend(fontsize=11)

    # Annotate bars
    for bar in list(bars_dm) + list(bars_edf):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — WCRT comparison: DM vs EDF for a specific task set
# ---------------------------------------------------------------------------

def plot_wcrt_comparison(res, save_path=None):
    """
    Grouped bar chart comparing DM and EDF WCRTs per task, with deadline shown.

    Helps visualise:
    * Which tasks benefit most from EDF (shorter WCRT under EDF).
    * Which tasks miss their deadline under DM but not EDF.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib/numpy not installed.")
        return

    tasks = res['tasks']
    dm_r  = res['dm_results']
    edf_r = res['edf_results']

    if not dm_r or not edf_r:
        print("  [plot] Insufficient results for WCRT comparison.")
        return

    # Sort tasks by DM priority (shortest deadline first)
    sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
    tids = [t.task_id for t in sorted_tasks]

    dm_wcrts  = [dm_r.get(tid,  {}).get('wcrt',  0) for tid in tids]
    edf_wcrts = [(edf_r.get(tid, {}).get('wcrt', 0)
                  if edf_r else 0) for tid in tids]
    deadlines = [t.deadline for t in sorted_tasks]

    x     = np.arange(len(tids))
    width = 0.30

    fig, ax = plt.subplots(figsize=(max(8, len(tids) * 1.1), 5))
    ax.bar(x - width/2, dm_wcrts,  width, label='DM WCRT',  color='steelblue',  alpha=0.85)
    ax.bar(x + width/2, edf_wcrts, width, label='EDF WCRT', color='darkorange', alpha=0.85)
    ax.step(np.append(x - 0.5, x[-1] + 0.5),
            deadlines + [deadlines[-1]],
            where='post', linestyle='--', color='red', linewidth=1.5,
            label='Deadline')

    ax.set_xticks(x)
    ax.set_xticklabels(tids, rotation=45, ha='right')
    ax.set_xlabel("Task (sorted by deadline)", fontsize=11)
    ax.set_ylabel("Response time", fontsize=11)
    ax.set_title(f"WCRT Comparison: DM vs EDF — {res['label']}\n"
                 f"U={res['utilization']:.3f}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — Analytical WCRT vs simulation CDF
# ---------------------------------------------------------------------------

def plot_analytical_vs_simulation(res, save_path=None):
    """
    For each task: empirical CDF of observed response times with vertical
    lines at P95, P99, observed max, and analytical EDF WCRT.

    If the analytical WCRT line is always to the right of the CDF, the bound
    is confirmed.  A gap between mean and WCRT shows WCRT pessimism.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib/numpy not installed.")
        return

    sim_r = res['sim_results']
    edf_r = res['edf_results']
    tasks = res['tasks']

    if sim_r is None:
        print("  [plot] No simulation results — skipping CDF plot.")
        return

    n     = len(tasks)
    cols  = min(3, n)
    rows  = int(np.ceil(n / cols)) if n > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    fig.suptitle(
        f"Analytical EDF WCRT vs Simulated CDF — {res['label']}\n"
        f"U={res['utilization']:.3f}  miss_rate={res['miss_rate']*100:.1f}%",
        fontsize=11
    )

    flat_axes = []
    if n == 1:
        flat_axes = [axes]
    elif rows == 1:
        flat_axes = list(axes)
    else:
        flat_axes = [ax for row in axes for ax in row]

    sorted_tasks = sorted(tasks, key=lambda t: t.deadline)

    for idx, task in enumerate(sorted_tasks):
        ax  = flat_axes[idx]
        tid = task.task_id
        s   = sim_r.get(tid, {})

        if not s or s['n_samples'] == 0:
            ax.set_title(f"{tid} — no data")
            ax.axis('off')
            continue

        rts = np.array(s['all_rts'])
        rts_sorted = np.sort(rts)
        cdf = np.arange(1, len(rts_sorted) + 1) / len(rts_sorted)

        ax.plot(rts_sorted, cdf, color='steelblue', linewidth=1.4, label='Simulated CDF')

        # Analytical EDF WCRT
        if edf_r and tid in edf_r:
            wcrt = edf_r[tid]['wcrt']
            ax.axvline(wcrt, color='red', linewidth=1.5, linestyle='--',
                       label=f'EDF WCRT={wcrt}')

        # Deadline
        ax.axvline(task.deadline, color='black', linewidth=1.0, linestyle=':',
                   label=f'Deadline={task.deadline}')

        # Mean
        ax.axvline(s['mean_rt'], color='green', linewidth=1.0, linestyle='-.',
                   label=f"Mean={s['mean_rt']:.0f}")

        ax.set_title(f"{tid}  (D={task.deadline})", fontsize=9)
        ax.set_xlabel("Response time", fontsize=8)
        ax.set_ylabel("CDF", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)

    for idx in range(len(sorted_tasks), len(flat_axes)):
        flat_axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table across all task sets
# ---------------------------------------------------------------------------

def print_summary_table(all_results):
    """Print a concise summary of all analysed task sets."""
    print(f"\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'File':<32} {'U':>6}  {'DM':>6}  {'EDF':>6}  {'Miss%':>6}")
    print("-" * 80)

    for r in all_results:
        dm_s  = "PASS" if r['dm_schedulable']  else "FAIL"
        edf_s = "PASS" if r['edf_schedulable'] else "FAIL"
        miss  = f"{r['miss_rate']*100:.1f}%" if r['miss_rate'] == r['miss_rate'] else "N/A"
        print(f"{r['label']:<32} {r['utilization']:>6.3f}  {dm_s:>6}  {edf_s:>6}  {miss:>6}")

    total  = len(all_results)
    dm_ok  = sum(1 for r in all_results if r['dm_schedulable'])
    edf_ok = sum(1 for r in all_results if r['edf_schedulable'])
    print("-" * 80)
    print(f"  DM schedulable:  {dm_ok}/{total} ({dm_ok/total*100:.1f}%)")
    print(f"  EDF schedulable: {edf_ok}/{total} ({edf_ok/total*100:.1f}%)")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="02225 DRTS Mini-Project 1 — Full Schedulability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything with defaults (all u30/u50/u70/u90 sets, 300 sim runs)
  python analysis.py

  # Only the 70% utilisation sets
  python analysis.py --util u70

  # Single CSV file
  python analysis.py --csv Task-sets/taskset-1.csv

  # 1000 simulation runs, save plots
  python analysis.py --runs 1000 --save-plots plots/

  # Verbose: print per-task tables for every set
  python analysis.py --verbose --no-plots
""",
    )
    parser.add_argument(
        "--util", nargs="+", default=UTIL_FOLDERS,
        help="Utilisation folder(s) to process (default: u30 u50 u70 u90)"
    )
    parser.add_argument(
        "--csv", nargs="+", default=None,
        help="Specific CSV file(s) to analyse (overrides --util)"
    )
    parser.add_argument(
        "--runs", type=int, default=300,
        help="Number of stochastic simulation runs per task set (default: 300)"
    )
    parser.add_argument(
        "--save-plots", metavar="DIR", default=None,
        help="Directory to save plots as PNG files (default: show interactively)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Disable all matplotlib output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-task WCRT tables"
    )
    args = parser.parse_args()

    save_dir = args.save_plots
    if save_dir:
        _ensure_dir(save_dir)

    # ------------------------------------------------------------------
    # Determine which CSV files to process
    # ------------------------------------------------------------------
    if args.csv:
        # Single / explicit file mode
        all_results = []
        for csv_path in args.csv:
            tasks = _safe_load(csv_path)
            if tasks is None:
                continue
            print(f"\nAnalysing: {csv_path}")
            res = analyse_one(tasks, csv_path, n_runs=args.runs, verbose=True)
            all_results.append(res)

            if args.verbose:
                _dm.print_dm_results(res['dm_results'], tasks, csv_path=csv_path)
                _edf.print_edf_wcrt(res['edf_results'], tasks, csv_path=csv_path)
                _sim.print_simulation_results(
                    res['sim_results'], res['edf_results'],
                    tasks, res['miss_rate'], csv_path=csv_path
                )

            if not args.no_plots:
                sp_wcrt = os.path.join(save_dir, f"wcrt_{os.path.splitext(os.path.basename(csv_path))[0]}.png") if save_dir else None
                sp_cdf  = os.path.join(save_dir, f"cdf_{os.path.splitext(os.path.basename(csv_path))[0]}.png")  if save_dir else None
                plot_wcrt_comparison(res, save_path=sp_wcrt)
                plot_analytical_vs_simulation(res, save_path=sp_cdf)

    else:
        # Batch mode: process all utilisation folders
        util_dirs = []
        for u in args.util:
            d = os.path.join(TASK_SETS_DIR, u)
            if os.path.isdir(d):
                util_dirs.append(d)
            else:
                print(f"  WARNING: directory not found: {d}")

        if not util_dirs:
            print("ERROR: No utilisation directories found. "
                  "Check that Task-sets/u30 … u90 exist.")
            sys.exit(1)

        all_results = batch_analysis(util_dirs, n_runs=args.runs,
                                      verbose=args.verbose)

        if not all_results:
            print("No results produced.")
            sys.exit(0)

        print_summary_table(all_results)

        if not args.no_plots:
            # Plot 1: Schedulability rate vs utilisation
            sp1 = os.path.join(save_dir, "schedulability_vs_utilization.png") if save_dir else None
            plot_schedulability_vs_utilization(all_results, save_path=sp1)

            # Plot 2 + 3: Pick one representative task set per utilisation level
            for u in args.util:
                # Find first schedulable-by-EDF result for this utilisation
                recs = [r for r in all_results
                        if u in r['csv_path'] and r['edf_results'] is not None]
                if not recs:
                    continue
                rep = recs[0]
                base = f"{u}_{os.path.splitext(rep['label'])[0]}"
                sp2 = os.path.join(save_dir, f"wcrt_{base}.png") if save_dir else None
                sp3 = os.path.join(save_dir, f"cdf_{base}.png")  if save_dir else None
                plot_wcrt_comparison(rep, save_path=sp2)
                plot_analytical_vs_simulation(rep, save_path=sp3)

            # Also plot the "border" task set (taskset-1) if present
            border = os.path.join(TASK_SETS_DIR, "taskset-1.csv")
            if os.path.isfile(border):
                tasks = _safe_load(border)
                if tasks:
                    res = analyse_one(tasks, border, n_runs=args.runs, verbose=args.verbose)
                    sp2 = os.path.join(save_dir, "wcrt_taskset1_border.png") if save_dir else None
                    sp3 = os.path.join(save_dir, "cdf_taskset1_border.png")  if save_dir else None
                    print(f"\n  Border set (taskset-1): "
                          f"U={res['utilization']:.3f}  "
                          f"DM={'PASS' if res['dm_schedulable'] else 'FAIL'}  "
                          f"EDF={'PASS' if res['edf_schedulable'] else 'FAIL'}")
                    if not args.no_plots:
                        plot_wcrt_comparison(res, save_path=sp2)
                        plot_analytical_vs_simulation(res, save_path=sp3)


if __name__ == "__main__":
    main()
