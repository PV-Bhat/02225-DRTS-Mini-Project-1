"""
Deadline Monotonic (DM) Worst-Case Response Time Analysis
=========================================================

Theory
------
Deadline Monotonic (DM) is a fixed-priority preemptive scheduling algorithm for
single-core systems where task deadlines are at most equal to their periods
(constrained deadlines: Dᵢ ≤ Tᵢ).

Priority assignment rule:
    The task with the SHORTER relative deadline receives the HIGHER priority.
    Ties in deadline are broken by shorter period (DM reduces to Rate-Monotonic
    when all Dᵢ = Tᵢ).

Schedulability test — Response-Time Analysis (RTA) [Buttazzo §4.5.2]:
    For each task τᵢ (sorted highest-to-lowest priority, i.e. shortest deadline
    first), the Worst-Case Response Time is the fixed point of:

        R⁽⁰⁾ᵢ  = Cᵢ
        R⁽ᵏ⁺¹⁾ᵢ = Cᵢ + Σⱼ ∈ hp(i)  ⌈ R⁽ᵏ⁾ᵢ / Tⱼ ⌉ · Cⱼ

    where hp(i) is the set of tasks with strictly higher priority than τᵢ.
    Iteration stops when R⁽ᵏ⁺¹⁾ᵢ == R⁽ᵏ⁾ᵢ  (fixed point reached).
    The task is NOT schedulable if the fixed point R > Dᵢ.

Why use DM?
-----------
* DM is optimal among all fixed-priority algorithms for constrained-deadline
  periodic tasks: if any fixed-priority assignment can schedule a set, DM can.
* Simple to implement (static priorities, computed offline).
* Predictable: high-priority tasks (short deadline) always get bounded, short
  WCRTs regardless of lower-priority load.
* Preferred when determinism and simplicity matter more than peak utilisation.

Limitations vs EDF
------------------
* DM can fail to schedule task sets that EDF handles: EDF uses the full CPU
  bandwidth (schedulable iff U ≤ 1), whereas DM has no tight utilisation bound
  for constrained-deadline sets.
* At utilisation > 0.85 the probability of DM failure rises sharply.
  This is why taskset-1 (U ≈ 0.85) is provided to demonstrate EDF superiority.

How to run
----------
Analyse a single task set (prints a formatted table):
    python DM-logic.py Task-sets/u70/taskset-0.csv

Analyse multiple task sets at once:
    python DM-logic.py Task-sets/u70/*.csv

Use analysis.py for batch processing across all utilisation levels with plots.

Dependencies
------------
Task-model.py is loaded at runtime via importlib (hyphen in filename prevents
a normal `import` statement).  No third-party packages are required.
"""

import math
import importlib.util
import os
import sys


# ---------------------------------------------------------------------------
# Load Task-model at runtime (hyphen in filename requires importlib)
# ---------------------------------------------------------------------------

def _load_module(name, filename):
    """Import a sibling module whose filename contains hyphens."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tm               = _load_module("task_model", "Task-model.py")
Task              = _tm.Task
load_taskset      = _tm.load_taskset
total_utilization = _tm.total_utilization


# ---------------------------------------------------------------------------
# DM Priority assignment
# ---------------------------------------------------------------------------

def assign_dm_priorities(tasks):
    """
    Sort tasks by DM priority and set the `.priority` attribute on each.

    Priority rule: shorter relative deadline → higher priority (rank 0).
    Equal-deadline ties are broken by shorter period.

    Parameters
    ----------
    tasks : list[Task]

    Returns
    -------
    list[Task]
        Same Task objects, ordered from highest priority (rank 0) to lowest.
    """
    sorted_tasks = sorted(tasks, key=lambda t: (t.deadline, t.period))
    for rank, task in enumerate(sorted_tasks):
        task.priority = rank       # 0 = highest priority
    return sorted_tasks


# ---------------------------------------------------------------------------
# Core WCRT computation
# ---------------------------------------------------------------------------

def compute_dm_wcrt(tasks):
    """
    Compute the Worst-Case Response Time for every task under DM scheduling.

    Implements the iterative fixed-point analysis from Buttazzo §4.5.2
    (Figure 4.17).

    Algorithm for task τᵢ (after sorting by DM priority):
        hp(i) = tasks with higher priority = sorted_tasks[0..i-1]

        R ← Cᵢ                             # initial guess
        loop:
            R_new ← Cᵢ + Σⱼ ∈ hp(i) ⌈R/Tⱼ⌉ · Cⱼ
            if R_new == R: break            # fixed point reached
            R ← R_new
            if R > Dᵢ: break               # already infeasible, stop early

        WCRTᵢ = R
        schedulable iff R ≤ Dᵢ

    The term ⌈R/Tⱼ⌉ · Cⱼ is the maximum interference from τⱼ during a busy
    window of length R: it can release at most ⌈R/Tⱼ⌉ jobs in [0, R).

    Parameters
    ----------
    tasks : list[Task]
        Unordered list of periodic tasks.

    Returns
    -------
    dict
        task_id → {
            'wcrt'        : int    — WCRT value (may exceed deadline if infeasible)
            'deadline'    : int    — Relative deadline Dᵢ
            'wcet'        : int    — Execution time Cᵢ
            'period'      : int    — Period Tᵢ
            'priority'    : int    — DM rank (0 = highest priority)
            'schedulable' : bool   — True iff WCRT ≤ deadline
        }
    """
    sorted_tasks = assign_dm_priorities(tasks)
    results = {}

    for i, task in enumerate(sorted_tasks):
        hp = sorted_tasks[:i]          # higher-priority tasks

        # Fixed-point iteration
        R = task.wcet                  # R⁽⁰⁾ = Cᵢ
        while True:
            interference = sum(
                math.ceil(R / hp_task.period) * hp_task.wcet
                for hp_task in hp
            )
            R_next = task.wcet + interference

            if R_next == R:
                break                  # converged
            R = R_next
            if R > task.deadline:
                break                  # early exit: infeasible

        results[task.task_id] = {
            'wcrt'        : R,
            'deadline'    : task.deadline,
            'wcet'        : task.wcet,
            'period'      : task.period,
            'priority'    : task.priority,
            'schedulable' : R <= task.deadline,
        }

    return results


def is_schedulable_dm(tasks):
    """
    Return True iff the task set is schedulable under DM.

    Uses full response-time analysis (not just a utilisation bound), so the
    result is both necessary and sufficient for DM schedulability.

    Parameters
    ----------
    tasks : list[Task]

    Returns
    -------
    bool
    """
    results = compute_dm_wcrt(tasks)
    return all(r['schedulable'] for r in results.values())


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_dm_results(results, tasks, csv_path=""):
    """Print a formatted DM WCRT result table to stdout."""
    u = total_utilization(tasks)
    label = os.path.basename(csv_path) if csv_path else "task set"

    print(f"\n{'='*64}")
    print(f"  DM Analysis — {label}")
    print(f"  Total utilisation U = {u:.4f}")
    print(f"{'='*64}")
    print(f"{'Task':<8} {'Rank':>5} {'C':>6} {'T':>8} {'D':>8} "
          f"{'WCRT':>8}  Status")
    print("-" * 64)

    all_ok = True
    for tid, r in results.items():
        status = "PASS" if r['schedulable'] else "FAIL ← deadline missed"
        print(f"{tid:<8} {r['priority']:>5} {r['wcet']:>6} {r['period']:>8} "
              f"{r['deadline']:>8} {r['wcrt']:>8}  {status}")
        if not r['schedulable']:
            all_ok = False

    print("-" * 64)
    verdict = "SCHEDULABLE under DM" if all_ok else "NOT SCHEDULABLE under DM"
    print(f"  Result: {verdict}")
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python DM-logic.py <taskset.csv> [<taskset2.csv> ...]")
        print("\nExamples:")
        print("  python DM-logic.py Task-sets/u70/taskset-0.csv")
        print("  python DM-logic.py Task-sets/u90/*.csv")
        sys.exit(0)

    for path in sys.argv[1:]:
        try:
            tasks = load_taskset(path)
        except FileNotFoundError:
            print(f"ERROR: file not found: {path}")
            continue

        if not tasks:
            print(f"WARNING: no tasks loaded from {path}")
            continue

        results = compute_dm_wcrt(tasks)
        print_dm_results(results, tasks, csv_path=path)
