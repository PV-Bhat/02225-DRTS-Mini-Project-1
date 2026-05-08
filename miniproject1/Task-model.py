"""
Task Model for Real-Time Scheduling Analysis
=============================================

This module defines the core data structures and utility functions used by all
other analysis modules.

Why this exists
---------------
Both DM-logic.py and EDF-logic.py need a common representation of a periodic
real-time task (τᵢ) and helpers for loading task sets from the CSV files
produced by task_generator.py.  Centralising these here avoids duplication and
ensures every module works from identical data.

CSV format (from task_generator.py / priority_generator.py):
    Name, Jitter, BCET, WCET, Period, Deadline, PE

How to use
----------
Because the module file name contains a hyphen (Task-model.py), Python's normal
`import` statement cannot be used.  Every sibling module loads it at runtime
with the helper below:

    import importlib.util, os
    def _load_module(name, filename):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    _tm = _load_module("task_model", "Task-model.py")

You can also run this file directly for a quick sanity-check:
    python Task-model.py Task-sets/u70/taskset-0.csv
"""

import csv
import math
from functools import reduce


# ---------------------------------------------------------------------------
# Core data class
# ---------------------------------------------------------------------------

class Task:
    """
    A single periodic real-time task τᵢ.

    Parameters
    ----------
    task_id  : str   — Identifier, e.g. "T0"
    bcet     : int   — Best-Case Execution Time (lower bound on actual runtime)
    wcet     : int   — Worst-Case Execution Time  Cᵢ  (used for analysis)
    period   : int   — Period Tᵢ  (jobs are released every Tᵢ time units)
    deadline : int   — Relative deadline Dᵢ  (Dᵢ ≤ Tᵢ for constrained deadlines)
    jitter   : int   — Release jitter (0 = synchronous / deterministic release)
    pe       : int   — Processing element index (0 = single core)

    Notes
    -----
    * `priority` is left as None here; it is assigned externally by
      assign_dm_priorities() inside DM-logic.py.
    * For all worst-case analytical calculations use `wcet` as Cᵢ.
    * For stochastic simulation, sample execution time uniformly from
      [bcet, wcet].
    """

    def __init__(self, task_id, bcet, wcet, period, deadline, jitter=0, pe=0):
        self.task_id  = task_id
        self.bcet     = bcet       # Best-Case Execution Time
        self.wcet     = wcet       # Worst-Case Execution Time  (Cᵢ)
        self.period   = period     # Period  (Tᵢ)
        self.deadline = deadline   # Relative deadline  (Dᵢ)
        self.jitter   = jitter     # Release jitter (unused in synchronous analysis)
        self.pe       = pe         # Processing element
        self.priority = None       # Set by DM priority assignment

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def utilization(self):
        """CPU utilisation contribution:  Uᵢ = Cᵢ / Tᵢ  (using WCET)."""
        return self.wcet / self.period

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Task({self.task_id!r}, C={self.wcet}, T={self.period}, "
            f"D={self.deadline}, U={self.utilization:.4f})"
        )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_taskset(csv_path):
    """
    Load a task set from a CSV file produced by task_generator.py.

    Supported column names (case-insensitive, whitespace-stripped):
        Name / TaskID, Jitter, BCET, WCET, Period, Deadline, PE

    Parameters
    ----------
    csv_path : str or Path
        Path to the .csv file.

    Returns
    -------
    list[Task]
        Tasks in the order they appear in the file.

    Raises
    ------
    FileNotFoundError  if csv_path does not exist.
    ValueError         if a required column (WCET, Period) is missing.
    """
    tasks = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return tasks
        # Normalise column headers
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items() if k and k.strip()}

            task_id  = (row.get('Name')     or row.get('TaskID') or
                        row.get('name')     or row.get('taskid') or '?')
            bcet     = int(float(row.get('BCET',     row.get('bcet',     '0'))))
            wcet     = int(float(row.get('WCET',     row.get('wcet',     '0'))))
            period   = int(float(row.get('Period',   row.get('period',   '1'))))
            deadline = int(float(row.get('Deadline', row.get('deadline', str(period)))))
            jitter   = int(float(row.get('Jitter',   row.get('jitter',   '0'))))
            pe       = int(float(row.get('PE',       row.get('pe',       '0'))))

            if period == 0:
                continue  # skip degenerate rows

            tasks.append(Task(task_id, bcet, wcet, period, deadline, jitter, pe))
    return tasks


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def _lcm(a, b):
    """Least-common multiple of two positive integers."""
    return a * b // math.gcd(a, b)


def hyperperiod(tasks):
    """
    Compute the hyperperiod  H = lcm(T₁, T₂, …, Tₙ).

    The EDF schedule for a synchronous periodic task set is cyclic with period H,
    so analysing [0, H) is sufficient for exact WCRT determination.

    Returns
    -------
    int
        The LCM of all task periods.  Can be very large for non-harmonic sets.

    Notes
    -----
    If the returned value exceeds a practical simulation budget, callers should
    warn the user rather than running an infeasible simulation.
    """
    return reduce(_lcm, [t.period for t in tasks])


def total_utilization(tasks):
    """
    Compute total CPU utilisation  U = Σ Cᵢ/Tᵢ  (using WCET for all tasks).

    Returns
    -------
    float
        Values > 1.0 mean the task set is infeasible under any scheduler.
    """
    return sum(t.wcet / t.period for t in tasks)


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_taskset(tasks, title="Task Set"):
    """Print a formatted table of the task set."""
    print(f"\n{title}")
    print(f"{'Name':<8} {'BCET':>6} {'WCET':>6} {'Period':>8} {'Deadline':>10} {'Uᵢ':>8}")
    print("-" * 52)
    for t in tasks:
        print(
            f"{t.task_id:<8} {t.bcet:>6} {t.wcet:>6} "
            f"{t.period:>8} {t.deadline:>10} {t.utilization:>8.4f}"
        )
    u = total_utilization(tasks)
    print("-" * 52)
    print(f"{'Total U':>42} {u:>8.4f}")


# ---------------------------------------------------------------------------
# Standalone entry-point (quick sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python Task-model.py <taskset.csv>")
        sys.exit(0)

    ts = load_taskset(sys.argv[1])
    print_taskset(ts, title=f"Loaded from: {sys.argv[1]}")
    H = hyperperiod(ts)
    print(f"\nHyperperiod H = {H}")
    print(f"Number of jobs in [0,H): {sum(H // t.period for t in ts)}")
