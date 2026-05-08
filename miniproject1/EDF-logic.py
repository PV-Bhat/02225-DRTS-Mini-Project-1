"""
EDF (Earliest Deadline First) Schedulability Analysis and WCRT Computation
===========================================================================

Theory
------
Earliest Deadline First (EDF) is a dynamic-priority preemptive scheduling
algorithm.  At every instant the processor is assigned to the ready task whose
ABSOLUTE deadline is earliest.  EDF is optimal for uniprocessor scheduling:
a task set is schedulable by EDF iff it is schedulable by any algorithm.

This module implements TWO complementary analyses for periodic tasks with
constrained deadlines (Dᵢ ≤ Tᵢ) on a single core.

─────────────────────────────────────────────────────────────────────────────
A.  Schedulability Check — Processor Demand Criterion (Buttazzo §4.6.1)
─────────────────────────────────────────────────────────────────────────────
For constrained-deadline synchronous periodic tasks:

    Necessary condition:   U = Σ Cᵢ/Tᵢ ≤ 1

    Sufficient + necessary condition (processor demand):
        ∀ t in scheduling_points(0, H):
            dbf(0, t) = Σᵢ  max(0, ⌊(t + Tᵢ - Dᵢ) / Tᵢ⌋) · Cᵢ  ≤  t

    where dbf(0,t) is the Demand Bound Function — the maximum processor work
    demanded by all tasks in any window of length t.

    Scheduling points are the absolute deadlines of every job in [0, H]:
        t = k·Tᵢ + Dᵢ  for  k = 0, 1, …  while  t ≤ H

─────────────────────────────────────────────────────────────────────────────
B.  WCRT Computation — Analytical Hyperperiod Simulation
─────────────────────────────────────────────────────────────────────────────
For a synchronous periodic task set (all first releases at t = 0) with fixed
WCET execution times, the EDF schedule is deterministic and repeats every H
time units.  The exact WCRT is therefore obtained by simulating [0, H) with
every job executing for exactly its WCET (Buttazzo appendix A of the assignment):

    1.  H = lcm(T₁, …, Tₙ)
    2.  Generate all jobs:  τᵢ releases job k at r_{i,k} = k·Tᵢ  (k·Tᵢ < H)
                            absolute deadline  d_{i,k} = r_{i,k} + Dᵢ
    3.  Simulate preemptive EDF from t=0 to t=H:
            - jobs execute for exactly WCET
            - at every instant the ready job with earliest absolute deadline runs
    4.  Record finish time f_{i,k} for each job
    5.  R_{i,k} = f_{i,k} − r_{i,k}
    6.  WCRTᵢ = max_k { R_{i,k} }

Why EDF?
--------
* EDF achieves the theoretical utilisation bound: schedulable iff U ≤ 1.
* At high utilisation (>0.85) EDF schedules task sets that DM cannot.
* "Rate Monotonic vs. EDF: Judgment Day" (Buttazzo 2005) showed EDF's
  superiority is decisive as utilisation approaches 1.
* Trade-off: EDF priority changes at every job release → harder to implement
  in hardware and requires dynamic context decisions.

Hyperperiod size warning
------------------------
Non-harmonic periods can produce astronomically large hyperperiods.  The
simulation is event-driven (O(n_jobs · log n_jobs)), so even large H is
feasible as long as the total number of jobs n_jobs = Σᵢ H/Tᵢ is manageable.
A warning is issued when n_jobs > MAX_JOBS_SIMULATION.

How to run
----------
    python EDF-logic.py Task-sets/u70/taskset-0.csv
    python EDF-logic.py Task-sets/u90/*.csv

Use analysis.py for batch comparison across all utilisation levels.

Dependencies
------------
Task-model.py (loaded via importlib), Python standard library only.
"""

import math
import heapq
import importlib.util
import os
import sys
from functools import reduce


# ---------------------------------------------------------------------------
# Load Task-model at runtime
# ---------------------------------------------------------------------------

def _load_module(name, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tm               = _load_module("task_model", "Task-model.py")
Task              = _tm.Task
load_taskset      = _tm.load_taskset
hyperperiod       = _tm.hyperperiod
total_utilization = _tm.total_utilization


# ---------------------------------------------------------------------------
# Tuneable limits
# ---------------------------------------------------------------------------

#: Warn (but still run) if the analytical simulation would process more jobs.
MAX_JOBS_SIMULATION = 100_000

#: Skip simulation and return None results if jobs exceed this threshold.
#: Non-harmonic period sets produce hyperperiods in the billions; a
#: Python heap-based simulation of even 300 K jobs takes several seconds.
#: The DM WCRT (fixed-point) and EDF schedulability check (dbf) are always
#: fast regardless of hyperperiod size, so schedulability results are always
#: available even when WCRT simulation is skipped.
HARD_JOB_LIMIT = 200_000

#: Maximum number of scheduling-point checks for the processor demand test.
#: For non-harmonic task sets the hyperperiod can be astronomically large;
#: this caps the dbf scan to a manageable window while still catching all
#: practically-relevant violations.  For constrained-deadline sets with U < 1
#: the dbf grows slower than t asymptotically, so violations (if any) appear
#: early in the schedule.
MAX_DBF_CHECKPOINTS = 200_000


# ---------------------------------------------------------------------------
# A. Schedulability check — Processor Demand Criterion
# ---------------------------------------------------------------------------

def _scheduling_points(tasks, H, max_pts=MAX_DBF_CHECKPOINTS):
    """
    Return a sorted list of EDF scheduling points in (0, t_max].

    A scheduling point is the absolute deadline of any job:
        t = k·Tᵢ + Dᵢ  for k = 0, 1, …  while  k·Tᵢ < H  and  t ≤ H

    For non-harmonic task sets H can be astronomically large.  We cap the
    scan at max_pts points: once we have that many distinct values we stop
    collecting more.  The cap is safe because:
      * If U ≤ 1, any dbf(0,t) > t violation must appear within a finite
        "critical busy period" that is never larger than  Σ C_i / (1 - U).
      * We also hard-cap t at Σ_i(C_i)/(1-U) when U < 1 so the loop stops
        well before reaching H.

    Returns sorted list of up to max_pts scheduling points.
    """
    U = sum(t.wcet / t.period for t in tasks)

    # Compute a practical upper time bound: 3× the level-i busy period bound
    # (generous) or fall back to H when U == 1.
    if U < 1.0 - 1e-9:
        t_bound = min(H, int(sum(t.wcet for t in tasks) / (1.0 - U) * 3) + 1)
    else:
        t_bound = H

    pts = []
    seen = set()
    for task in tasks:
        k = 0
        while True:
            deadline = k * task.period + task.deadline
            if deadline > t_bound:
                break
            if deadline not in seen:
                seen.add(deadline)
                pts.append(deadline)
                if len(pts) >= max_pts:
                    break
            k += 1
        if len(pts) >= max_pts:
            break

    pts.sort()
    return pts[:max_pts]


def _dbf(tasks, t):
    """
    Demand Bound Function  dbf(0, t):  maximum processor work demanded in [0,t].

    For constrained-deadline tasks (Dᵢ ≤ Tᵢ):
        dbf(0,t) = Σᵢ  max(0, ⌊(t + Tᵢ − Dᵢ) / Tᵢ⌋) · Cᵢ
    """
    demand = 0
    for task in tasks:
        n_jobs = max(0, math.floor((t + task.period - task.deadline) / task.period))
        demand += n_jobs * task.wcet
    return demand


def edf_schedulability_check(tasks):
    """
    Test EDF schedulability via the Processor Demand Criterion (Buttazzo §4.6.1).

    Steps:
      1. Check necessary condition: U ≤ 1.
      2. Compute H = lcm of all periods.
      3. For every scheduling point t in (0, H]: check dbf(0,t) ≤ t.

    Parameters
    ----------
    tasks : list[Task]

    Returns
    -------
    schedulable : bool
        True iff the task set is schedulable under EDF.
    details : dict
        {
          'utilization'   : float,
          'hyperperiod'   : int,
          'n_checkpoints' : int,
          'violation_t'   : int or None   — first t where dbf > t, else None
          'violation_dbf' : int or None,
        }
    """
    U = total_utilization(tasks)
    if U > 1.0 + 1e-9:
        return False, {
            'utilization': U, 'hyperperiod': None,
            'n_checkpoints': 0,
            'violation_t': None, 'violation_dbf': None,
            'reason': 'U > 1 (necessary condition violated)',
        }

    H   = hyperperiod(tasks)
    pts = _scheduling_points(tasks, H)
    truncated = len(pts) >= MAX_DBF_CHECKPOINTS

    violation_t   = None
    violation_dbf = None
    for t in pts:
        d = _dbf(tasks, t)
        if d > t:
            violation_t   = t
            violation_dbf = d
            break

    schedulable = violation_t is None
    return schedulable, {
        'utilization'   : U,
        'hyperperiod'   : H,
        'n_checkpoints' : len(pts),
        'truncated'     : truncated,
        'violation_t'   : violation_t,
        'violation_dbf' : violation_dbf,
    }


# ---------------------------------------------------------------------------
# B. WCRT computation — Analytical Hyperperiod EDF Simulation
# ---------------------------------------------------------------------------

def _run_edf_simulation(tasks, H, random_et=False, rng=None):
    """
    Core preemptive EDF simulation over [0, H).

    All jobs released in the interval [0, H) are simulated.  Each job executes
    for exactly WCET (analytical mode) or a value drawn uniformly from
    [BCET, WCET] (stochastic mode).

    The simulation is event-driven: time advances to the next job release or
    job completion, whichever comes first.  Preemption occurs when a newly
    released job has an earlier absolute deadline than the currently running one.

    Parameters
    ----------
    tasks     : list[Task]
    H         : int   — hyperperiod / simulation horizon
    random_et : bool  — if True, sample execution time from [BCET, WCET]
    rng       : object with .randint(a,b) method, or None (uses random module)

    Returns
    -------
    jobs : list[dict]
        One entry per job with keys:
            task_id, release, abs_deadline, et (execution time used),
            remaining (0 when finished), finish (finish time or None if missed)
    deadline_missed : bool
    """
    if random_et and rng is None:
        import random as _rng
        rng = _rng

    # ------------------------------------------------------------------
    # 1. Generate all jobs sorted by release time (tie-break by deadline)
    # ------------------------------------------------------------------
    jobs = []
    for task in tasks:
        k = 0
        while k * task.period < H:
            r = k * task.period
            d = r + task.deadline
            if random_et:
                et = rng.randint(task.bcet, task.wcet) if task.bcet < task.wcet else task.wcet
            else:
                et = task.wcet
            jobs.append({
                'task_id'      : task.task_id,
                'release'      : r,
                'abs_deadline' : d,
                'et'           : et,
                'remaining'    : et,
                'finish'       : None,
            })
            k += 1
    jobs.sort(key=lambda j: (j['release'], j['abs_deadline']))

    # ------------------------------------------------------------------
    # 2. Event-driven preemptive EDF simulation
    # ------------------------------------------------------------------
    t          = 0
    ptr        = 0        # index of next job to release
    ready      = []       # min-heap: (abs_deadline, release_tie, seq, job)
    running    = None     # currently executing job
    run_start  = 0        # time when 'running' last started or was resumed
    seq        = 0        # tie-breaking counter for heap

    def push_ready(job):
        nonlocal seq
        heapq.heappush(ready, (job['abs_deadline'], job['release'], seq, job))
        seq += 1

    # Release all jobs with release == 0
    while ptr < len(jobs) and jobs[ptr]['release'] == 0:
        push_ready(jobs[ptr])
        ptr += 1

    # Pick the first job to run
    if ready:
        _, _, _, running = heapq.heappop(ready)
        run_start = 0

    deadline_missed = False

    while running is not None or ptr < len(jobs) or ready:
        if running is None:
            # CPU idle — jump to the next release
            if ptr >= len(jobs):
                break
            t = jobs[ptr]['release']
            while ptr < len(jobs) and jobs[ptr]['release'] <= t:
                push_ready(jobs[ptr])
                ptr += 1
            if ready:
                _, _, _, running = heapq.heappop(ready)
                run_start = t
            continue

        # When does the current job finish (if uninterrupted)?
        finish_at   = run_start + running['remaining']
        next_rel    = jobs[ptr]['release'] if ptr < len(jobs) else math.inf

        if finish_at <= next_rel:
            # ── Job completes before any new release ──────────────────
            t = finish_at
            running['finish']    = t
            running['remaining'] = 0
            if t > running['abs_deadline']:
                deadline_missed = True

            # Release jobs at this exact moment
            while ptr < len(jobs) and jobs[ptr]['release'] <= t:
                push_ready(jobs[ptr])
                ptr += 1

            running = None
            if ready:
                _, _, _, running = heapq.heappop(ready)
                run_start = t

        else:
            # ── New release(s) happen before job finishes ─────────────
            elapsed = next_rel - run_start
            running['remaining'] -= elapsed
            run_start = next_rel
            t = next_rel

            while ptr < len(jobs) and jobs[ptr]['release'] <= t:
                push_ready(jobs[ptr])
                ptr += 1

            # Preemption check: is any ready job's deadline earlier?
            if ready and ready[0][0] < running['abs_deadline']:
                push_ready(running)           # put running back in queue
                _, _, _, running = heapq.heappop(ready)
                run_start = t

    return jobs, deadline_missed


def compute_edf_wcrt(tasks, max_jobs=None):
    """
    Compute the Worst-Case Response Time for every task under EDF.

    Uses the analytical hyperperiod simulation: all jobs execute for exactly
    WCET, producing the deterministic worst-case schedule.

    Parameters
    ----------
    tasks   : list[Task]
    max_jobs : int or None
        If the total number of jobs in the hyperperiod exceeds this value the
        simulation is skipped and None is returned.  Defaults to
        HARD_JOB_LIMIT.

    Returns
    -------
    dict or None
        task_id → {
            'wcrt'        : int   — worst-case response time
            'deadline'    : int   — Dᵢ
            'wcet'        : int   — Cᵢ
            'period'      : int   — Tᵢ
            'schedulable' : bool  — True iff WCRT ≤ deadline
        }
        Returns None if the hyperperiod is too large to simulate.
    """
    if max_jobs is None:
        max_jobs = HARD_JOB_LIMIT

    H = hyperperiod(tasks)
    n_jobs = sum(H // t.period for t in tasks)

    if n_jobs > max_jobs:
        print(f"  [EDF-WCRT] Skipping simulation: {n_jobs:,} jobs > limit {max_jobs:,}")
        return None

    if n_jobs > MAX_JOBS_SIMULATION:
        print(f"  [EDF-WCRT] Warning: large simulation — {n_jobs:,} jobs")

    jobs, _ = _run_edf_simulation(tasks, H, random_et=False)

    # Collect per-task maximum response time
    wcrt_map = {}     # task_id → max response time seen
    for job in jobs:
        if job['finish'] is not None:
            rt = job['finish'] - job['release']
            tid = job['task_id']
            if tid not in wcrt_map or rt > wcrt_map[tid]:
                wcrt_map[tid] = rt

    results = {}
    for task in tasks:
        tid = task.task_id
        wcrt = wcrt_map.get(tid, task.wcet)   # fallback = WCET (1 job, no interference)
        results[tid] = {
            'wcrt'        : wcrt,
            'deadline'    : task.deadline,
            'wcet'        : task.wcet,
            'period'      : task.period,
            'schedulable' : wcrt <= task.deadline,
        }
    return results


def is_schedulable_edf(tasks):
    """
    Return True iff the task set is schedulable under EDF.

    Uses the Processor Demand Criterion (necessary and sufficient for
    synchronous constrained-deadline periodic tasks).

    Parameters
    ----------
    tasks : list[Task]

    Returns
    -------
    bool
    """
    ok, _ = edf_schedulability_check(tasks)
    return ok


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_edf_schedulability(ok, details, csv_path=""):
    label = os.path.basename(csv_path) if csv_path else "task set"
    trunc = " (capped)" if details.get('truncated') else ""
    print(f"\n{'='*64}")
    print(f"  EDF Schedulability — {label}")
    print(f"  U = {details['utilization']:.4f}  |  "
          f"H = {details.get('hyperperiod', 'N/A')}  |  "
          f"Checkpoints: {details['n_checkpoints']}{trunc}")
    print(f"{'='*64}")
    if ok:
        print("  Result: SCHEDULABLE under EDF (demand bound satisfied)")
    else:
        reason = details.get('reason', '')
        if reason:
            print(f"  Result: NOT SCHEDULABLE — {reason}")
        else:
            t = details['violation_t']
            d = details['violation_dbf']
            print(f"  Result: NOT SCHEDULABLE — dbf({t}) = {d} > {t}")
    print(f"{'='*64}")


def print_edf_wcrt(results, tasks, csv_path=""):
    """Print a formatted EDF WCRT result table."""
    if results is None:
        print("  [EDF-WCRT] Results unavailable (hyperperiod too large).")
        return
    u = total_utilization(tasks)
    label = os.path.basename(csv_path) if csv_path else "task set"
    print(f"\n{'='*64}")
    print(f"  EDF WCRT — {label}")
    print(f"  Total utilisation U = {u:.4f}")
    print(f"{'='*64}")
    print(f"{'Task':<8} {'C':>6} {'T':>8} {'D':>8} {'WCRT':>8}  Status")
    print("-" * 64)

    all_ok = True
    for tid, r in results.items():
        status = "PASS" if r['schedulable'] else "FAIL ← deadline missed"
        print(f"{tid:<8} {r['wcet']:>6} {r['period']:>8} {r['deadline']:>8} "
              f"{r['wcrt']:>8}  {status}")
        if not r['schedulable']:
            all_ok = False

    print("-" * 64)
    verdict = "SCHEDULABLE under EDF" if all_ok else "NOT SCHEDULABLE under EDF"
    print(f"  Result: {verdict}")
    print(f"{'='*64}")


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python EDF-logic.py <taskset.csv> [<taskset2.csv> ...]")
        print("\nExamples:")
        print("  python EDF-logic.py Task-sets/u70/taskset-0.csv")
        print("  python EDF-logic.py Task-sets/u90/*.csv")
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

        ok, details = edf_schedulability_check(tasks)
        print_edf_schedulability(ok, details, csv_path=path)

        wcrt_results = compute_edf_wcrt(tasks)
        print_edf_wcrt(wcrt_results, tasks, csv_path=path)
