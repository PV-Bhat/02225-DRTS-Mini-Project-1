#!/usr/bin/env python3
"""Generate harmonic/semi-harmonic task sets for Mini Project 1.

Creates 80 CSV files (20 per utilization bracket) under:
    Task-sets-harmonic/{u30,u50,u70,u90}/taskset-<k>.csv
"""

import argparse
import csv
import os
import random
from pathlib import Path

PERIOD_POOL = [1000, 2000, 4000, 5000, 10000, 20000, 40000, 50000, 100000]
UTIL_LEVELS = {
    "u30": 0.30,
    "u50": 0.50,
    "u70": 0.70,
    "u90": 0.90,
}


def uunifast(n: int, total_u: float):
    utils = []
    sum_u = total_u
    for i in range(1, n):
        next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
        utils.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    utils.append(sum_u)
    random.shuffle(utils)
    return utils


def build_taskset(n_tasks: int, target_u: float):
    tasks = []
    utils = uunifast(n_tasks, target_u)

    for idx, u_i in enumerate(utils):
        period = random.choice(PERIOD_POOL)
        wcet = max(1, int(round(u_i * period)))
        if wcet >= period:
            wcet = period - 1
        bcet = max(1, int(round(wcet * random.uniform(0.4, 0.9))))
        deadline = random.randint(max(wcet, int(0.6 * period)), period)
        jitter = random.randint(0, max(1, period // 20))
        tasks.append({
            "Name": f"T{idx}",
            "Jitter": jitter,
            "BCET": bcet,
            "WCET": wcet,
            "Period": period,
            "Deadline": deadline,
            "PE": 0,
        })
    return tasks


def write_taskset_csv(path: Path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Name", "Jitter", "BCET", "WCET", "Period", "Deadline", "PE"],
        )
        writer.writeheader()
        writer.writerows(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="Task-sets-harmonic", help="Output directory")
    parser.add_argument("--sets-per-util", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_root = Path(__file__).resolve().parent / args.out
    for util_name, util_val in UTIL_LEVELS.items():
        util_dir = out_root / util_name
        for i in range(args.sets_per_util):
            tasks = build_taskset(args.tasks, util_val)
            write_taskset_csv(util_dir / f"taskset-{i}.csv", tasks)

    print(f"Generated harmonic dataset at: {out_root}")


if __name__ == "__main__":
    main()
