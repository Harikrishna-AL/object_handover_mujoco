"""Print a run's metrics from its tensorboard logs.

Everything the training script prints is also written to disk, so a lost
terminal costs nothing. This reads the event files back and prints the handover
metrics as a table, which is usually what you wanted from the scrollback anyway.

    python scripts/summarize.py runs/valley4
    python scripts/summarize.py runs/valley4 --rows 30
    python scripts/summarize.py runs/*/          # compare several runs
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Metrics worth seeing, in the order they answer questions:
# is it reaching, is it holding, is it dropping, is it scoring.
KEYS = [
    "handover/closest_approach_m",
    "handover/peak_load_fraction",
    "handover/drop_rate",
    "handover/success_rate",
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
]
SHORT = {
    "handover/closest_approach_m": "approach",
    "handover/peak_load_fraction": "peak_f",
    "handover/drop_rate": "drop",
    "handover/success_rate": "success",
    "rollout/ep_len_mean": "ep_len",
    "rollout/ep_rew_mean": "ep_rew",
}


def find_runs(run_dir: str) -> list[str]:
    """Every distinct training run under `run_dir`.

    Stable Baselines does not overwrite a log directory -- it adds PPO_1, PPO_2,
    and so on. Globbing all event files and merging them silently interleaves
    separate runs into one nonsense series, which is exactly what this did
    before. Each event file is its own run.
    """
    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True)
    return sorted({os.path.dirname(f) for f in files})


def read_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Pull scalar series out of the event files in exactly this directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        sys.exit("tensorboard is not installed: pip install tensorboard")

    series: dict[str, list[tuple[int, float]]] = {}
    files = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
    if not files:
        return series

    for path in sorted(files):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, []).extend(
                (event.step, event.value) for event in acc.Scalars(tag)
            )
    for tag in series:
        series[tag].sort()
    return series


def summarize(run_dir: str, rows: int) -> None:
    series = read_scalars(run_dir)
    if not series:
        print(f"{run_dir}: no event files found")
        return

    present = [k for k in KEYS if k in series]
    if not present:
        print(f"{run_dir}: no handover metrics; tags present: {sorted(series)[:8]}")
        return

    steps = [s for s, _ in series[present[0]]]
    total = steps[-1] if steps else 0
    print(f"\n=== {run_dir}   ({len(steps)} logged points, up to step {total:,}) ===")
    print("  " + f"{'step':>10}" + "".join(f"{SHORT[k]:>12}" for k in present))
    print("  " + "-" * (10 + 12 * len(present)))

    stride = max(1, len(steps) // max(1, rows))
    for i in range(0, len(steps), stride):
        line = f"  {steps[i]:>10,}"
        for key in present:
            values = series[key]
            line += f"{values[i][1]:>12.3f}" if i < len(values) else f"{'-':>12}"
        print(line)

    # First and last decile, which is what "did it improve" actually means.
    print("\n  " + f"{'':>10}" + "".join(f"{SHORT[k]:>12}" for k in present))
    for label, sl in (("first 10%", slice(0, max(1, len(steps) // 10))),
                      ("last 10%", slice(-max(1, len(steps) // 10), None))):
        line = f"  {label:>10}"
        for key in present:
            chunk = [v for _, v in series[key][sl]]
            line += f"{sum(chunk) / len(chunk):>12.3f}" if chunk else f"{'-':>12}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rows", type=int, default=20)
    args = ap.parse_args()
    for root in args.runs:
        root = root.rstrip("/")
        found = find_runs(root)
        if not found:
            print(f"{root}: no event files found")
            continue
        if len(found) > 1:
            print(f"\n{root}: {len(found)} separate runs in this directory "
                  f"(later ones are more recent)")
        for run in found:
            summarize(run, args.rows)


if __name__ == "__main__":
    main()
