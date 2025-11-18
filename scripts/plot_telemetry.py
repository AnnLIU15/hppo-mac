"""Plot telemetry curves using matplotlib and scienceplots styles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

try:  # scienceplots extends matplotlib styles
    import scienceplots  # noqa: F401  # pylint: disable=unused-import
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scienceplots is required, please install the optional dependency.") from exc

plt.style.use(["science", "grid", "no-latex"])  # activate SciencePlots theme


def _load_trace(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _aggregate(traces: Iterable[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    arrays = [trace[key] for trace in traces]
    if not arrays:
        return np.array([], dtype=np.float32)
    stacked = np.stack(arrays, axis=0)
    return stacked


def plot_telemetry(paths: List[Path], output: Path | None, show: bool, dpi: int = 200) -> None:
    traces = [_load_trace(path) for path in paths]
    if not traces:
        raise SystemExit("No telemetry files provided.")

    steps = traces[0]["step"].astype(np.int32)
    backlog_cbra = _aggregate(traces, "backlog_cbra")
    backlog_pbra = _aggregate(traces, "backlog_pbra")
    collisions = _aggregate(traces, "collisions")
    throughput = _aggregate(traces, "throughput")
    acb = _aggregate(traces, "acb")
    reward = _aggregate(traces, "reward")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8), layout="constrained")

    ax_backlog, ax_collision, ax_control = axes

    ax_backlog.set_title("Pending Backoff Queue Size")
    ax_backlog.plot(steps, backlog_cbra.mean(axis=0), label="CBRA", color="#1f77b4")
    ax_backlog.fill_between(
        steps,
        backlog_cbra.min(axis=0),
        backlog_cbra.max(axis=0),
        color="#1f77b4",
        alpha=0.15,
    )
    ax_backlog.plot(steps, backlog_pbra.mean(axis=0), label="PBRA", color="#ff7f0e")
    ax_backlog.set_ylabel("Terminals")
    ax_backlog.legend(loc="upper right")

    ax_collision.set_title("Throughput vs Collisions")
    ax_collision.plot(steps, throughput.mean(axis=0), label="Throughput", color="#2ca02c")
    ax_collision.plot(steps, collisions.mean(axis=0), label="Collisions", color="#d62728")
    ax_collision.set_ylabel("Count / step")
    ax_collision.legend(loc="upper right")

    ax_control.set_title("Control Signals")
    ax_control.plot(steps, acb.mean(axis=0), label="ACB", color="#9467bd")
    ax_control.plot(
        steps,
        reward.mean(axis=0),
        label="Reward",
        color="#8c564b",
        linestyle="--",
    )
    ax_control.set_ylabel("Value")
    ax_control.set_xlabel("Decision step")
    ax_control.legend(loc="upper right")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
        print(f"Saved plot to {output}")  # noqa: T201

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot telemetry .npz files.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more telemetry npz files produced by main.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when saving (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = [path for path in args.inputs if path.exists()]
    if not existing:
        raise SystemExit("None of the provided telemetry files exist.")
    plot_telemetry(existing, args.output, args.show, dpi=args.dpi)


if __name__ == "__main__":
    main()
