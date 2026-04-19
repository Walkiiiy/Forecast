#!/usr/bin/env python3
"""
Auto-select target-rate ladders for confidence-interval fitting.

Default behavior compares two 5%-step ladders:
- edge ladder:   95, 90, ..., 5
- center ladder: 97.5, 92.5, ..., 2.5

Selection objective:
1) prefer monotonic non-increasing interval actual rate,
2) minimize cumulative weighted abs gap to target,
3) minimize target-bin weighted abs gap to target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fit_confidence_target_intervals import (
    fit_intervals,
    make_cumulative_from_intervals,
    parse_targets_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-select best target ladders for confidence interval fitting.")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--confidence-col", type=str, default="raw_confidence")
    parser.add_argument("--label-col", type=str, default="actual_label")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--step", type=float, default=5.0, help="Step size in percent for built-in ladders.")
    parser.add_argument(
        "--candidate-modes",
        type=str,
        default="edge,center",
        help="Comma-separated modes from {edge,center}.",
    )
    parser.add_argument(
        "--extra-targets",
        type=str,
        default="",
        help=(
            "Optional extra target ladders; use ';' to separate ladders and ',' inside each ladder. "
            "Example: '97.5,92.5,...,2.5;95,90,...,5'"
        ),
    )
    parser.add_argument("--summary-json", type=str, default="auto_targets_search_summary.json")
    parser.add_argument("--summary-md", type=str, default="auto_targets_search_summary.md")
    parser.add_argument("--selected-targets-file", type=str, default="auto_selected_targets.txt")
    parser.add_argument(
        "--expect-targets",
        type=str,
        default="",
        help="Optional expected targets sequence; if provided, script exits non-zero when mismatch.",
    )
    return parser.parse_args()


def build_edge_targets(step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--step must be > 0")
    start = 100.0 - step
    end = step
    vals: List[float] = []
    x = start
    while x >= end - 1e-12:
        vals.append(round(x / 100.0, 10))
        x -= step
    return vals


def build_center_targets(step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--step must be > 0")
    start = 100.0 - step / 2.0
    end = step / 2.0
    vals: List[float] = []
    x = start
    while x >= end - 1e-12:
        vals.append(round(x / 100.0, 10))
        x -= step
    return vals


def format_targets_pct(targets: List[float]) -> str:
    return ",".join(f"{t * 100:.1f}".rstrip("0").rstrip(".") for t in targets)


def load_conf_and_labels(pred_csv: Path, confidence_col: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(pred_csv)
    if confidence_col not in df.columns:
        raise ValueError(f"missing confidence column: {confidence_col}")
    if label_col not in df.columns:
        raise ValueError(f"missing label column: {label_col}")
    df = df[[confidence_col, label_col]].copy()
    df[confidence_col] = pd.to_numeric(df[confidence_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[confidence_col, label_col]).copy()
    df = df.sort_values(confidence_col, ascending=False).reset_index(drop=True)
    conf = df[confidence_col].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy(dtype=np.int8)
    return conf, labels


def evaluate_ladder(
    name: str,
    targets: List[float],
    conf: np.ndarray,
    labels: np.ndarray,
    min_count: int,
) -> Dict[str, object]:
    intervals = fit_intervals(conf=conf, y=labels, targets=targets, min_count=min_count, add_tail_bin=True)
    target_df = intervals[~intervals["is_tail_bin"]].copy()
    cumulative_df = make_cumulative_from_intervals(intervals)

    if len(target_df) > 0:
        weighted_gap = float(np.average(target_df["abs_gap_to_target"], weights=target_df["count"]))
        mono = (
            bool(np.all(target_df["actual_rate"].values[:-1] >= target_df["actual_rate"].values[1:] - 1e-12))
            if len(target_df) > 1
            else True
        )
    else:
        weighted_gap = float("inf")
        mono = False

    if len(cumulative_df) > 0:
        cumulative_gap = float(
            np.average(
                cumulative_df["cumulative_abs_gap_to_target"],
                weights=cumulative_df["cumulative_count"],
            )
        )
    else:
        cumulative_gap = float("inf")

    return {
        "name": name,
        "targets": targets,
        "targets_text": format_targets_pct(targets),
        "target_bins_count": int(len(target_df)),
        "tail_bins_count": int(intervals["is_tail_bin"].sum()),
        "monotonic_nonincreasing_actual_rate": mono,
        "target_bins_weighted_abs_gap": weighted_gap,
        "cumulative_weighted_abs_gap": cumulative_gap,
    }


def parse_extra_ladders(extra_text: str) -> List[List[float]]:
    out: List[List[float]] = []
    if not extra_text.strip():
        return out
    for chunk in extra_text.split(";"):
        t = chunk.strip()
        if not t:
            continue
        out.append(parse_targets_text(t))
    return out


def pick_best(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    return sorted(
        candidates,
        key=lambda c: (
            0 if c["monotonic_nonincreasing_actual_rate"] else 1,
            c["cumulative_weighted_abs_gap"],
            c["target_bins_weighted_abs_gap"],
            -c["target_bins_count"],
        ),
    )[0]


def main() -> None:
    args = parse_args()
    pred_csv = Path(args.predictions_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(f"predictions file not found: {pred_csv}")

    out_dir = Path(args.output_dir) if args.output_dir else pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    conf, labels = load_conf_and_labels(pred_csv, args.confidence_col, args.label_col)

    mode_names = [m.strip().lower() for m in args.candidate_modes.split(",") if m.strip()]
    ladders: List[Tuple[str, List[float]]] = []
    for mode in mode_names:
        if mode == "edge":
            ladders.append(("edge", build_edge_targets(args.step)))
        elif mode == "center":
            ladders.append(("center", build_center_targets(args.step)))
        else:
            raise ValueError(f"unsupported candidate mode: {mode}")

    for idx, t in enumerate(parse_extra_ladders(args.extra_targets), start=1):
        ladders.append((f"extra_{idx}", t))

    if not ladders:
        raise ValueError("no candidate ladders found")

    candidates = [
        evaluate_ladder(name=name, targets=targets, conf=conf, labels=labels, min_count=args.min_count)
        for name, targets in ladders
    ]
    best = pick_best(candidates)

    summary = {
        "predictions_csv": str(pred_csv),
        "confidence_col": args.confidence_col,
        "label_col": args.label_col,
        "samples": int(len(labels)),
        "positive_rate": float(np.mean(labels)),
        "min_count": int(args.min_count),
        "step_percent": float(args.step),
        "candidates": candidates,
        "selected": best,
    }

    summary_json = out_dir / args.summary_json
    selected_targets_file = out_dir / args.selected_targets_file
    summary_md = out_dir / args.summary_md

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_targets_file.write_text(best["targets_text"] + "\n", encoding="utf-8")

    md_lines = [
        "# Auto Target Ladder Search Summary",
        "",
        "## Inputs",
        f"- predictions_csv: `{pred_csv}`",
        f"- samples: `{summary['samples']}`",
        f"- positive_rate: `{summary['positive_rate']:.6f}`",
        f"- min_count: `{args.min_count}`",
        f"- step: `{args.step:.3f}%`",
        "",
        "## Candidates",
        "",
        "| name | monotonic | target_bins_weighted_abs_gap | cumulative_weighted_abs_gap | target_bins_count | targets |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for c in sorted(
        candidates,
        key=lambda x: (
            0 if x["monotonic_nonincreasing_actual_rate"] else 1,
            x["cumulative_weighted_abs_gap"],
            x["target_bins_weighted_abs_gap"],
            -x["target_bins_count"],
        ),
    ):
        md_lines.append(
            "| "
            + f"{c['name']} | {int(c['monotonic_nonincreasing_actual_rate'])} | "
            + f"{c['target_bins_weighted_abs_gap']:.6f} | {c['cumulative_weighted_abs_gap']:.6f} | "
            + f"{c['target_bins_count']} | `{c['targets_text']}` |"
        )

    md_lines += [
        "",
        "## Selected",
        "",
        f"- name: `{best['name']}`",
        f"- targets: `{best['targets_text']}`",
        f"- target_bins_weighted_abs_gap: `{best['target_bins_weighted_abs_gap']:.6f}`",
        f"- cumulative_weighted_abs_gap: `{best['cumulative_weighted_abs_gap']:.6f}`",
        "",
        "## Next Command",
        "",
        "```bash",
        "python src/fit_confidence_target_intervals.py \\",
        f"  --predictions-csv {pred_csv} \\",
        f"  --output-dir {out_dir} \\",
        f"  --min-count {args.min_count} \\",
        f"  --targets \"{best['targets_text']}\"",
        "```",
        "",
    ]
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    if args.expect_targets.strip():
        expected = parse_targets_text(args.expect_targets)
        expected_text = format_targets_pct(expected)
        if best["targets_text"] != expected_text:
            print("[mismatch] expected:", expected_text)
            print("[mismatch] selected:", best["targets_text"])
            raise SystemExit(2)

    print("[done] summary-json:", summary_json)
    print("[done] summary-md:", summary_md)
    print("[done] selected-targets-file:", selected_targets_file)
    print("[selected]", best["name"], best["targets_text"])


if __name__ == "__main__":
    main()
