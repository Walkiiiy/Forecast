#!/usr/bin/env python3
"""
Fit variable-length confidence intervals so each interval's actual rate
is as close as possible to target levels (e.g., 95%, 90%, ..., 5%),
under a minimum sample-count constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit target-rate confidence intervals from prediction output.")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--confidence-col", type=str, default="raw_confidence")
    parser.add_argument("--label-col", type=str, default="actual_label")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--target-start", type=float, default=95.0)
    parser.add_argument("--target-end", type=float, default=5.0)
    parser.add_argument("--target-step", type=float, default=5.0)
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated target rates in percent, e.g. '97.5,92.5,87.5,...'. If set, overrides start/end/step.",
    )
    parser.add_argument("--add-tail-bin", action="store_true", default=True)
    return parser.parse_args()


def build_targets(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("target-step must be > 0")
    if start < end:
        raise ValueError("target-start must be >= target-end")
    targets: List[float] = []
    x = float(start)
    while x >= end - 1e-12:
        targets.append(float(round(x / 100.0, 10)))
        x -= step
    return targets


def parse_targets_text(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        t = part.strip()
        if not t:
            continue
        vals.append(float(t) / 100.0)
    if not vals:
        raise ValueError("--targets is empty after parsing")
    # Ensure descending order for cumulative top-down segmentation.
    vals = sorted(vals, reverse=True)
    return vals


def fit_intervals(
    conf: np.ndarray,
    y: np.ndarray,
    targets: List[float],
    min_count: int,
    add_tail_bin: bool = True,
) -> pd.DataFrame:
    n = len(y)
    prefix = np.r_[0, np.cumsum(y)]

    def seg_stats(a: int, b: int):
        cnt = b - a
        pos = int(prefix[b] - prefix[a])
        rate = pos / cnt
        avg_conf = float(np.mean(conf[a:b]))
        return cnt, pos, rate, avg_conf

    rows = []
    start = 0

    for i, t in enumerate(targets):
        remaining_targets = len(targets) - i - 1
        min_end = start + min_count
        max_end = n - remaining_targets * min_count
        if min_end > max_end:
            break

        best_end = min_end
        best_err = float("inf")
        best_span = None

        for end in range(min_end, max_end + 1):
            cnt, pos, rate, avg_conf = seg_stats(start, end)
            err = abs(rate - t)
            span = cnt
            # Tie-break: prefer shorter interval for sharper high-confidence segmentation.
            if (err < best_err - 1e-12) or (abs(err - best_err) <= 1e-12 and (best_span is None or span < best_span)):
                best_err = err
                best_end = end
                best_span = span

        cnt, pos, rate, avg_conf = seg_stats(start, best_end)
        rows.append(
            {
                "interval_id": len(rows) + 1,
                "target_rate": t,
                "start_index": start,
                "end_index_exclusive": best_end,
                "count": cnt,
                "positive_count": pos,
                "negative_count": cnt - pos,
                "actual_rate": rate,
                "avg_confidence": avg_conf,
                "abs_gap_to_target": abs(rate - t),
                "conf_upper": float(conf[start]),
                "conf_lower": float(conf[best_end - 1]),
                "is_tail_bin": False,
            }
        )
        start = best_end

    if add_tail_bin and start < n:
        cnt, pos, rate, avg_conf = seg_stats(start, n)
        rows.append(
            {
                "interval_id": len(rows) + 1,
                "target_rate": np.nan,
                "start_index": start,
                "end_index_exclusive": n,
                "count": cnt,
                "positive_count": pos,
                "negative_count": cnt - pos,
                "actual_rate": rate,
                "avg_confidence": avg_conf,
                "abs_gap_to_target": np.nan,
                "conf_upper": float(conf[start]),
                "conf_lower": float(conf[n - 1]),
                "is_tail_bin": True,
            }
        )

    return pd.DataFrame(rows)


def make_cumulative_from_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    non_tail = intervals[~intervals["is_tail_bin"]].copy()
    if non_tail.empty:
        return pd.DataFrame(
            columns=[
                "cumulative_id",
                "target_rate",
                "cumulative_count",
                "cumulative_positive_count",
                "cumulative_negative_count",
                "cumulative_actual_rate",
                "cumulative_avg_confidence",
                "cumulative_abs_gap_to_target",
                "conf_upper",
                "conf_lower",
            ]
        )

    cum_rows = []
    total_count = 0
    total_pos = 0
    conf_weighted_sum = 0.0
    for i, r in non_tail.iterrows():
        cnt = int(r["count"])
        pos = int(r["positive_count"])
        avg_conf = float(r["avg_confidence"])
        target = float(r["target_rate"])

        total_count += cnt
        total_pos += pos
        conf_weighted_sum += avg_conf * cnt

        cum_rate = total_pos / max(total_count, 1)
        cum_avg_conf = conf_weighted_sum / max(total_count, 1)
        cum_rows.append(
            {
                "cumulative_id": len(cum_rows) + 1,
                "target_rate": target,
                "cumulative_count": total_count,
                "cumulative_positive_count": total_pos,
                "cumulative_negative_count": total_count - total_pos,
                "cumulative_actual_rate": cum_rate,
                "cumulative_avg_confidence": cum_avg_conf,
                "cumulative_abs_gap_to_target": abs(cum_rate - target),
                "conf_upper": float(non_tail.iloc[0]["conf_upper"]),
                "conf_lower": float(r["conf_lower"]),
            }
        )
    out = pd.DataFrame(cum_rows)
    int_cols = ["cumulative_id", "cumulative_count", "cumulative_positive_count", "cumulative_negative_count"]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].astype(np.int64)
    return out


def to_md_table(df: pd.DataFrame, cols: List[str]) -> List[str]:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return f"{float(v):.6f}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return lines


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions file not found: {pred_path}")

    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    if args.confidence_col not in df.columns:
        raise ValueError(f"missing confidence column: {args.confidence_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"missing label column: {args.label_col}")

    df = df[[args.confidence_col, args.label_col]].copy()
    df[args.confidence_col] = pd.to_numeric(df[args.confidence_col], errors="coerce")
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df.dropna(subset=[args.confidence_col, args.label_col]).copy()
    df = df.sort_values(args.confidence_col, ascending=False).reset_index(drop=True)

    conf = df[args.confidence_col].to_numpy(dtype=np.float64)
    y = df[args.label_col].to_numpy(dtype=np.int8)

    if args.targets.strip():
        targets = parse_targets_text(args.targets)
    else:
        targets = build_targets(args.target_start, args.target_end, args.target_step)
    intervals = fit_intervals(
        conf=conf,
        y=y,
        targets=targets,
        min_count=args.min_count,
        add_tail_bin=args.add_tail_bin,
    )

    target_df = intervals[~intervals["is_tail_bin"]].copy()
    cumulative_df = make_cumulative_from_intervals(intervals)
    if len(target_df) > 1:
        monotonic = bool(np.all(target_df["actual_rate"].values[:-1] >= target_df["actual_rate"].values[1:] - 1e-12))
    else:
        monotonic = True

    weighted_gap = float(
        np.average(target_df["abs_gap_to_target"], weights=target_df["count"])
    ) if len(target_df) > 0 else np.nan
    cumulative_weighted_gap = float(
        np.average(cumulative_df["cumulative_abs_gap_to_target"], weights=cumulative_df["cumulative_count"])
    ) if len(cumulative_df) > 0 else np.nan

    summary = {
        "predictions_csv": str(pred_path),
        "confidence_col": args.confidence_col,
        "label_col": args.label_col,
        "samples": int(len(df)),
        "positive_rate": float(np.mean(y)),
        "targets": targets,
        "min_count": int(args.min_count),
        "target_bins_count": int(len(target_df)),
        "tail_bin_count": int(intervals["is_tail_bin"].sum()),
        "target_bins_weighted_abs_gap": weighted_gap,
        "target_bins_monotonic_nonincreasing_actual_rate": monotonic,
        "cumulative_weighted_abs_gap_to_target": cumulative_weighted_gap,
    }

    csv_path = out_dir / "confidence_target_intervals.csv"
    cum_csv_path = out_dir / "confidence_target_intervals_cumulative.csv"
    md_path = out_dir / "confidence_target_intervals_report.md"
    json_path = out_dir / "confidence_target_intervals_summary.json"

    intervals.to_csv(csv_path, index=False)
    cumulative_df.to_csv(cum_csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Confidence 区间重划分报告（目标准确率阶梯）",
        "",
        "## 设置",
        f"- 输入文件: `{pred_path}`",
        f"- 最小区间样本数: `{args.min_count}`",
        f"- 目标准确率序列: `{', '.join([f'{t*100:.1f}%' for t in targets])}`",
        "",
        "## 总结",
        f"- 样本数: `{summary['samples']}`",
        f"- 正样本率: `{summary['positive_rate']:.6f}`",
        f"- 目标区间数: `{summary['target_bins_count']}`",
        f"- 尾部区间数: `{summary['tail_bin_count']}`",
        f"- 目标区间加权绝对误差: `{summary['target_bins_weighted_abs_gap']:.6f}`",
        f"- 累计区间加权绝对误差: `{summary['cumulative_weighted_abs_gap_to_target']:.6f}`",
        f"- 目标区间实际率是否单调不增: `{summary['target_bins_monotonic_nonincreasing_actual_rate']}`",
        "",
        "## 区间结果",
        "",
    ]
    lines += to_md_table(
        intervals,
        [
            "interval_id",
            "target_rate",
            "count",
            "actual_rate",
            "avg_confidence",
            "abs_gap_to_target",
            "conf_upper",
            "conf_lower",
            "is_tail_bin",
        ],
    )
    lines += [
        "",
        "## 累计区间结果（Cumulative）",
        "",
        "说明：第 k 行表示从最高置信度区间累计到当前目标区间后的统计。",
        "",
    ]
    lines += to_md_table(
        cumulative_df,
        [
            "cumulative_id",
            "target_rate",
            "cumulative_count",
            "cumulative_actual_rate",
            "cumulative_avg_confidence",
            "cumulative_abs_gap_to_target",
            "conf_upper",
            "conf_lower",
        ],
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] {csv_path}")
    print(f"[done] {cum_csv_path}")
    print(f"[done] {json_path}")
    print(f"[done] {md_path}")


if __name__ == "__main__":
    main()
