#!/usr/bin/env python3
"""
Train global RandomForest and export raw-confidence 5% bin reports.

Outputs:
- random_forest_raw_confidence_5pct.csv
- random_forest_raw_confidence_5pct_cumulative.csv
- random_forest_raw_confidence_5pct_summary.json
- random_forest_raw_confidence_5pct.md
- random_forest_raw_confidence_5pct_cumulative.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score

from confidence_forecast_extreme_rebuild import BuildConfig, build_dataset


BIN_EDGES = np.linspace(0.0, 1.0, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RandomForest raw confidence 5% reports.")
    parser.add_argument("--data-folder", type=str, default="data/top350")
    parser.add_argument("--output-dir", type=str, default="outputs/global_model_accuracy_benchmark_v2")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--history-days", type=int, default=21)
    parser.add_argument("--min-days", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--calib-ratio", type=float, default=0.15)
    parser.add_argument("--max-negative-gap", type=int, default=2)
    parser.add_argument("--behavior-prior", type=float, default=2.0)
    parser.add_argument("--user-group-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def threshold_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.int8)
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def candidate_thresholds(probs: np.ndarray, n: int = 301) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n)
    th = np.unique(np.quantile(probs, qs))
    th = np.unique(np.concatenate([th, np.array([0.5])]))
    return np.clip(th, 0.0, 1.0)


def select_threshold_by_f1(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for t in candidate_thresholds(probs):
        m = threshold_metrics(probs, labels, float(t))
        if m["f1"] > best["f1"] + 1e-12:
            best = m
        elif abs(m["f1"] - best["f1"]) <= 1e-12 and m["recall"] > best["recall"]:
            best = m
    return best


def binary_logloss_mean(y: np.ndarray, p: np.ndarray) -> float:
    pp = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=np.float64)
    ll = -(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp))
    return float(np.mean(ll))


def make_bin_table(probs: np.ndarray, labels: np.ndarray, best_f1_threshold: float) -> pd.DataFrame:
    bin_ids = np.clip(np.digitize(probs, BIN_EDGES, right=False) - 1, 0, len(BIN_EDGES) - 2)
    total = len(labels)
    rows: List[Dict[str, object]] = []
    for i in range(len(BIN_EDGES) - 1):
        left = float(BIN_EDGES[i])
        right = float(BIN_EDGES[i + 1])
        m = bin_ids == i
        count = int(np.sum(m))
        if count > 0:
            p_bin = probs[m]
            y_bin = labels[m]
            avg_p = float(np.mean(p_bin))
            act = float(np.mean(y_bin))
            gap = abs(avg_p - act)
            ll = binary_logloss_mean(y_bin, p_bin)
        else:
            avg_p = np.nan
            act = np.nan
            gap = np.nan
            ll = np.nan

        contains_t = (best_f1_threshold >= left) and (
            best_f1_threshold < right or (right == 1.0 and best_f1_threshold <= right)
        )
        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "bin_label": f"{int(left*100):02d}-{int(right*100):02d}",
                "count": count,
                "count_ratio": float(count / max(total, 1)),
                "avg_raw_confidence": avg_p,
                "actual_positive_rate": act,
                "abs_gap": gap,
                "bin_logloss": ll,
                "contains_best_f1_threshold": bool(contains_t),
            }
        )
    return pd.DataFrame(rows)


def make_cumulative_table(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    total = len(labels)
    rows: List[Dict[str, object]] = []
    for i in range(len(BIN_EDGES) - 1):
        left = float(BIN_EDGES[i])
        m = probs >= left
        count = int(np.sum(m))
        if count > 0:
            p = probs[m]
            y = labels[m]
            avg_p = float(np.mean(p))
            act = float(np.mean(y))
            gap = abs(avg_p - act)
            ll = binary_logloss_mean(y, p)
        else:
            avg_p = np.nan
            act = np.nan
            gap = np.nan
            ll = np.nan
        rows.append(
            {
                "threshold_left": left,
                "threshold_label": f">={int(left*100):02d}%",
                "cumulative_count": count,
                "cumulative_count_ratio": float(count / max(total, 1)),
                "cumulative_avg_raw_confidence": avg_p,
                "cumulative_actual_positive_rate": act,
                "cumulative_abs_gap": gap,
                "cumulative_logloss": ll,
            }
        )
    return pd.DataFrame(rows)


def _fmt_cell(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        return f"{float(v):.6f}"
    return str(v)


def md_table_from_df(df: pd.DataFrame, cols: List[str]) -> List[str]:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for _, r in df[cols].iterrows():
        rows.append("| " + " | ".join(_fmt_cell(r[c]) for c in cols) + " |")
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_cfg = BuildConfig(
        history_days=args.history_days,
        min_days=args.min_days,
        train_ratio=args.train_ratio,
        calib_ratio=args.calib_ratio,
        max_negative_gap=args.max_negative_gap,
        behavior_prior=args.behavior_prior,
        user_group_count=args.user_group_count,
    )

    (
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        meta_df,
        _aux_calib,
        _aux_test,
        day_stats,
    ) = build_dataset(
        data_folder=Path(args.data_folder),
        cfg=build_cfg,
        max_files=args.max_files,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=24,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    p_calib = model.predict_proba(X_calib)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    best = select_threshold_by_f1(p_calib, y_calib)
    test_at_best = threshold_metrics(p_test, y_test, best["threshold"])
    auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan")
    test_logloss = float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6)))

    bin_df = make_bin_table(p_test, y_test, best["threshold"])
    cum_df = make_cumulative_table(p_test, y_test)

    bin_csv = out_dir / "random_forest_raw_confidence_5pct.csv"
    cum_csv = out_dir / "random_forest_raw_confidence_5pct_cumulative.csv"
    pred_csv = out_dir / "random_forest_raw_test_predictions.csv"
    sum_json = out_dir / "random_forest_raw_confidence_5pct_summary.json"
    md_main = out_dir / "random_forest_raw_confidence_5pct.md"
    md_cum = out_dir / "random_forest_raw_confidence_5pct_cumulative.md"

    bin_df.to_csv(bin_csv, index=False)
    cum_df.to_csv(cum_csv, index=False)

    # Per-sample prediction export (test set): confidence + result.
    if len(meta_df) == len(y_test):
        pred_df = meta_df.copy()
    else:
        pred_df = pd.DataFrame({"row_id": np.arange(len(y_test), dtype=np.int64)})
    pred_df["actual_label"] = y_test.astype(np.int8)
    pred_df["raw_confidence"] = p_test.astype(np.float64)
    pred_df["pred_label_best_f1"] = (p_test >= best["threshold"]).astype(np.int8)
    pred_df["is_correct_best_f1"] = (pred_df["pred_label_best_f1"].to_numpy() == y_test).astype(np.int8)
    bin_idx = np.clip(np.digitize(p_test, BIN_EDGES, right=False) - 1, 0, len(BIN_EDGES) - 2)
    pred_df["confidence_bin_5pct"] = [f"{int(BIN_EDGES[i]*100):02d}-{int(BIN_EDGES[i+1]*100):02d}" for i in bin_idx]
    pred_df["actual_result"] = np.where(pred_df["actual_label"] == 1, "connected", "not_connected")
    pred_df.to_csv(pred_csv, index=False)

    summary = {
        "model": "random_forest_global",
        "threshold_selection": "best_f1_on_calibration",
        "best_f1_threshold_calib": float(best["threshold"]),
        "calib_best_f1": {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
        },
        "test_at_best_f1_threshold": {
            "precision": float(test_at_best["precision"]),
            "recall": float(test_at_best["recall"]),
            "f1": float(test_at_best["f1"]),
        },
        "test_auc": auc,
        "test_logloss": test_logloss,
        "day_stats": day_stats,
    }
    sum_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RandomForest Raw Confidence 5% Bins (Best-F1 Config)",
        "",
        f"- best_f1_threshold (calib): `{best['threshold']:.6f}`",
        (
            "- test @ best_f1_threshold: "
            f"precision=`{test_at_best['precision']:.6f}`, "
            f"recall=`{test_at_best['recall']:.6f}`, "
            f"f1=`{test_at_best['f1']:.6f}`"
        ),
        f"- per-sample predictions: `{pred_csv}`",
        "",
    ]
    lines += md_table_from_df(
        bin_df,
        [
            "bin_label",
            "count",
            "count_ratio",
            "avg_raw_confidence",
            "actual_positive_rate",
            "abs_gap",
            "bin_logloss",
        ],
    )
    lines += [
        "",
        "## Cumulative View (>= Threshold)",
        "",
        "说明：`>=XX%` 表示预测概率至少为 XX% 的样本，按下方桶位累加。",
        "",
    ]
    lines += md_table_from_df(
        cum_df,
        [
            "threshold_label",
            "cumulative_count",
            "cumulative_count_ratio",
            "cumulative_avg_raw_confidence",
            "cumulative_actual_positive_rate",
            "cumulative_abs_gap",
            "cumulative_logloss",
        ],
    )
    md_main.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines2 = [
        "# RandomForest Raw Confidence Cumulative 5% Table",
        "",
        "说明：`>=XX%` 表示预测概率“至少为 XX%”的样本集合（从该桶累加到 100%）。",
        "",
    ]
    lines2 += md_table_from_df(
        cum_df,
        [
            "threshold_label",
            "cumulative_count",
            "cumulative_count_ratio",
            "cumulative_avg_raw_confidence",
            "cumulative_actual_positive_rate",
            "cumulative_abs_gap",
            "cumulative_logloss",
        ],
    )
    md_cum.write_text("\n".join(lines2) + "\n", encoding="utf-8")

    print(f"[done] {bin_csv}")
    print(f"[done] {cum_csv}")
    print(f"[done] {pred_csv}")
    print(f"[done] {sum_json}")
    print(f"[done] {md_main}")
    print(f"[done] {md_cum}")


if __name__ == "__main__":
    main()
