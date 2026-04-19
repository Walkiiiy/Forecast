#!/usr/bin/env python3
"""
Benchmark global models for 30-min connection prediction.

Focus:
- Evaluate core model predictive quality (no high-tail mapping tricks).
- Use one global model across all users.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from confidence_forecast_extreme_rebuild import BuildConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global model benchmark (accuracy-focused).")
    parser.add_argument("--data-folder", type=str, default="data/top350")
    parser.add_argument("--output-dir", type=str, default="outputs/global_model_accuracy_benchmark_v1")
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


def topk_precision(probs: np.ndarray, labels: np.ndarray, frac: float) -> float:
    n = len(labels)
    k = max(1, int(round(n * frac)))
    idx = np.argsort(-probs)[:k]
    return float(np.mean(labels[idx]))


def topk_recall(probs: np.ndarray, labels: np.ndarray, frac: float) -> float:
    n_pos = int(np.sum(labels))
    if n_pos <= 0:
        return 0.0
    n = len(labels)
    k = max(1, int(round(n * frac)))
    idx = np.argsort(-probs)[:k]
    tp = int(np.sum(labels[idx]))
    return float(tp / n_pos)


def binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    pred_pos_rate = float(np.mean(y_pred))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_rate": pred_pos_rate,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def threshold_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.int8)
    out = binary_prf(labels, pred)
    out["threshold"] = float(threshold)
    return out


def candidate_thresholds(probs: np.ndarray, n: int = 301) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n)
    th = np.unique(np.quantile(probs, qs))
    th = np.unique(np.concatenate([th, np.array([0.5])]))
    return np.clip(th, 0.0, 1.0)


def select_threshold_by_f1(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best: Dict[str, float] = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for t in candidate_thresholds(probs):
        m = threshold_metrics(probs, labels, float(t))
        if m["f1"] > best["f1"] + 1e-12:
            best = m
        elif abs(m["f1"] - best["f1"]) <= 1e-12 and m["recall"] > best["recall"]:
            best = m
    return best


def select_threshold_by_recall_with_precision_floor(
    probs: np.ndarray,
    labels: np.ndarray,
    min_precision: float,
) -> Optional[Dict[str, float]]:
    best: Optional[Dict[str, float]] = None
    for t in candidate_thresholds(probs):
        m = threshold_metrics(probs, labels, float(t))
        if m["precision"] + 1e-12 < min_precision:
            continue
        if best is None or (m["recall"] > best["recall"] + 1e-12):
            best = m
        elif best is not None and abs(m["recall"] - best["recall"]) <= 1e-12 and m["f1"] > best["f1"]:
            best = m
    return best


def eval_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "samples": float(len(labels)),
        "positive_rate": float(np.mean(labels)),
        "auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "logloss": float(log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(labels, probs)),
        "top_0_5pct_precision": topk_precision(probs, labels, 0.005),
        "top_1pct_precision": topk_precision(probs, labels, 0.01),
        "top_2pct_precision": topk_precision(probs, labels, 0.02),
        "top_0_5pct_recall": topk_recall(probs, labels, 0.005),
        "top_1pct_recall": topk_recall(probs, labels, 0.01),
        "top_2pct_recall": topk_recall(probs, labels, 0.02),
    }


def fit_histgb(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    pos_ratio = float(np.mean(y_train))
    pos_weight = max(1.0, (1.0 - pos_ratio) / max(pos_ratio, 1e-8))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_depth=8,
        min_samples_leaf=80,
        l2_regularization=1.0,
        categorical_features=[X_train.shape[1] - 1],
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    pos_ratio = float(np.mean(y_train))
    scale_pos_weight = max(1.0, (1.0 - pos_ratio) / max(pos_ratio, 1e-8))
    model = XGBClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=24,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(name: str, model, X_calib, y_calib, X_test, y_test) -> Dict[str, object]:
    p_calib_raw = model.predict_proba(X_calib)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    calib_raw = eval_metrics(p_calib_raw, y_calib)
    test_raw = eval_metrics(p_test_raw, y_test)
    calib_raw_best_f1 = select_threshold_by_f1(p_calib_raw, y_calib)
    test_raw_at_best_f1 = threshold_metrics(p_test_raw, y_test, calib_raw_best_f1["threshold"])
    calib_raw_recall_p70 = select_threshold_by_recall_with_precision_floor(p_calib_raw, y_calib, 0.70)
    if calib_raw_recall_p70 is not None:
        test_raw_recall_p70 = threshold_metrics(p_test_raw, y_test, calib_raw_recall_p70["threshold"])
    else:
        test_raw_recall_p70 = None

    # Also report isotonic-calibrated metrics for fair probability quality comparison.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib_raw, y_calib)
    p_calib_iso = iso.predict(p_calib_raw)
    p_test_iso = iso.predict(p_test_raw)
    calib_iso = eval_metrics(p_calib_iso, y_calib)
    test_iso = eval_metrics(p_test_iso, y_test)
    calib_iso_best_f1 = select_threshold_by_f1(p_calib_iso, y_calib)
    test_iso_at_best_f1 = threshold_metrics(p_test_iso, y_test, calib_iso_best_f1["threshold"])
    calib_iso_recall_p70 = select_threshold_by_recall_with_precision_floor(p_calib_iso, y_calib, 0.70)
    if calib_iso_recall_p70 is not None:
        test_iso_recall_p70 = threshold_metrics(p_test_iso, y_test, calib_iso_recall_p70["threshold"])
    else:
        test_iso_recall_p70 = None

    return {
        "model": name,
        "metrics_calib_raw": calib_raw,
        "metrics_test_raw": test_raw,
        "calib_raw_best_f1": calib_raw_best_f1,
        "test_raw_at_best_f1": test_raw_at_best_f1,
        "calib_raw_recall_p70": calib_raw_recall_p70,
        "test_raw_recall_p70": test_raw_recall_p70,
        "metrics_calib_iso": calib_iso,
        "metrics_test_iso": test_iso,
        "calib_iso_best_f1": calib_iso_best_f1,
        "test_iso_at_best_f1": test_iso_at_best_f1,
        "calib_iso_recall_p70": calib_iso_recall_p70,
        "test_iso_recall_p70": test_iso_recall_p70,
    }


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
        _meta_test_df,
        _aux_calib,
        _aux_test,
        day_stats,
    ) = build_dataset(
        data_folder=Path(args.data_folder),
        cfg=build_cfg,
        max_files=args.max_files,
    )

    results: List[Dict[str, object]] = []

    model_defs: List[Tuple[str, object]] = [
        ("histgb_global", fit_histgb(X_train, y_train, args.seed)),
        ("xgboost_global", fit_xgboost(X_train, y_train, args.seed)),
        ("random_forest_global", fit_random_forest(X_train, y_train, args.seed)),
    ]

    for name, model in model_defs:
        print(f"[benchmark] evaluating {name}")
        results.append(evaluate_model(name, model, X_calib, y_calib, X_test, y_test))

    rows = []
    for r in results:
        m = r["metrics_test_raw"]
        mi = r["metrics_test_iso"]
        raw_best = r["test_raw_at_best_f1"]
        iso_best = r["test_iso_at_best_f1"]
        raw_p70 = r["test_raw_recall_p70"]
        rows.append(
            {
                "model": r["model"],
                "raw_auc": m["auc"],
                "raw_pr_auc": m["pr_auc"],
                "raw_logloss": m["logloss"],
                "raw_brier": m["brier"],
                "raw_top_1pct_precision": m["top_1pct_precision"],
                "raw_top_1pct_recall": m["top_1pct_recall"],
                "raw_bestf1_precision": raw_best["precision"],
                "raw_bestf1_recall": raw_best["recall"],
                "raw_bestf1_f1": raw_best["f1"],
                "raw_bestf1_threshold": raw_best["threshold"],
                "raw_recall_at_p70": raw_p70["recall"] if raw_p70 is not None else np.nan,
                "raw_precision_at_p70_threshold": raw_p70["precision"] if raw_p70 is not None else np.nan,
                "iso_auc": mi["auc"],
                "iso_pr_auc": mi["pr_auc"],
                "iso_logloss": mi["logloss"],
                "iso_brier": mi["brier"],
                "iso_top_1pct_precision": mi["top_1pct_precision"],
                "iso_top_1pct_recall": mi["top_1pct_recall"],
                "iso_bestf1_precision": iso_best["precision"],
                "iso_bestf1_recall": iso_best["recall"],
                "iso_bestf1_f1": iso_best["f1"],
                "iso_bestf1_threshold": iso_best["threshold"],
            }
        )
    table_df = pd.DataFrame(rows).sort_values(
        by=["raw_bestf1_f1", "raw_bestf1_recall", "raw_auc"],
        ascending=[False, False, False],
    )
    table_df.to_csv(out_dir / "model_benchmark_table.csv", index=False)

    summary = {
        "build_config": vars(build_cfg),
        "day_stats": day_stats,
        "results": results,
    }
    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    best_raw = table_df.iloc[0].to_dict()
    lines = [
        "# Global Model Accuracy Benchmark",
        "",
        "## Best By Recall-Aware F1 (Raw)",
        "",
        f"- model: `{best_raw['model']}`",
        f"- raw_bestf1_f1: `{best_raw['raw_bestf1_f1']:.6f}`",
        f"- raw_bestf1_precision: `{best_raw['raw_bestf1_precision']:.6f}`",
        f"- raw_bestf1_recall: `{best_raw['raw_bestf1_recall']:.6f}`",
        f"- raw_bestf1_threshold: `{best_raw['raw_bestf1_threshold']:.6f}`",
        f"- raw_recall_at_p70: `{best_raw['raw_recall_at_p70']:.6f}`",
        "",
        "## Test Set Comparison",
        "",
        "| model | raw_auc | raw_pr_auc | raw_logloss | raw_bestf1_precision | raw_bestf1_recall | raw_bestf1_f1 | raw_recall_at_p70 | iso_logloss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table_df.iterrows():
        lines.append(
            "| "
            + f"{row['model']} | {row['raw_auc']:.6f} | {row['raw_pr_auc']:.6f} | "
            + f"{row['raw_logloss']:.6f} | {row['raw_bestf1_precision']:.6f} | "
            + f"{row['raw_bestf1_recall']:.6f} | {row['raw_bestf1_f1']:.6f} | "
            + f"{row['raw_recall_at_p70']:.6f} | {row['iso_logloss']:.6f} |"
        )

    with open(out_dir / "model_benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[done] saved benchmark to", out_dir)
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
