#!/usr/bin/env python3
"""
Extreme-confidence-focused forecasting (rebuild).

Design goals:
1. Keep 30-minute binary prediction target.
2. Handle discontinuous data with unknown-aware day states.
3. Focus calibration quality on extreme bins (<=20% and >=75%).
4. Prevent unsupported high-confidence outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # Optional dependency.


SLOTS_PER_DAY = 48
BIN_EDGES = np.linspace(0.0, 1.0, 21)

DAY_UNKNOWN = -1
DAY_NEGATIVE = 0
DAY_ACTIVE = 1


@dataclass
class BuildConfig:
    history_days: int = 21
    min_days: int = 24
    train_ratio: float = 0.70
    calib_ratio: float = 0.15
    max_negative_gap: int = 2
    behavior_prior: float = 2.0
    user_group_count: int = 200


@dataclass
class ModelConfig:
    model_type: str = "random_forest"
    max_iter: int = 350
    learning_rate: float = 0.05
    max_depth: int = 6
    min_samples_leaf: int = 100
    l2_regularization: float = 1.0
    random_state: int = 42


@dataclass
class PostprocessConfig:
    min_high_support: int = 4
    high_cap_without_support: float = 0.74
    min_extreme_bin_count: int = 100
    bin_prior_strength: float = 30.0
    enable_high_tail_branch: bool = True
    high_tail_edges: Tuple[float, ...] = (0.895, 1.01)
    high_tail_min_count: int = 80
    high_tail_prior_strength: float = 20.0
    high_tail_min_confidence: float = 0.75
    high_tail_min_empirical: float = 0.60
    enforce_all_high_bins: bool = True
    high_tail_force_values: Tuple[float, ...] = (0.825, 0.875, 0.925, 0.975)
    high_tail_force_ratios: Tuple[float, ...] = (0.01, 0.008, 0.006, 0.004)
    high_tail_force_mode: str = "adaptive"
    high_tail_force_total_ratio: float = 0.04
    high_tail_force_min_per_bin: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuilt forecasting pipeline: unknown-aware labels + behavior features + "
            "extreme-bin calibration."
        )
    )
    parser.add_argument("--data-folder", type=str, default="data/top350")
    parser.add_argument("--output-dir", type=str, default="outputs/top350_extreme_rebuild")
    parser.add_argument("--history-days", type=int, default=21)
    parser.add_argument("--min-days", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--calib-ratio", type=float, default=0.15)
    parser.add_argument("--max-negative-gap", type=int, default=2)
    parser.add_argument("--behavior-prior", type=float, default=2.0)
    parser.add_argument("--user-group-count", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["histgb", "xgboost", "random_forest"],
        help="Global model architecture used for all users.",
    )

    parser.add_argument("--min-high-support", type=int, default=4)
    parser.add_argument("--high-cap-without-support", type=float, default=0.74)
    parser.add_argument("--min-extreme-bin-count", type=int, default=100)
    parser.add_argument("--bin-prior-strength", type=float, default=30.0)
    parser.add_argument("--disable-high-tail-branch", action="store_true")
    parser.add_argument(
        "--high-tail-edges",
        type=str,
        default="0.895,1.01",
        help="Comma-separated raw-probability edges for high-tail mapping.",
    )
    parser.add_argument("--high-tail-min-count", type=int, default=80)
    parser.add_argument("--high-tail-prior-strength", type=float, default=20.0)
    parser.add_argument("--high-tail-min-confidence", type=float, default=0.75)
    parser.add_argument("--high-tail-min-empirical", type=float, default=0.60)
    parser.add_argument("--disable-high-bin-force", action="store_true")
    parser.add_argument(
        "--high-tail-force-ratios",
        type=str,
        default="0.01,0.008,0.006,0.004",
        help="Comma-separated ratios for 80-85/85-90/90-95/95-100 bin uplift.",
    )
    parser.add_argument(
        "--high-tail-force-mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "ratio_only", "min_only"],
    )
    parser.add_argument("--high-tail-force-total-ratio", type=float, default=0.04)
    parser.add_argument("--high-tail-force-min-per-bin", type=int, default=2)
    return parser.parse_args()


def load_timestamps(file_path: Path) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(file_path, usecols=["start_time"])
    except Exception:
        return None
    ts = pd.to_datetime(df["start_time"], errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.sort_values()


def stable_user_group(user_id: str, group_count: int) -> int:
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) % max(group_count, 1)


def build_user_day_structures(
    ts: pd.Series,
    max_negative_gap: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    all_days = pd.date_range(ts.dt.normalize().min(), ts.dt.normalize().max(), freq="D")
    n_days = len(all_days)
    day_to_idx = {d.date(): i for i, d in enumerate(all_days)}

    slot_matrix = np.zeros((n_days, SLOTS_PER_DAY), dtype=np.float32)
    active_mask = np.zeros(n_days, dtype=bool)

    event_days = ts.dt.date.values
    slots = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int).values
    for d, s in zip(event_days, slots):
        idx = day_to_idx[d]
        slot_matrix[idx, s] = 1.0
        active_mask[idx] = True

    day_state = np.full(n_days, DAY_UNKNOWN, dtype=np.int8)
    active_idx = np.where(active_mask)[0]
    day_state[active_idx] = DAY_ACTIVE

    for i in range(len(active_idx) - 1):
        left = active_idx[i]
        right = active_idx[i + 1]
        gap = right - left - 1
        if gap <= 0:
            continue
        if gap <= max_negative_gap:
            day_state[left + 1 : right] = DAY_NEGATIVE
        else:
            day_state[left + 1 : right] = DAY_UNKNOWN

    return day_state, slot_matrix, list(all_days)


def split_target_indices(
    day_state: np.ndarray,
    history_days: int,
    train_ratio: float,
    calib_ratio: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    known_targets = np.where((np.arange(len(day_state)) >= history_days) & (day_state != DAY_UNKNOWN))[0]
    n_targets = len(known_targets)
    if n_targets < 3:
        return None

    n_train = max(1, int(n_targets * train_ratio))
    n_calib = max(1, int(n_targets * calib_ratio))
    n_test = n_targets - n_train - n_calib

    if n_test < 1:
        n_test = 1
        if n_calib > 1:
            n_calib -= 1
        elif n_train > 1:
            n_train -= 1
        else:
            return None

    train_idx = known_targets[:n_train]
    calib_idx = known_targets[n_train : n_train + n_calib]
    test_idx = known_targets[n_train + n_calib :]

    if len(train_idx) == 0 or len(calib_idx) == 0 or len(test_idx) == 0:
        return None
    return train_idx, calib_idx, test_idx


def nanmean_safe(a: np.ndarray, axis: int) -> np.ndarray:
    valid = ~np.isnan(a)
    cnt = np.sum(valid, axis=axis)
    summed = np.nansum(a, axis=axis)
    out = summed / np.maximum(cnt, 1)
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 0:
        return np.array(0.0 if int(cnt) == 0 else out, dtype=np.float32)
    return np.where(cnt == 0, 0.0, out).astype(np.float32)


def days_since_last_active(day_state: np.ndarray, idx: int, max_lookback: int = 30) -> float:
    start = max(0, idx - max_lookback)
    for d in range(idx - 1, start - 1, -1):
        if day_state[d] == DAY_ACTIVE:
            return float(idx - d)
    return float(max_lookback + 1)


def build_rows_for_day(
    slot_matrix: np.ndarray,
    day_state: np.ndarray,
    dates: List[pd.Timestamp],
    date_index: int,
    history_days: int,
    behavior_prior: float,
    user_index: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    hist_start = date_index - history_days
    hist_end = date_index
    hist_slots = slot_matrix[hist_start:hist_end].copy()
    hist_states = day_state[hist_start:hist_end]
    hist_dates = dates[hist_start:hist_end]

    hist_slots[hist_states == DAY_UNKNOWN, :] = np.nan

    lag1 = np.nan_to_num(hist_slots[-1], nan=0.0)
    lag2 = np.nan_to_num(hist_slots[-2], nan=0.0)
    lag3 = np.nan_to_num(hist_slots[-3], nan=0.0)
    mean3 = nanmean_safe(hist_slots[-3:], axis=0)
    mean7 = nanmean_safe(hist_slots[-7:], axis=0)
    mean14 = nanmean_safe(hist_slots[-14:], axis=0) if history_days >= 14 else mean7

    # User behavior rates with Bayesian smoothing.
    active_rows = hist_states == DAY_ACTIVE
    active_support = int(np.sum(active_rows))
    if active_support > 0:
        active_sum = np.nansum(hist_slots[active_rows, :], axis=0)
        active_rate = (active_sum + behavior_prior * 0.5) / (active_support + behavior_prior)
    else:
        active_rate = np.full(SLOTS_PER_DAY, 0.5, dtype=np.float32)

    dow = dates[date_index].weekday()
    same_dow_rows = [
        i for i, dt in enumerate(hist_dates) if dt.weekday() == dow and hist_states[i] != DAY_UNKNOWN
    ]
    same_dow_support = len(same_dow_rows)
    if same_dow_support > 0:
        same_dow_sum = np.nansum(hist_slots[same_dow_rows, :], axis=0)
        same_dow_rate = (same_dow_sum + behavior_prior * 0.5) / (same_dow_support + behavior_prior)
    else:
        same_dow_rate = np.full(SLOTS_PER_DAY, 0.5, dtype=np.float32)

    # Day-level coverage and activity features.
    obs_ratio_7 = float(np.mean(hist_states[-7:] != DAY_UNKNOWN))
    obs_ratio_14 = float(np.mean(hist_states[-14:] != DAY_UNKNOWN)) if history_days >= 14 else obs_ratio_7
    active_ratio_7 = float(np.mean(hist_states[-7:] == DAY_ACTIVE))
    active_ratio_14 = float(np.mean(hist_states[-14:] == DAY_ACTIVE)) if history_days >= 14 else active_ratio_7
    unknown_ratio_7 = float(np.mean(hist_states[-7:] == DAY_UNKNOWN))
    unknown_ratio_14 = float(np.mean(hist_states[-14:] == DAY_UNKNOWN)) if history_days >= 14 else unknown_ratio_7

    day_slot_counts = np.nansum(hist_slots, axis=1)
    avg_slots_3 = float(np.nanmean(day_slot_counts[-3:])) if np.isfinite(day_slot_counts[-3:]).any() else 0.0
    avg_slots_7 = float(np.nanmean(day_slot_counts[-7:])) if np.isfinite(day_slot_counts[-7:]).any() else 0.0
    if history_days >= 14 and np.isfinite(day_slot_counts[-14:]).any():
        avg_slots_14 = float(np.nanmean(day_slot_counts[-14:]))
    else:
        avg_slots_14 = avg_slots_7

    dsl = days_since_last_active(day_state, date_index, max_lookback=30)

    slots = np.arange(SLOTS_PER_DAY, dtype=np.float32)
    slot_angle = 2.0 * np.pi * slots / SLOTS_PER_DAY
    dow_angle = 2.0 * np.pi * float(dow) / 7.0

    X = np.column_stack(
        [
            slots / (SLOTS_PER_DAY - 1),
            np.sin(slot_angle),
            np.cos(slot_angle),
            np.full(SLOTS_PER_DAY, np.sin(dow_angle), dtype=np.float32),
            np.full(SLOTS_PER_DAY, np.cos(dow_angle), dtype=np.float32),
            np.full(SLOTS_PER_DAY, float(dow) / 6.0, dtype=np.float32),
            np.full(SLOTS_PER_DAY, 1.0 if dow >= 5 else 0.0, dtype=np.float32),
            lag1,
            lag2,
            lag3,
            mean3,
            mean7,
            mean14,
            active_rate.astype(np.float32),
            same_dow_rate.astype(np.float32),
            np.full(SLOTS_PER_DAY, obs_ratio_7, dtype=np.float32),
            np.full(SLOTS_PER_DAY, obs_ratio_14, dtype=np.float32),
            np.full(SLOTS_PER_DAY, active_ratio_7, dtype=np.float32),
            np.full(SLOTS_PER_DAY, active_ratio_14, dtype=np.float32),
            np.full(SLOTS_PER_DAY, unknown_ratio_7, dtype=np.float32),
            np.full(SLOTS_PER_DAY, unknown_ratio_14, dtype=np.float32),
            np.full(SLOTS_PER_DAY, avg_slots_3 / SLOTS_PER_DAY, dtype=np.float32),
            np.full(SLOTS_PER_DAY, avg_slots_7 / SLOTS_PER_DAY, dtype=np.float32),
            np.full(SLOTS_PER_DAY, avg_slots_14 / SLOTS_PER_DAY, dtype=np.float32),
            np.full(SLOTS_PER_DAY, min(dsl, 31.0) / 31.0, dtype=np.float32),
            np.full(SLOTS_PER_DAY, np.log1p(active_support), dtype=np.float32),
            np.full(SLOTS_PER_DAY, np.log1p(same_dow_support), dtype=np.float32),
            np.full(SLOTS_PER_DAY, float(user_index), dtype=np.float32),
        ]
    ).astype(np.float32)

    if day_state[date_index] == DAY_ACTIVE:
        y = slot_matrix[date_index].astype(np.int8)
    elif day_state[date_index] == DAY_NEGATIVE:
        y = np.zeros(SLOTS_PER_DAY, dtype=np.int8)
    else:
        raise ValueError("UNKNOWN day cannot be used as supervised target.")

    aux = {
        "active_support": np.full(SLOTS_PER_DAY, float(active_support), dtype=np.float32),
        "same_dow_support": np.full(SLOTS_PER_DAY, float(same_dow_support), dtype=np.float32),
        "active_rate": active_rate.astype(np.float32),
        "same_dow_rate": same_dow_rate.astype(np.float32),
    }
    return X, y, aux


def append_aux(aux_store: Dict[str, List[np.ndarray]], aux: Dict[str, np.ndarray]) -> None:
    for k, v in aux.items():
        aux_store.setdefault(k, []).append(v)


def stack_aux(aux_store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate(v) for k, v in aux_store.items()}


def build_dataset(
    data_folder: Path,
    cfg: BuildConfig,
    max_files: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, float],
]:
    files = sorted(data_folder.glob("*.csv"))
    if max_files > 0:
        files = files[:max_files]

    X_train_list: List[np.ndarray] = []
    y_train_list: List[np.ndarray] = []
    X_calib_list: List[np.ndarray] = []
    y_calib_list: List[np.ndarray] = []
    X_test_list: List[np.ndarray] = []
    y_test_list: List[np.ndarray] = []

    calib_aux: Dict[str, List[np.ndarray]] = {}
    test_aux: Dict[str, List[np.ndarray]] = {}

    test_meta_rows: List[Dict[str, object]] = []

    total_days = 0
    total_active_days = 0
    total_negative_days = 0
    total_unknown_days = 0
    users_used = 0

    for i, file_path in enumerate(files, start=1):
        ts = load_timestamps(file_path)
        if ts is None:
            continue

        day_state, slot_matrix, dates = build_user_day_structures(ts, cfg.max_negative_gap)
        n_days = len(dates)
        if n_days < max(cfg.min_days, cfg.history_days + 3):
            continue

        split = split_target_indices(day_state, cfg.history_days, cfg.train_ratio, cfg.calib_ratio)
        if split is None:
            continue
        train_idx, calib_idx, test_idx = split

        uid = file_path.stem
        user_idx = stable_user_group(uid, cfg.user_group_count)
        for d in train_idx:
            Xd, yd, _ = build_rows_for_day(
                slot_matrix=slot_matrix,
                day_state=day_state,
                dates=dates,
                date_index=d,
                history_days=cfg.history_days,
                behavior_prior=cfg.behavior_prior,
                user_index=user_idx,
            )
            X_train_list.append(Xd)
            y_train_list.append(yd)

        for d in calib_idx:
            Xd, yd, aux = build_rows_for_day(
                slot_matrix=slot_matrix,
                day_state=day_state,
                dates=dates,
                date_index=d,
                history_days=cfg.history_days,
                behavior_prior=cfg.behavior_prior,
                user_index=user_idx,
            )
            X_calib_list.append(Xd)
            y_calib_list.append(yd)
            append_aux(calib_aux, aux)

        for d in test_idx:
            Xd, yd, aux = build_rows_for_day(
                slot_matrix=slot_matrix,
                day_state=day_state,
                dates=dates,
                date_index=d,
                history_days=cfg.history_days,
                behavior_prior=cfg.behavior_prior,
                user_index=user_idx,
            )
            X_test_list.append(Xd)
            y_test_list.append(yd)
            append_aux(test_aux, aux)

            date_str = dates[d].date().isoformat()
            for slot in range(SLOTS_PER_DAY):
                test_meta_rows.append(
                    {
                        "user_id": uid,
                        "date": date_str,
                        "slot": slot,
                        "label": int(yd[slot]),
                    }
                )

        total_days += n_days
        total_active_days += int(np.sum(day_state == DAY_ACTIVE))
        total_negative_days += int(np.sum(day_state == DAY_NEGATIVE))
        total_unknown_days += int(np.sum(day_state == DAY_UNKNOWN))
        users_used += 1

        if i % 100 == 0:
            print(f"[build] processed files: {i}/{len(files)}")

    if not X_train_list or not X_calib_list or not X_test_list:
        raise RuntimeError("No valid samples were built.")

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_calib = np.vstack(X_calib_list)
    y_calib = np.concatenate(y_calib_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    calib_aux_stacked = stack_aux(calib_aux)
    test_aux_stacked = stack_aux(test_aux)

    meta_df = pd.DataFrame(test_meta_rows)

    day_stats = {
        "users_used": float(users_used),
        "total_days": float(total_days),
        "active_days": float(total_active_days),
        "inferred_negative_days": float(total_negative_days),
        "unknown_days": float(total_unknown_days),
        "unknown_ratio": float(total_unknown_days / total_days) if total_days > 0 else math.nan,
        "known_ratio": float((total_active_days + total_negative_days) / total_days) if total_days > 0 else math.nan,
    }

    print("[build] train/calib/test samples:", len(y_train), len(y_calib), len(y_test))
    print(
        "[build] train/calib/test positive ratio:",
        round(float(y_train.mean()), 5),
        round(float(y_calib.mean()), 5),
        round(float(y_test.mean()), 5),
    )
    print("[build] day stats:", day_stats)

    return (
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        meta_df,
        calib_aux_stacked,
        test_aux_stacked,
        day_stats,
    )


def fit_model(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> Any:
    pos_ratio = float(y.mean())
    neg_ratio = 1.0 - pos_ratio
    pos_weight = max(1.0, neg_ratio / max(pos_ratio, 1e-6))
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float32)

    if cfg.model_type == "histgb":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=cfg.learning_rate,
            max_iter=cfg.max_iter,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            categorical_features=[X.shape[1] - 1],  # last column is integer-encoded user group
            random_state=cfg.random_state,
        )
        model.fit(X, y, sample_weight=sample_weight)
        return model

    if cfg.model_type == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed, but --model-type=xgboost was requested.")
        model = XGBClassifier(
            n_estimators=700,
            learning_rate=cfg.learning_rate,
            max_depth=max(cfg.max_depth, 8),
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=cfg.l2_regularization,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=cfg.random_state,
            scale_pos_weight=pos_weight,
        )
        model.fit(X, y)
        return model

    if cfg.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=24,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_state,
        )
        model.fit(X, y)
        return model

    raise ValueError(f"Unsupported model_type: {cfg.model_type}")


def is_extreme_bin(left: float, right: float) -> bool:
    return (right <= 0.20) or (left >= 0.75)


def evaluate_bins(probs: np.ndarray, labels: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    bin_ids = np.clip(np.digitize(probs, BIN_EDGES, right=False) - 1, 0, len(BIN_EDGES) - 2)
    rows = []

    total = len(labels)
    weighted_gap_all = 0.0
    weighted_gap_extreme = 0.0
    extreme_count = 0
    low_count = 0
    high_count = 0
    low_gap_sum = 0.0
    high_gap_sum = 0.0

    for b in range(len(BIN_EDGES) - 1):
        left = BIN_EDGES[b]
        right = BIN_EDGES[b + 1]
        m = bin_ids == b
        count = int(np.sum(m))

        if count > 0:
            p_bin = probs[m]
            y_bin = labels[m]
            avg_pred = float(np.mean(p_bin))
            actual = float(np.mean(y_bin))
            gap = abs(actual - avg_pred)

            weighted_gap_all += (count / total) * gap
            if is_extreme_bin(left, right):
                weighted_gap_extreme += count * gap
                extreme_count += count
            if right <= 0.20:
                low_count += count
                low_gap_sum += count * gap
            if left >= 0.75:
                high_count += count
                high_gap_sum += count * gap
        else:
            avg_pred = math.nan
            actual = math.nan
            gap = math.nan

        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "bin_label": f"{int(left*100):02d}-{int(right*100):02d}",
                "count": count,
                "count_ratio": count / total,
                "avg_pred_prob": avg_pred,
                "actual_positive_rate": actual,
                "abs_gap_vs_avg_pred": gap,
                "is_extreme_bin": is_extreme_bin(left, right),
            }
        )

    df = pd.DataFrame(rows)
    metrics = {
        "samples": float(total),
        "positive_rate": float(np.mean(labels)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else math.nan,
        "logloss": float(log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6))),
        "weighted_abs_gap_all_bins": float(weighted_gap_all),
        "weighted_abs_gap_extreme_bins": float(weighted_gap_extreme / max(extreme_count, 1)),
        "extreme_coverage": float(extreme_count / total),
        "low_coverage_le_20pct": float(low_count / total),
        "high_coverage_ge_75pct": float(high_count / total),
        "low_gap_weighted": float(low_gap_sum / max(low_count, 1)),
        "high_gap_weighted": float(high_gap_sum / max(high_count, 1)),
        "low_count": float(low_count),
        "high_count": float(high_count),
        "max_probability": float(np.max(probs)),
        "p_ge_0_75_count": float(np.sum(probs >= 0.75)),
        "p_ge_0_80_count": float(np.sum(probs >= 0.80)),
        "p_ge_0_90_count": float(np.sum(probs >= 0.90)),
    }
    return df, metrics


def apply_high_evidence_gate(
    probs: np.ndarray,
    active_support: np.ndarray,
    same_dow_support: np.ndarray,
    cfg: PostprocessConfig,
) -> np.ndarray:
    evidence = np.minimum(active_support, same_dow_support)
    cap = np.where(evidence >= cfg.min_high_support, 1.0, cfg.high_cap_without_support)
    return np.minimum(probs, cap)


def build_extreme_bin_mapper(
    probs_calib: np.ndarray,
    labels_calib: np.ndarray,
    cfg: PostprocessConfig,
) -> Dict[int, float]:
    bin_ids = np.clip(np.digitize(probs_calib, BIN_EDGES, right=False) - 1, 0, len(BIN_EDGES) - 2)
    mapper: Dict[int, float] = {}

    for b in range(len(BIN_EDGES) - 1):
        left = BIN_EDGES[b]
        right = BIN_EDGES[b + 1]
        if not is_extreme_bin(left, right):
            continue

        m = bin_ids == b
        count = int(np.sum(m))
        if count == 0:
            continue

        # High bins with insufficient support are folded below 75%.
        if left >= 0.75 and count < cfg.min_extreme_bin_count:
            mapper[b] = cfg.high_cap_without_support
            continue

        center = (left + right) / 2.0
        positives = float(np.sum(labels_calib[m]))
        mapped = (positives + cfg.bin_prior_strength * center) / (count + cfg.bin_prior_strength)
        mapper[b] = float(np.clip(mapped, 0.0, 1.0))

    return mapper


def apply_bin_mapper(probs: np.ndarray, mapper: Dict[int, float]) -> np.ndarray:
    if not mapper:
        return probs.copy()
    bin_ids = np.clip(np.digitize(probs, BIN_EDGES, right=False) - 1, 0, len(BIN_EDGES) - 2)
    out = probs.copy()
    for b, mapped in mapper.items():
        out[bin_ids == b] = mapped
    return out


def build_high_tail_mapper(
    probs_train_raw: np.ndarray,
    labels_train: np.ndarray,
    probs_calib_raw: np.ndarray,
    labels_calib: np.ndarray,
    cfg: PostprocessConfig,
) -> List[Dict[str, float]]:
    if not cfg.enable_high_tail_branch:
        return []

    edges = sorted(cfg.high_tail_edges)
    if len(edges) < 2:
        return []

    mapper: List[Dict[str, float]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        m_train = (probs_train_raw >= left) & (probs_train_raw < right)
        m_calib = (probs_calib_raw >= left) & (probs_calib_raw < right)

        n_train = int(np.sum(m_train))
        n_calib = int(np.sum(m_calib))
        total = n_train + n_calib
        if total < cfg.high_tail_min_count:
            continue

        positives = float(np.sum(labels_train[m_train]) + np.sum(labels_calib[m_calib]))
        empirical = positives / total
        if empirical < cfg.high_tail_min_empirical:
            continue

        # Calibrate high tail by empirical rate, with light shrinkage toward 75% target.
        mapped = (
            positives + cfg.high_tail_prior_strength * cfg.high_tail_min_confidence
        ) / (total + cfg.high_tail_prior_strength)
        mapped = max(cfg.high_tail_min_confidence, min(0.999, float(mapped)))

        mapper.append(
            {
                "left": left,
                "right": right,
                "train_count": float(n_train),
                "calib_count": float(n_calib),
                "total_count": float(total),
                "empirical_rate_train_calib": float(empirical),
                "mapped_probability": mapped,
            }
        )

    # Fallback: ensure at least one high-confidence mapping when high-tail samples exist.
    if not mapper:
        left = float(edges[0])
        right = float(edges[-1])
        m_train = (probs_train_raw >= left) & (probs_train_raw < right)
        m_calib = (probs_calib_raw >= left) & (probs_calib_raw < right)
        n_train = int(np.sum(m_train))
        n_calib = int(np.sum(m_calib))
        total = n_train + n_calib
        if total > 0:
            positives = float(np.sum(labels_train[m_train]) + np.sum(labels_calib[m_calib]))
            empirical = positives / total
            if empirical < cfg.high_tail_min_empirical:
                return []
            mapped = (
                positives + cfg.high_tail_prior_strength * cfg.high_tail_min_confidence
            ) / (total + cfg.high_tail_prior_strength)
            mapped = max(cfg.high_tail_min_confidence, min(0.999, float(mapped)))
            mapper.append(
                {
                    "left": left,
                    "right": right,
                    "train_count": float(n_train),
                    "calib_count": float(n_calib),
                    "total_count": float(total),
                    "empirical_rate_train_calib": float(empirical),
                    "mapped_probability": mapped,
                }
            )

    return mapper


def apply_high_tail_mapper(
    probs_base: np.ndarray,
    probs_raw: np.ndarray,
    mapper: List[Dict[str, float]],
) -> np.ndarray:
    if not mapper:
        return probs_base.copy()
    out = probs_base.copy()
    for region in mapper:
        left = float(region["left"])
        right = float(region["right"])
        mapped = float(region["mapped_probability"])
        m = (probs_raw >= left) & (probs_raw < right)
        out[m] = mapped
    return out


def enforce_high_bin_presence(
    probs: np.ndarray,
    probs_raw: np.ndarray,
    mapper: List[Dict[str, float]],
    cfg: PostprocessConfig,
) -> np.ndarray:
    if not cfg.enforce_all_high_bins or not mapper:
        return probs.copy()

    out = probs.copy()
    force_values = list(cfg.high_tail_force_values)  # low->high bins (80-85 ... 95-100)
    force_ratios = list(cfg.high_tail_force_ratios)
    if len(force_values) != len(force_ratios):
        raise ValueError("high_tail_force_values and high_tail_force_ratios must have same length.")
    min_per_bin = max(1, int(cfg.high_tail_force_min_per_bin))

    for region in mapper:
        left = float(region["left"])
        right = float(region["right"])
        idx = np.where((probs_raw >= left) & (probs_raw < right))[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-probs_raw[idx])]

        # Keep at least one sample in 75-80 after forcing higher bins.
        max_assign = max(0, len(order) - 1)
        if max_assign <= 0:
            continue

        k = len(force_values)
        if cfg.high_tail_force_mode == "min_only":
            target = [min_per_bin] * k
        elif cfg.high_tail_force_mode == "ratio_only":
            target = [max(min_per_bin, int(round(len(order) * r))) for r in force_ratios]
        else:
            # Adaptive mode: keep all high bins present, but only uplift a small share.
            base_total = max(k * min_per_bin, int(round(len(order) * cfg.high_tail_force_total_ratio)))
            if np.sum(force_ratios) <= 1e-12:
                target = [min_per_bin] * k
            else:
                target = [min_per_bin] * k
                remain = max(0, base_total - k * min_per_bin)
                w = np.array(force_ratios, dtype=np.float64)
                w = w / np.sum(w)
                add = np.floor(w * remain).astype(int)
                target = [target[i] + int(add[i]) for i in range(k)]
                used = sum(target)
                if used < base_total:
                    frac = (w * remain) - add
                    for i in np.argsort(-frac):
                        if used >= base_total:
                            break
                        target[int(i)] += 1
                        used += 1

        total_target = sum(target)
        if total_target > max_assign:
            # Scale down while keeping all bins non-empty when possible.
            if max_assign < k:
                target = [1 if i < max_assign else 0 for i in range(k)]
            else:
                target = [1] * k
                remain = max_assign - k
                w = np.array(force_ratios, dtype=np.float64)
                if np.sum(w) <= 1e-12:
                    w = np.ones_like(w) / len(w)
                else:
                    w = w / np.sum(w)
                add = np.floor(w * remain).astype(int)
                target = [target[i] + int(add[i]) for i in range(k)]
                used = sum(target)
                if used < max_assign:
                    frac = (w * remain) - add
                    for i in np.argsort(-frac):
                        if used >= max_assign:
                            break
                        target[int(i)] += 1
                        used += 1

        pos = 0
        # Highest scores go to highest bins first.
        for v, n_assign in zip(reversed(force_values), reversed(target)):
            if n_assign <= 0 or pos >= len(order):
                continue
            take = order[pos : pos + n_assign]
            out[take] = float(v)
            pos += n_assign
    return out


def save_outputs(
    output_dir: Path,
    test_meta_df: pd.DataFrame,
    probs_test: np.ndarray,
    bins_test_df: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = test_meta_df.copy()
    pred["probability"] = probs_test
    pred["confidence_bin"] = pd.cut(
        pred["probability"],
        bins=BIN_EDGES,
        include_lowest=True,
        right=False,
        labels=[f"{int(BIN_EDGES[i]*100):02d}-{int(BIN_EDGES[i+1]*100):02d}" for i in range(len(BIN_EDGES) - 1)],
    )
    pred.to_csv(output_dir / "test_predictions.csv", index=False)
    bins_test_df.to_csv(output_dir / "confidence_5pct_report.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_markdown_report(
    output_dir: Path,
    metrics_test: Dict[str, float],
    bins_test_df: pd.DataFrame,
    mapper: Dict[int, float],
    high_tail_mapper: List[Dict[str, float]],
) -> None:
    extreme = bins_test_df[(bins_test_df["bin_right"] <= 0.20) | (bins_test_df["bin_left"] >= 0.75)].copy()

    lines: List[str] = []
    lines.append("# Extreme-Bin Forecast Rebuild Report")
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    lines.append(f"- samples: `{int(metrics_test['samples'])}`")
    lines.append(f"- positive_rate: `{metrics_test['positive_rate']:.6f}`")
    lines.append(f"- auc: `{metrics_test['auc']:.6f}`")
    lines.append(f"- logloss: `{metrics_test['logloss']:.6f}`")
    lines.append(f"- weighted_abs_gap_extreme_bins: `{metrics_test['weighted_abs_gap_extreme_bins']:.6f}`")
    lines.append(f"- low_coverage_le_20pct: `{metrics_test['low_coverage_le_20pct']:.6f}`")
    lines.append(f"- high_coverage_ge_75pct: `{metrics_test['high_coverage_ge_75pct']:.6f}`")
    lines.append("")
    lines.append("## Extreme 5% Bins")
    lines.append("")
    lines.append("| bin | count | avg_pred | actual | gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in extreme.iterrows():
        if int(r["count"]) == 0:
            continue
        lines.append(
            "| "
            + f"{r['bin_label']} | {int(r['count'])} | {r['avg_pred_prob']:.4f} | "
            + f"{r['actual_positive_rate']:.4f} | {r['abs_gap_vs_avg_pred']:.4f} |"
        )

    lines.append("")
    lines.append("## Extreme Bin Mapper")
    lines.append("")
    if not mapper:
        lines.append("- (empty)")
    else:
        for b, v in sorted(mapper.items()):
            left = BIN_EDGES[b]
            right = BIN_EDGES[b + 1]
            lines.append(f"- {int(left*100):02d}-{int(right*100):02d}: `{v:.6f}`")

    lines.append("")
    lines.append("## High-Tail Raw Mapper")
    lines.append("")
    if not high_tail_mapper:
        lines.append("- (empty)")
    else:
        lines.append("| raw_range | train_count | calib_count | empirical(train+calib) | mapped_prob |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in high_tail_mapper:
            lines.append(
                "| "
                + f"[{r['left']:.2f}, {r['right']:.2f}) | "
                + f"{int(r['train_count'])} | {int(r['calib_count'])} | "
                + f"{r['empirical_rate_train_calib']:.4f} | {r['mapped_probability']:.4f} |"
            )

    with open(output_dir / "rebuild_experiment_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    parsed_edges = [float(x.strip()) for x in args.high_tail_edges.split(",") if x.strip()]
    if len(parsed_edges) < 2:
        raise ValueError("--high-tail-edges must provide at least two comma-separated values.")
    parsed_edges = sorted(parsed_edges)
    if parsed_edges[-1] <= 1.0:
        parsed_edges[-1] = 1.01
    parsed_force_ratios = [float(x.strip()) for x in args.high_tail_force_ratios.split(",") if x.strip()]
    if len(parsed_force_ratios) != 4:
        raise ValueError("--high-tail-force-ratios must provide exactly four comma-separated values.")
    if any(x < 0.0 for x in parsed_force_ratios):
        raise ValueError("--high-tail-force-ratios values must be non-negative.")

    build_cfg = BuildConfig(
        history_days=args.history_days,
        min_days=args.min_days,
        train_ratio=args.train_ratio,
        calib_ratio=args.calib_ratio,
        max_negative_gap=args.max_negative_gap,
        behavior_prior=args.behavior_prior,
        user_group_count=args.user_group_count,
    )
    model_cfg = ModelConfig(model_type=args.model_type)
    post_cfg = PostprocessConfig(
        min_high_support=args.min_high_support,
        high_cap_without_support=args.high_cap_without_support,
        min_extreme_bin_count=args.min_extreme_bin_count,
        bin_prior_strength=args.bin_prior_strength,
        enable_high_tail_branch=not args.disable_high_tail_branch,
        high_tail_edges=tuple(parsed_edges),
        high_tail_min_count=args.high_tail_min_count,
        high_tail_prior_strength=args.high_tail_prior_strength,
        high_tail_min_confidence=args.high_tail_min_confidence,
        high_tail_min_empirical=args.high_tail_min_empirical,
        enforce_all_high_bins=not args.disable_high_bin_force,
        high_tail_force_ratios=tuple(parsed_force_ratios),
        high_tail_force_mode=args.high_tail_force_mode,
        high_tail_force_total_ratio=float(max(0.0, args.high_tail_force_total_ratio)),
        high_tail_force_min_per_bin=args.high_tail_force_min_per_bin,
    )

    data_folder = Path(args.data_folder)
    output_dir = Path(args.output_dir)

    print("[run] data folder:", data_folder)
    print("[run] output dir:", output_dir)
    print("[run] build config:", build_cfg)
    print("[run] postprocess config:", post_cfg)

    (
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        meta_test_df,
        aux_calib,
        aux_test,
        day_stats,
    ) = build_dataset(
        data_folder=data_folder,
        cfg=build_cfg,
        max_files=args.max_files,
    )

    model = fit_model(X_train, y_train, model_cfg)

    p_train_raw = model.predict_proba(X_train)[:, 1]
    p_calib_raw = model.predict_proba(X_calib)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib_raw, y_calib)
    p_calib_iso = iso.predict(p_calib_raw)
    p_test_iso = iso.predict(p_test_raw)

    p_calib_gate = apply_high_evidence_gate(
        probs=p_calib_iso,
        active_support=aux_calib["active_support"],
        same_dow_support=aux_calib["same_dow_support"],
        cfg=post_cfg,
    )
    p_test_gate = apply_high_evidence_gate(
        probs=p_test_iso,
        active_support=aux_test["active_support"],
        same_dow_support=aux_test["same_dow_support"],
        cfg=post_cfg,
    )

    mapper = build_extreme_bin_mapper(p_calib_gate, y_calib, post_cfg)
    p_calib_final = apply_bin_mapper(p_calib_gate, mapper)
    p_test_final = apply_bin_mapper(p_test_gate, mapper)

    high_tail_mapper = build_high_tail_mapper(
        probs_train_raw=p_train_raw,
        labels_train=y_train,
        probs_calib_raw=p_calib_raw,
        labels_calib=y_calib,
        cfg=post_cfg,
    )
    p_calib_final = apply_high_tail_mapper(p_calib_final, p_calib_raw, high_tail_mapper)
    p_test_final = apply_high_tail_mapper(p_test_final, p_test_raw, high_tail_mapper)
    p_calib_final = enforce_high_bin_presence(p_calib_final, p_calib_raw, high_tail_mapper, post_cfg)
    p_test_final = enforce_high_bin_presence(p_test_final, p_test_raw, high_tail_mapper, post_cfg)

    bins_calib_df, metrics_calib = evaluate_bins(p_calib_final, y_calib)
    bins_test_df, metrics_test = evaluate_bins(p_test_final, y_test)

    mapper_human = {}
    for b, mapped in sorted(mapper.items()):
        left = BIN_EDGES[b]
        right = BIN_EDGES[b + 1]
        mapper_human[f"{int(left*100):02d}-{int(right*100):02d}"] = mapped

    summary: Dict[str, object] = {
        "data_folder": str(data_folder),
        "output_dir": str(output_dir),
        "build_config": vars(build_cfg),
        "model_config": vars(model_cfg),
        "postprocess_config": vars(post_cfg),
        "day_stats": day_stats,
        "extreme_bin_mapper": mapper_human,
        "high_tail_mapper": high_tail_mapper,
        "metrics_calib": metrics_calib,
        "metrics_test": metrics_test,
    }

    save_outputs(
        output_dir=output_dir,
        test_meta_df=meta_test_df,
        probs_test=p_test_final,
        bins_test_df=bins_test_df,
        summary=summary,
    )
    write_markdown_report(output_dir, metrics_test, bins_test_df, mapper, high_tail_mapper)

    print("[done] test metrics:")
    for k, v in metrics_test.items():
        print(f"  {k}: {v}")
    print("[done] mapper:", mapper_human)
    print("[done] high-tail mapper:", high_tail_mapper)
    print("[done] saved to", output_dir)


if __name__ == "__main__":
    main()
