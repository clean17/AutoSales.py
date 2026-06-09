#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stable_rule_miner_final_v4_7_no_today_ret2_d5.py

v4 strict validator + return-score adaptation + no-today ret2 depth5 final filters.

추가
selection_avg_return 검증 구간의 평균 수익률이 더 높으면 점수 높게
profit_factor        손익비 계산, 큰 손실의 룰을 걸러낸다
월별 return          월별 최소 수익률이 깨지지 않는지 확인
no-hit return        정확도 지표와 실제 수익률 지표 사이의 간극을 줄이는 기능
룰 export 기능        결과 룰을 py 스크립트로 생성

This version removes direct and indirect today_pct features:
- today_pct
- price_power_value = today_pct * log1p(today_tr_val_eok)
- body_value_power = body_ratio * today_pct * log1p(today_tr_val_eok)

Main target:
- final precision >= 50%
- final avg_return >= 2.0%
- fixed-rule forward stability
- max_depth default = 5

Recommended run:
python stable_rule_miner_final_v4_7_no_today_ret2_d5.py ^
  --csv csv/low_result_7_v2_desc.csv ^
  --out stable_rule_miner_final_v4_7_no_today_ret2_d5_out ^
  --target target_before_stop_10 ^
  --date-col today ^
  --feature-set no_today ^
  --threshold-mode quantile ^
  --quantile-grid v4 ^
  --manual-thresholds off ^
  --use-corr-pruning ^
  --max-depth 5 ^
  --beam-width 350 ^
  --top-k 120 ^
  --fw-top-rules 100 ^
  --fw-splits 6

--intraday_return까지 제거
python stable_rule_miner_final_v4_9_mine_and_export_rules.py ^
  --csv csv/low_result_7_v2_desc.csv ^
  --out stable_rule_miner_final_v4_9_all_no_today_no_intraday_out ^
  --target target_before_stop_10 ^
  --date-col today ^
  --feature-set all_no_today_no_intraday ^
  --threshold-mode quantile ^
  --quantile-grid v4 ^
  --manual-thresholds off ^
  --use-corr-pruning ^
  --max-depth 5 ^
  --beam-width 300 ^
  --top-k 150 ^
  --fw-top-rules 120 ^
  --fw-splits 6 ^
  --buy-rules-filename lowscan_positive_rules_auto.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


TARGET_COL = "target_before_stop_10"

# =============================================================================
# Feature sets adjusted for no-today ret2 depth5 run
# =============================================================================

FORBIDDEN_TODAY_DERIVED_FEATURES = {
    "today_pct",
    "price_power_value",
    "body_value_power",
}


FEATURE_GROUPS = {
    "upper_wick_ratio": "CANDLE",
    "lower_wick_ratio": "CANDLE",
    "vol5": "VOLATILITY",
    "vol15": "VOLATILITY",
    "room_to_60d_high": "HIGH_ROOM",
    "gap_pct": "GAP",
    "rebound_from_7d_low": "REBOUND",
    "tr_value_ratio_5d": "VOLUME_POWER",
    "intraday_return": "INTRADAY",
    "dist_to_ma20": "POSITION",
    "pct_vs_lastweek": "WEEK_POSITION",
    "ma5_chg_rate": "TREND",
    "price_power_value": "POWER",
    "BB_perc": "BAND",
    "body_value_power": "POWER",
    "max_drop_7d": "DROP",
    "today_pct": "PRICE",
    "dist_to_ma5": "POSITION",
    "ATR_pct": "VOLATILITY",
}

CORE_FEATURES = [
    "upper_wick_ratio",
    "vol5",
    "vol15",
    "room_to_60d_high",
    "gap_pct",
    "rebound_from_7d_low",
    "lower_wick_ratio",
    "tr_value_ratio_5d",
    "intraday_return",
]

# v4_5 15-hour run에서 final/filter까지 반복적으로 살아남은 핵심 피쳐만 모은 세트.
# 먼저 이 feature-set으로 돌리면 시간을 줄이면서도 핵심 패턴을 유지할 가능성이 높다.
NO_TODAY_FEATURES = [
    "gap_pct",
    "intraday_return",
    "max_drop_7d",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "ma5_chg_rate",
    "pct_vs_lastweek",
    "vol5",
    "vol15",
    "room_to_60d_high",
]

# Backward-compatible alias. In this version core2 intentionally means no-today.
CORE2_FEATURES = NO_TODAY_FEATURES

BALANCED_FEATURES = [
    "upper_wick_ratio",
    "vol5",
    "vol15",
    "room_to_60d_high",
    "gap_pct",
    "rebound_from_7d_low",
    "lower_wick_ratio",
    "tr_value_ratio_5d",
    "intraday_return",
    "dist_to_ma20",
    "pct_vs_lastweek",
    "ma5_chg_rate",
    "BB_perc",
    "max_drop_7d",
]

ALL_FEATURES = [
    "upper_wick_ratio",
    "vol5",
    "vol15",
    "room_to_60d_high",
    "gap_pct",
    "rebound_from_7d_low",
    "lower_wick_ratio",
    "tr_value_ratio_5d",
    "intraday_return",
    "dist_to_ma20",
    "pct_vs_lastweek",
    "ma5_chg_rate",
    "BB_perc",
    "max_drop_7d",
    "dist_to_ma5",
    "ATR_pct",
]

# 전체 피쳐에서 today_pct 직접/간접 파생 3개만 제거한 세트.
ALL_NO_TODAY_FEATURES = [
    f for f in ALL_FEATURES
    if f not in FORBIDDEN_TODAY_DERIVED_FEATURES
]

# 전체 no_today 세트에서 intraday_return까지 추가 제거한 실전성 점검용 세트.
ALL_NO_TODAY_NO_INTRADAY_FEATURES = [
    f for f in ALL_NO_TODAY_FEATURES
    if f != "intraday_return"
]

# Direction revised from v5 fast output.
ALLOWED_OPS = {
    "upper_wick_ratio": ["<="],
    "vol5": [">="],
    "vol15": [">="],
    "rebound_from_7d_low": [">="],
    "tr_value_ratio_5d": [">="],
    "intraday_return": [">="],
    "ATR_pct": [">="],
    "max_drop_7d": ["<="],

    # v5 fast result favored these directions, but interval is kept optional for safer discovery.
    "gap_pct": [">="],
    "room_to_60d_high": ["<="],
    "lower_wick_ratio": ["<="],
    "dist_to_ma20": [">="],
    "pct_vs_lastweek": [">="],
    "ma5_chg_rate": [">="],
    "BB_perc": [">="],
    "dist_to_ma5": [">=", "<="],
}

# Only keep interval candidates where both-side bounds have plausible meaning.
INTERVAL_FEATURES = {
    "gap_pct",
    "room_to_60d_high",
    "lower_wick_ratio",
    "dist_to_ma20",
    "pct_vs_lastweek",
    "ma5_chg_rate",
    "BB_perc",
    "dist_to_ma5",
}


@dataclass(frozen=True)
class Atom:
    feature: str
    op: str
    threshold: float
    source: str = "quantile"

    def name(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.6g}"


@dataclass
class Rule:
    atoms: Tuple[Atom, ...]
    train_metrics: Dict
    selection_metrics: Dict
    final_metrics: Dict
    train_monthly: Dict
    selection_monthly: Dict
    final_monthly: Dict
    score: float
    train_mask_key: str
    selection_mask_key: str

    def name(self) -> str:
        return " AND ".join(a.name() for a in self.atoms)

    def features(self) -> List[str]:
        return sorted(set(a.feature for a in self.atoms))


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["today", "date", "Date", "datetime", "trade_date", "trd_date", "일자", "날짜", "기준일", "ymd", "YMD"]:
        if c in df.columns:
            return c
    return None


def choose_features(feature_set: str, custom_features: Optional[str] = None) -> List[str]:
    if custom_features:
        features = [x.strip() for x in custom_features.split(",") if x.strip()]
    elif feature_set == "core":
        features = CORE_FEATURES
    elif feature_set in {"core2", "no_today"}:
        features = NO_TODAY_FEATURES
    elif feature_set == "balanced":
        features = BALANCED_FEATURES
    elif feature_set == "all":
        features = ALL_FEATURES
    elif feature_set == "all_no_today":
        features = ALL_NO_TODAY_FEATURES
    elif feature_set == "all_no_today_no_intraday":
        features = ALL_NO_TODAY_NO_INTRADAY_FEATURES
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")

    # Hard safety: never allow direct/indirect today_pct features in this script.
    # This also protects custom --features runs.
    return [f for f in features if f not in FORBIDDEN_TODAY_DERIVED_FEATURES]


def quantile_grid(name: str) -> List[float]:
    if name == "decile":
        return [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    if name == "v4":
        return [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975]
    if name == "fine":
        return [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975]
    raise ValueError(f"unknown quantile_grid: {name}")


def default_extra_thresholds() -> Dict[str, List[Tuple[str, float]]]:
    # Optional. Keep small, interpretable values only. Default run should use --manual-thresholds off.
    return {
        "upper_wick_ratio": [("<=", 0.0), ("<=", 0.02), ("<=", 0.05), ("<=", 0.10)],
        "vol5": [(">=", 6.0), (">=", 8.0), (">=", 9.0), (">=", 10.0)],
        "vol15": [(">=", 5.0), (">=", 7.0), (">=", 9.0)],
        "room_to_60d_high": [("<=", 20.0), ("<=", 30.0), ("<=", 50.0)],
        "gap_pct": [(">=", 0.0), (">=", 1.0), (">=", 2.0), (">=", 4.0)],
        "lower_wick_ratio": [("<=", 0.0), ("<=", 0.05), ("<=", 0.10)],
        "tr_value_ratio_5d": [(">=", 2.0), (">=", 4.0), (">=", 8.0)],
        "intraday_return": [(">=", 4.0), (">=", 6.0), (">=", 8.0), (">=", 10.0)],
        "today_pct": [(">=", 5.0), (">=", 8.0), (">=", 10.0), (">=", 12.0)],
        "max_drop_7d": [("<=", -3.0), ("<=", -5.0), ("<=", -7.0), ("<=", -10.0)],
    }


def prepare_df(df: pd.DataFrame, target_col: str, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")
    if date_col not in df.columns:
        raise ValueError(f"date column not found: {date_col}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    df["quarter"] = df[date_col].dt.to_period("Q").astype(str)
    df["year"] = df[date_col].dt.year.astype(str)
    return df


def split_train_selection_test_by_date(df: pd.DataFrame, date_col: str, train_ratio: float, selection_valid_ratio: float, final_test_ratio: float):
    total = train_ratio + selection_valid_ratio + final_test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    work = df.sort_values(date_col).reset_index(drop=True).copy()
    dates = list(pd.Series(work[date_col].dropna().unique()).sort_values())
    if len(dates) < 3:
        n = len(work)
        i1 = int(n * train_ratio)
        i2 = int(n * (train_ratio + selection_valid_ratio))
        train = work.iloc[:i1].copy().reset_index(drop=True)
        selection = work.iloc[i1:i2].copy().reset_index(drop=True)
        final = work.iloc[i2:].copy().reset_index(drop=True)
        info = {"split_mode": "row_fallback"}
        return train, selection, final, info

    i1 = int(len(dates) * train_ratio)
    i2 = int(len(dates) * (train_ratio + selection_valid_ratio))
    i1 = max(1, min(i1, len(dates) - 2))
    i2 = max(i1 + 1, min(i2, len(dates) - 1))
    selection_start = dates[i1]
    final_start = dates[i2]

    train = work[work[date_col] < selection_start].copy().reset_index(drop=True)
    selection = work[(work[date_col] >= selection_start) & (work[date_col] < final_start)].copy().reset_index(drop=True)
    final = work[work[date_col] >= final_start].copy().reset_index(drop=True)

    info = {
        "split_mode": "date",
        "train_start_date": train[date_col].min() if len(train) else None,
        "train_end_date": train[date_col].max() if len(train) else None,
        "selection_start_date": selection[date_col].min() if len(selection) else None,
        "selection_end_date": selection[date_col].max() if len(selection) else None,
        "final_start_date": final[date_col].min() if len(final) else None,
        "final_end_date": final[date_col].max() if len(final) else None,
    }
    return train, selection, final, info


def wilson_lcb(success: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return np.nan
    p = success / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def target_suffix(target_col: str) -> Optional[str]:
    if target_col.startswith("target_before_stop_"):
        return target_col.replace("target_before_stop_", "")
    return None


def build_trade_return_series(df: pd.DataFrame, target_col: str, args) -> pd.Series:
    suffix = target_suffix(target_col)
    target_pct = pd.to_numeric(df["target_pct"], errors="coerce") if "target_pct" in df.columns else pd.Series(args.target_pct, index=df.index)
    stop_loss = pd.to_numeric(df["stop_loss"], errors="coerce") if "stop_loss" in df.columns else pd.Series(args.stop_loss, index=df.index)
    target_pct = target_pct.fillna(args.target_pct)
    stop_loss = stop_loss.fillna(args.stop_loss)

    ret = pd.Series(0.0, index=df.index)
    target_hit = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int) == 1
    ret[target_hit] = target_pct[target_hit]

    if suffix is not None:
        stop_col = f"stop_before_target_{suffix}"
        same_col = f"target_stop_same_day_{suffix}"
        none_col = f"no_target_no_stop_{suffix}"
        if stop_col in df.columns:
            stop_hit = pd.to_numeric(df[stop_col], errors="coerce").fillna(0).astype(int) == 1
            ret[stop_hit] = stop_loss[stop_hit]
        if same_col in df.columns:
            same_hit = pd.to_numeric(df[same_col], errors="coerce").fillna(0).astype(int) == 1
            ret[same_hit] = args.same_day_return
        if none_col in df.columns:
            none_hit = pd.to_numeric(df[none_col], errors="coerce").fillna(0).astype(int) == 1
            if args.no_hit_return_col and args.no_hit_return_col in df.columns:
                r = pd.to_numeric(df[args.no_hit_return_col], errors="coerce").fillna(0.0)
            elif "validation_close_rate7" in df.columns:
                r = pd.to_numeric(df["validation_close_rate7"], errors="coerce").fillna(0.0)
            else:
                r = pd.Series(0.0, index=df.index)
            lo = np.minimum(stop_loss, target_pct)
            hi = np.maximum(stop_loss, target_pct)
            ret[none_hit] = r.clip(lower=lo, upper=hi)[none_hit]
    return ret.astype(float)


def calc_basic_metrics(y: np.ndarray, mask: np.ndarray) -> Dict:
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask).astype(bool)
    total_count = int(len(y))
    selected_count = int(mask.sum())
    base_rate = float(y.mean()) if total_count else np.nan
    if selected_count == 0:
        return {
            "selected_count": 0,
            "positive_count": 0,
            "precision": np.nan,
            "precision_lcb": np.nan,
            "base_rate": base_rate,
            "lift": np.nan,
            "coverage": 0.0,
            "selected_rate": 0.0,
        }
    positive_count = int(y[mask].sum())
    precision = positive_count / selected_count
    total_pos = int(y.sum())
    return {
        "selected_count": selected_count,
        "positive_count": positive_count,
        "precision": precision,
        "precision_lcb": wilson_lcb(positive_count, selected_count),
        "base_rate": base_rate,
        "lift": precision / base_rate if base_rate and base_rate > 0 else np.nan,
        "coverage": positive_count / total_pos if total_pos > 0 else 0.0,
        "selected_rate": selected_count / total_count if total_count else 0.0,
    }


def add_return_metrics(metrics: Dict, returns: pd.Series, mask: np.ndarray) -> Dict:
    out = dict(metrics)
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        out.update({"avg_return": np.nan, "median_return": np.nan, "sum_return": 0.0, "win_rate_return": np.nan, "avg_win": np.nan, "avg_loss": np.nan, "win_loss_ratio": np.nan, "profit_factor": np.nan})
        return out
    r = pd.to_numeric(returns[mask], errors="coerce").dropna()
    if len(r) == 0:
        out.update({"avg_return": np.nan, "median_return": np.nan, "sum_return": 0.0, "win_rate_return": np.nan, "avg_win": np.nan, "avg_loss": np.nan, "win_loss_ratio": np.nan, "profit_factor": np.nan})
        return out
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = losses.sum() if len(losses) else 0.0
    out.update({
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "sum_return": float(r.sum()),
        "win_rate_return": float((r > 0).mean()),
        "avg_win": float(avg_win) if np.isfinite(avg_win) else np.nan,
        "avg_loss": float(avg_loss) if np.isfinite(avg_loss) else np.nan,
        "win_loss_ratio": float(avg_win / abs(avg_loss)) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0 else np.nan,
        "profit_factor": float(gross_win / abs(gross_loss)) if gross_loss < 0 else np.nan,
    })
    return out


def calc_metrics_df(df: pd.DataFrame, target_col: str, mask: np.ndarray, args) -> Dict:
    m = calc_basic_metrics(df[target_col].values, mask)
    returns = build_trade_return_series(df, target_col, args)
    return add_return_metrics(m, returns, mask)


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.md5(np.packbits(np.asarray(mask).astype(bool)).tobytes()).hexdigest()


def canonical_atoms_key(atoms: Tuple[Atom, ...]):
    return tuple(sorted((a.feature, a.op, round(float(a.threshold), 10)) for a in atoms))


def canonicalize_atoms(atoms: Tuple[Atom, ...]) -> Tuple[Atom, ...]:
    return tuple(sorted(atoms, key=lambda a: (a.feature, a.op, a.threshold)))


def apply_atom(df: pd.DataFrame, atom: Atom) -> np.ndarray:
    s = pd.to_numeric(df[atom.feature], errors="coerce")
    if atom.op == ">=":
        return (s >= atom.threshold).fillna(False).to_numpy()
    if atom.op == "<=":
        return (s <= atom.threshold).fillna(False).to_numpy()
    raise ValueError(f"unknown op: {atom.op}")


def apply_rule(df: pd.DataFrame, atoms: Tuple[Atom, ...]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for a in atoms:
        mask &= apply_atom(df, a)
    return mask


def build_corr_pairs(df: pd.DataFrame, features: List[str], corr_threshold: float):
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman")
    rows = []
    pairs = set()
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            c = corr.loc[a, b]
            if pd.isna(c):
                continue
            if abs(c) >= corr_threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman_corr": c, "abs_corr": abs(c)})
                pairs.add(frozenset([a, b]))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("abs_corr", ascending=False)
    return out, pairs


def has_correlated_pair(used_features: Set[str], new_feature: str, corr_pairs: Set[frozenset]) -> bool:
    return any(frozenset([f, new_feature]) in corr_pairs for f in used_features)


def make_atoms(train: pd.DataFrame, features: List[str], quantiles: List[float], extra_thresholds: Dict[str, List[Tuple[str, float]]], allowed_ops: Dict[str, List[str]]) -> List[Atom]:
    atoms: List[Atom] = []
    for f in features:
        if f not in train.columns:
            continue
        s = pd.to_numeric(train[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = s.dropna()
        if vals.nunique() < 8:
            continue
        candidates: List[Tuple[float, str]] = []
        for q in quantiles:
            v = vals.quantile(q)
            if np.isfinite(v):
                candidates.append((round(float(v), 10), f"q{q:.3f}"))
        for op, th in extra_thresholds.get(f, []):
            candidates.append((round(float(th), 10), "manual"))
        seen_val_source = set()
        for th, source in candidates:
            for op in allowed_ops.get(f, [">=", "<="]):
                if source == "manual" and (op, th) not in set((o, round(float(t), 10)) for o, t in extra_thresholds.get(f, [])):
                    continue
                key = (f, op, th)
                if key in seen_val_source:
                    continue
                seen_val_source.add(key)
                atoms.append(Atom(f, op, float(th), source))
    out = []
    seen = set()
    for a in atoms:
        key = (a.feature, a.op, round(a.threshold, 10))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def can_add_atom(base_atoms: Tuple[Atom, ...], atom: Atom, corr_pairs: Set[frozenset], args) -> bool:
    used_features = set(a.feature for a in base_atoms)
    same = [a for a in base_atoms if a.feature == atom.feature]

    if atom.feature not in INTERVAL_FEATURES:
        if atom.feature in used_features:
            return False
    else:
        if len(same) >= 2:
            return False
        if len(same) == 1:
            old = same[0]
            if old.op == atom.op:
                return False
            if old.op == ">=" and atom.op == "<=" and old.threshold > atom.threshold:
                return False
            if old.op == "<=" and atom.op == ">=" and atom.threshold > old.threshold:
                return False

    if args.use_corr_pruning and atom.feature not in used_features:
        if has_correlated_pair(used_features, atom.feature, corr_pairs):
            return False
    return True


def monthly_summary_from_mask(df: pd.DataFrame, target_col: str, mask: np.ndarray, args):
    df = df.reset_index(drop=True).copy()
    mask_all = np.asarray(mask).astype(bool)
    rows = []
    for month, g in df.groupby("month", sort=True):
        idx = g.index.to_numpy()
        m = calc_metrics_df(g.reset_index(drop=True), target_col, mask_all[idx], args)
        m["month"] = month
        rows.append(m)
    monthly_df = pd.DataFrame(rows)
    if len(monthly_df) == 0:
        return monthly_df, {"n_usable_months": 0, "mean_month_precision": np.nan, "min_month_precision": np.nan, "std_month_precision": np.nan, "mean_month_return": np.nan, "min_month_return": np.nan, "std_month_return": np.nan, "mean_month_count": np.nan, "total_month_count": 0, "pass_month_rate": 0.0, "bad_month_count": 0, "crash_month_count": 999}
    usable = monthly_df[monthly_df["selected_count"] >= args.min_month_count].copy()
    if len(usable) == 0:
        return monthly_df, {"n_usable_months": 0, "mean_month_precision": np.nan, "min_month_precision": np.nan, "std_month_precision": np.nan, "mean_month_return": np.nan, "min_month_return": np.nan, "std_month_return": np.nan, "mean_month_count": np.nan, "total_month_count": int(monthly_df["selected_count"].sum()), "pass_month_rate": 0.0, "bad_month_count": len(monthly_df), "crash_month_count": len(monthly_df)}
    pass_mask = (usable["precision"] >= args.month_pass_precision) & (usable["lift"] >= args.month_pass_lift) & (usable["avg_return"] >= args.month_pass_avg_return)
    crash_mask = (usable["precision"] < args.month_crash_precision) | (usable["avg_return"] < args.month_crash_avg_return)
    return monthly_df, {
        "n_usable_months": len(usable),
        "mean_month_precision": float(usable["precision"].mean()),
        "min_month_precision": float(usable["precision"].min()),
        "std_month_precision": float(usable["precision"].std(ddof=0)),
        "mean_month_return": float(usable["avg_return"].mean()),
        "min_month_return": float(usable["avg_return"].min()),
        "std_month_return": float(usable["avg_return"].std(ddof=0)),
        "mean_month_count": float(usable["selected_count"].mean()),
        "total_month_count": int(monthly_df["selected_count"].sum()),
        "pass_month_rate": float(pass_mask.mean()),
        "bad_month_count": int((~pass_mask).sum()),
        "crash_month_count": int(crash_mask.sum()),
    }


def eval_rule(df: pd.DataFrame, target_col: str, atoms: Tuple[Atom, ...], args):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics_df(df, target_col, mask, args)
    monthly_df, mon = monthly_summary_from_mask(df, target_col, mask, args)
    return metrics, mon, monthly_df, mask


def eval_rule_fast(df: pd.DataFrame, target_col: str, atoms: Tuple[Atom, ...], args):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics_df(df, target_col, mask, args)
    return metrics, mask


def safe_num(x, default=0.0) -> float:
    try:
        if x is None or not np.isfinite(x):
            return default
        return float(x)
    except Exception:
        return default


def breakeven_precision(args) -> float:
    target = abs(float(args.target_pct))
    stop = abs(float(args.stop_loss))
    return stop / (target + stop) if target + stop else 0.5


def beam_score_fast(metrics, args) -> float:
    count = metrics["selected_count"]
    precision = metrics["precision"]
    lift = metrics["lift"]
    avg_return = metrics.get("avg_return", np.nan)
    if count < args.beam_min_count:
        return -1e18
    if not np.isfinite(precision) or not np.isfinite(lift):
        return -1e18
    if lift < args.beam_min_lift:
        return -1e18
    return safe_num(avg_return) * 22 + precision * 50 + safe_num(metrics.get("precision_lcb")) * 28 + lift * 9 + math.log1p(count) * 2.0


def final_score(train_m, selection_m, train_mon, selection_mon, args):
    train_count = train_m["selected_count"]
    selection_count = selection_m["selected_count"]
    train_p = train_m["precision"]
    selection_p = selection_m["precision"]
    train_lift = train_m["lift"]
    selection_lift = selection_m["lift"]
    selection_return = selection_m.get("avg_return", np.nan)
    if train_count < args.min_train_count or selection_count < args.min_selection_count:
        return -1e18
    if not all(np.isfinite(x) for x in [train_p, selection_p, train_lift, selection_lift, selection_return]):
        return -1e18
    if train_lift < args.min_train_lift or selection_lift < args.min_selection_lift:
        return -1e18
    if train_p < args.min_train_precision or selection_p < args.min_selection_precision:
        return -1e18
    if selection_return < args.min_selection_avg_return:
        return -1e18
    gap = abs(train_p - selection_p)
    if gap > args.max_precision_gap:
        return -1e18
    if selection_mon["n_usable_months"] < args.min_selection_usable_months:
        return -1e18
    if selection_mon["pass_month_rate"] < args.min_selection_pass_month_rate:
        return -1e18
    if np.isfinite(selection_mon["min_month_precision"]) and selection_mon["min_month_precision"] < args.min_selection_month_min_precision:
        return -1e18
    if np.isfinite(selection_mon.get("min_month_return", np.nan)) and selection_mon["min_month_return"] < args.min_selection_month_min_return:
        return -1e18
    if selection_mon.get("crash_month_count", 0) > args.max_selection_crash_months:
        return -1e18
    selection_lcb = selection_m["precision_lcb"]
    if np.isfinite(selection_lcb) and selection_lcb < args.min_selection_lcb:
        return -1e18

    selection_std = safe_num(selection_mon.get("std_month_precision"))
    return_std = safe_num(selection_mon.get("std_month_return"))
    selection_min_p = safe_num(selection_mon.get("min_month_precision"))
    selection_min_ret = safe_num(selection_mon.get("min_month_return"))

    precision_bonus_60 = max(0.0, selection_p - 0.60) * 120
    precision_bonus_70 = max(0.0, selection_p - 0.70) * 220
    count_bonus = math.log1p(selection_count) * args.selection_count_weight
    return (
        selection_p * 110
        + selection_lcb * 75
        + selection_lift * 18
        + safe_num(selection_return) * 42
        + safe_num(selection_m.get("profit_factor")) * 7
        + safe_num(selection_mon["mean_month_precision"]) * 55
        + selection_min_p * 45
        + safe_num(selection_mon["mean_month_return"]) * 25
        + selection_min_ret * 18
        + safe_num(selection_mon["pass_month_rate"]) * 45
        + train_p * 15
        + count_bonus
        + precision_bonus_60
        + precision_bonus_70
        - gap * 80
        - selection_std * 45
        - return_std * 10
        - selection_mon["bad_month_count"] * 3
        - selection_mon.get("crash_month_count", 0) * 25
    )


def search_rules(train, selection, final_test, atoms, corr_pairs, args):
    beam = [tuple()]
    selected_candidates: List[Rule] = []
    seen_rules = set()
    seen_train_masks = set()
    for depth in range(1, args.max_depth + 1):
        print(f"[INFO] searching depth={depth}")
        beam_candidates: List[Rule] = []
        for base_atoms in beam:
            for atom in atoms:
                if not can_add_atom(base_atoms, atom, corr_pairs, args):
                    continue
                new_atoms = canonicalize_atoms(tuple(list(base_atoms) + [atom]))
                key = canonical_atoms_key(new_atoms)
                if key in seen_rules:
                    continue
                seen_rules.add(key)
                train_fast_m, train_mask = eval_rule_fast(train, args.target, new_atoms, args)
                bscore_fast = beam_score_fast(train_fast_m, args)
                if bscore_fast <= -1e17:
                    continue
                train_key = mask_hash(train_mask)
                if train_key in seen_train_masks and len(new_atoms) > 1:
                    continue
                seen_train_masks.add(train_key)
                selection_fast_m, selection_mask = eval_rule_fast(selection, args.target, new_atoms, args)
                if selection_fast_m["selected_count"] < args.fast_min_selection_count:
                    continue
                if not np.isfinite(selection_fast_m["precision"]):
                    continue
                if selection_fast_m["precision"] < args.fast_min_selection_precision:
                    continue
                if not np.isfinite(selection_fast_m["lift"]) or selection_fast_m["lift"] < args.fast_min_selection_lift:
                    continue
                if np.isfinite(selection_fast_m.get("avg_return", np.nan)) and selection_fast_m["avg_return"] < args.fast_min_selection_avg_return:
                    continue
                train_m, train_mon, _, train_mask = eval_rule(train, args.target, new_atoms, args)
                selection_m, selection_mon, _, selection_mask = eval_rule(selection, args.target, new_atoms, args)
                final_m, final_mon, _, _ = eval_rule(final_test, args.target, new_atoms, args)
                fscore = final_score(train_m, selection_m, train_mon, selection_mon, args)
                score = fscore if fscore > -1e17 else bscore_fast
                rule = Rule(new_atoms, train_m, selection_m, final_m, train_mon, selection_mon, final_mon, score, train_key, mask_hash(selection_mask))
                beam_candidates.append(rule)
                if fscore > -1e17:
                    selected_candidates.append(rule)
        if not beam_candidates:
            print(f"[WARN] no beam candidates at depth={depth}")
            break
        beam_candidates = sorted(beam_candidates, key=lambda r: r.score, reverse=True)
        beam = [r.atoms for r in beam_candidates[:args.beam_width]]
        print(f"[INFO] depth={depth}, beam_candidates={len(beam_candidates)}, selected_candidates={len(selected_candidates)}, kept_beam={len(beam)}")
    selected_candidates = sorted(selected_candidates, key=lambda r: r.score, reverse=True)
    deduped: List[Rule] = []
    for r in selected_candidates:
        if is_near_duplicate(r, deduped, args):
            continue
        deduped.append(r)
        if len(deduped) >= args.top_k:
            break
    return deduped


def is_near_duplicate(new_rule: Rule, selected: List[Rule], args) -> bool:
    for r in selected:
        if new_rule.train_mask_key == r.train_mask_key or new_rule.selection_mask_key == r.selection_mask_key:
            return True
        c1 = new_rule.selection_metrics["selected_count"]
        c0 = r.selection_metrics["selected_count"]
        p1 = new_rule.selection_metrics["precision"]
        p0 = r.selection_metrics["precision"]
        if c0 == 0 or c1 == 0:
            continue
        count_change = abs(c1 - c0) / max(c0, 1)
        precision_gain = p1 - p0 if np.isfinite(p1) and np.isfinite(p0) else 0.0
        f_new = set(new_rule.features())
        f_old = set(r.features())
        if count_change <= args.dup_count_tol and precision_gain <= args.dup_precision_gain:
            if f_old.issubset(f_new) or f_new.issubset(f_old):
                return True
    return False


def simplify_rule_by_dropping_atoms(rule: Rule, train, selection, final_test, args) -> Rule:
    current = rule
    improved = True
    while improved and len(current.atoms) > 1:
        improved = False
        best_candidate = None
        best_tuple = None
        for i in range(len(current.atoms)):
            new_atoms = canonicalize_atoms(tuple(a for j, a in enumerate(current.atoms) if j != i))
            train_m, train_mon, _, train_mask = eval_rule(train, args.target, new_atoms, args)
            selection_m, selection_mon, _, selection_mask = eval_rule(selection, args.target, new_atoms, args)
            final_m, final_mon, _, _ = eval_rule(final_test, args.target, new_atoms, args)
            fs = final_score(train_m, selection_m, train_mon, selection_mon, args)
            if fs <= -1e17:
                continue
            old_p = current.selection_metrics["precision"]
            new_p = selection_m["precision"]
            old_count = current.selection_metrics["selected_count"]
            new_count = selection_m["selected_count"]
            if new_p + args.simplify_precision_drop < old_p:
                continue
            if new_count < old_count * args.simplify_min_count_ratio:
                continue
            tup = (-len(new_atoms), fs, safe_num(selection_m.get("avg_return")), new_p, new_count)
            if best_tuple is None or tup > best_tuple:
                best_tuple = tup
                best_candidate = Rule(new_atoms, train_m, selection_m, final_m, train_mon, selection_mon, final_mon, fs, mask_hash(train_mask), mask_hash(selection_mask))
        if best_candidate is not None:
            current = best_candidate
            improved = True
    return current


def flatten(prefix: str, d: Dict) -> Dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def rules_to_df(rules: List[Rule], args) -> pd.DataFrame:
    rows = []
    be = breakeven_precision(args)
    for rank, r in enumerate(rules, start=1):
        row = {"rank": rank, "rule": r.name(), "n_atoms": len(r.atoms), "features": ",".join(r.features()), "score": r.score, "breakeven_precision": be}
        row.update(flatten("train", r.train_metrics))
        row.update(flatten("selection", r.selection_metrics))
        row.update(flatten("final", r.final_metrics))
        row.update(flatten("train_month", r.train_monthly))
        row.update(flatten("selection_month", r.selection_monthly))
        row.update(flatten("final_month", r.final_monthly))
        row["precision_gap_train_selection_abs"] = abs(row["train_precision"] - row["selection_precision"])
        row["precision_gap_selection_final_abs"] = abs(row["selection_precision"] - row["final_precision"])
        row["pass_selection_50_n30"] = row["selection_precision"] >= 0.50 and row["selection_selected_count"] >= 30 and row["selection_avg_return"] >= 0.5
        row["pass_selection_60_n25"] = row["selection_precision"] >= 0.60 and row["selection_selected_count"] >= 25 and row["selection_avg_return"] >= 1.0
        row["pass_selection_70_n20"] = row["selection_precision"] >= 0.70 and row["selection_selected_count"] >= 20 and row["selection_avg_return"] >= 2.0
        row["pass_final_40_n40"] = row["final_precision"] >= 0.40 and row["final_selected_count"] >= 40 and row["final_avg_return"] >= 0.0
        row["pass_final_45_n30"] = row["final_precision"] >= 0.45 and row["final_selected_count"] >= 30 and row["final_avg_return"] >= 0.3
        row["pass_final_50_n20"] = row["final_precision"] >= 0.50 and row["final_selected_count"] >= 20 and row["final_avg_return"] >= 0.5
        row["pass_final_50_ret2_n30"] = row["final_precision"] >= 0.50 and row["final_selected_count"] >= 30 and row["final_avg_return"] >= 2.0
        row["type_high_precision"] = row["selection_precision"] >= args.type_hp_selection_precision and row["final_precision"] >= args.type_hp_final_precision and row["selection_precision_lcb"] >= args.type_hp_selection_lcb and row["selection_avg_return"] >= args.type_hp_avg_return
        row["type_high_coverage"] = row["selection_selected_count"] >= args.type_hc_selection_count and row["final_selected_count"] >= args.type_hc_final_count and row["selection_precision"] >= args.type_hc_selection_precision and row["selection_avg_return"] >= args.type_hc_avg_return
        row["type_stable"] = row["selection_month_pass_month_rate"] >= args.type_st_pass_month_rate and row["selection_month_crash_month_count"] <= args.type_st_max_crash_months and row["selection_avg_return"] >= args.type_st_avg_return
        row["type_aggressive"] = row["selection_precision"] >= args.type_ag_selection_precision and row["selection_avg_return"] >= args.type_ag_avg_return and row["selection_selected_count"] >= args.type_ag_selection_count
        row["type_defensive"] = row["selection_precision"] >= args.type_df_selection_precision and row["selection_precision_lcb"] >= args.type_df_selection_lcb and row["selection_month_crash_month_count"] <= args.type_df_max_crash_months and row["selection_avg_return"] >= args.type_df_avg_return
        rows.append(row)
    return pd.DataFrame(rows)


def collect_monthly_details(rules, train, selection, final_test, args):
    dfs = []
    for rank, r in enumerate(rules, start=1):
        for dataset_name, df in [("TRAIN", train), ("SELECTION_VALID", selection), ("FINAL_TEST", final_test)]:
            _, _, monthly_df, _ = eval_rule(df, args.target, r.atoms, args)
            monthly_df = monthly_df.copy()
            monthly_df["rank"] = rank
            monthly_df["rule"] = r.name()
            monthly_df["dataset"] = dataset_name
            dfs.append(monthly_df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_forward_splits(df: pd.DataFrame, n_splits: int, min_train_months: int):
    months = sorted(df["month"].unique())
    if len(months) <= min_train_months + 1:
        return []
    valid_months = months[min_train_months:]
    if len(valid_months) <= n_splits:
        chosen = valid_months
    else:
        idxs = np.linspace(0, len(valid_months) - 1, n_splits).round().astype(int)
        chosen = [valid_months[i] for i in idxs]
    return [{"valid_month": m, "train_months": [x for x in months if x < m]} for m in chosen]


def fixed_rule_forward_eval(df: pd.DataFrame, target_col: str, rules: List[Rule], args) -> pd.DataFrame:
    splits = build_forward_splits(df, args.fw_splits, args.fw_min_train_months)
    rows = []
    for rank, r in enumerate(rules, start=1):
        for s in splits:
            valid_month = s["valid_month"]
            test_df = df[df["month"] == valid_month].copy().reset_index(drop=True)
            if len(test_df) == 0:
                continue
            mask = apply_rule(test_df, r.atoms)
            m = calc_metrics_df(test_df, target_col, mask, args)
            row = {"rank": rank, "rule": r.name(), "valid_month": valid_month, "n_train_months_before": len(s["train_months"])}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_forward(forward_df: pd.DataFrame, args) -> pd.DataFrame:
    if len(forward_df) == 0:
        return pd.DataFrame()
    rows = []
    for (rank, rule), g in forward_df.groupby(["rank", "rule"], sort=False):
        usable = g[g["selected_count"] >= args.fw_min_count].copy()
        if len(usable) == 0:
            rows.append({"rank": rank, "rule": rule, "fw_n_splits": len(g), "fw_n_usable_splits": 0, "fw_mean_precision": np.nan, "fw_min_precision": np.nan, "fw_mean_return": np.nan, "fw_min_return": np.nan, "fw_total_count": 0, "fw_pass_split_rate": 0.0, "fw_crash_split_count": len(g)})
            continue
        pass_mask = (usable["precision"] >= args.fw_min_precision) & (usable["lift"] >= args.fw_min_lift) & (usable["avg_return"] >= args.fw_min_avg_return)
        crash_mask = (usable["precision"] < args.fw_crash_precision) | (usable["avg_return"] < args.fw_crash_avg_return)
        rows.append({
            "rank": rank,
            "rule": rule,
            "fw_n_splits": len(g),
            "fw_n_usable_splits": len(usable),
            "fw_mean_precision": float(usable["precision"].mean()),
            "fw_min_precision": float(usable["precision"].min()),
            "fw_mean_return": float(usable["avg_return"].mean()),
            "fw_min_return": float(usable["avg_return"].min()),
            "fw_total_count": int(usable["selected_count"].sum()),
            "fw_pass_split_rate": float(pass_mask.mean()),
            "fw_crash_split_count": int(crash_mask.sum()),
        })
    return pd.DataFrame(rows)


def build_final_test_filtered_rules(rules_df: pd.DataFrame, forward_summary: pd.DataFrame, args) -> pd.DataFrame:
    if len(rules_df) == 0:
        return pd.DataFrame()
    if len(forward_summary):
        merged = rules_df.merge(forward_summary, on=["rank", "rule"], how="left")
    else:
        merged = rules_df.copy()
        for c, v in [("fw_mean_precision", np.nan), ("fw_min_precision", np.nan), ("fw_mean_return", np.nan), ("fw_min_return", np.nan), ("fw_total_count", 0), ("fw_pass_split_rate", 0.0), ("fw_crash_split_count", 999)]:
            merged[c] = v
    out = merged[
        (merged["train_precision"] >= args.final_min_train_precision)
        & (merged["selection_precision"] >= args.final_min_selection_precision)
        & (merged["selection_selected_count"] >= args.final_min_selection_count)
        & (merged["selection_lift"] >= args.final_min_selection_lift)
        & (merged["selection_avg_return"] >= args.final_min_selection_avg_return)
        & (merged["selection_month_min_month_precision"] >= args.final_min_selection_month_min_precision)
        & (merged["selection_month_min_month_return"] >= args.final_min_selection_month_min_return)
        & (merged["selection_month_crash_month_count"] <= args.final_max_selection_crash_months)
        & (merged["final_precision"] >= args.final_min_test_precision)
        & (merged["final_selected_count"] >= args.final_min_test_count)
        & (merged["final_lift"] >= args.final_min_test_lift)
        & (merged["final_avg_return"] >= args.final_min_test_avg_return)
        & (merged["final_month_crash_month_count"] <= args.final_max_test_crash_months)
        & (merged["fw_mean_precision"] >= args.final_min_fw_mean_precision)
        & (merged["fw_min_precision"] >= args.final_min_fw_min_precision)
        & (merged["fw_mean_return"] >= args.final_min_fw_mean_return)
        & (merged["fw_min_return"] >= args.final_min_fw_min_return)
        & (merged["fw_pass_split_rate"] >= args.final_min_fw_pass_split_rate)
        & (merged["fw_total_count"] >= args.final_min_fw_total_count)
        & (merged["fw_crash_split_count"] <= args.final_max_fw_crash_splits)
    ].copy()
    if len(out):
        out = out.sort_values(["final_avg_return", "final_precision", "final_selected_count", "selection_precision"], ascending=[False, False, False, False])
    return out


def build_strict50_filtered_rules(rules_df: pd.DataFrame, forward_summary: pd.DataFrame, args) -> pd.DataFrame:
    """final precision 50% 이상 핵심 후보를 별도 저장한다."""
    if len(rules_df) == 0:
        return pd.DataFrame()
    if len(forward_summary):
        merged = rules_df.merge(forward_summary, on=["rank", "rule"], how="left")
    else:
        merged = rules_df.copy()
        for c, v in [("fw_mean_precision", np.nan), ("fw_min_precision", np.nan), ("fw_mean_return", np.nan), ("fw_min_return", np.nan), ("fw_total_count", 0), ("fw_pass_split_rate", 0.0), ("fw_crash_split_count", 999)]:
            merged[c] = v

    out = merged[
        (merged["selection_precision"] >= args.strict50_min_selection_precision)
        & (merged["selection_selected_count"] >= args.strict50_min_selection_count)
        & (merged["selection_avg_return"] >= args.strict50_min_selection_avg_return)
        & (merged["final_precision"] >= args.strict50_min_test_precision)
        & (merged["final_selected_count"] >= args.strict50_min_test_count)
        & (merged["final_avg_return"] >= args.strict50_min_test_avg_return)
        & (merged["fw_mean_precision"] >= args.strict50_min_fw_mean_precision)
        & (merged["fw_pass_split_rate"] >= args.strict50_min_fw_pass_split_rate)
        & (merged["fw_crash_split_count"] <= args.final_max_fw_crash_splits)
    ].copy()
    if len(out):
        out = out.sort_values(["final_precision", "final_avg_return", "final_selected_count", "fw_pass_split_rate"], ascending=[False, False, False, False])
    return out


def or_mask(df: pd.DataFrame, rules: List[Rule]) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    for r in rules:
        mask |= apply_rule(df, r.atoms)
    return mask


def has_enough_incremental_coverage(df: pd.DataFrame, combo: Tuple[Rule, ...], min_incremental_count: int) -> bool:
    base_mask = np.zeros(len(df), dtype=bool)
    for r in combo:
        m = apply_rule(df, r.atoms)
        inc = int((m & ~base_mask).sum())
        if inc < min_incremental_count:
            return False
        base_mask |= m
    return True


def eval_or_rule_set(train, selection, final_test, df_all, rules: List[Rule], target_col: str, args):
    train_mask = or_mask(train, rules)
    selection_mask = or_mask(selection, rules)
    final_mask = or_mask(final_test, rules)
    train_m = calc_metrics_df(train, target_col, train_mask, args)
    selection_m = calc_metrics_df(selection, target_col, selection_mask, args)
    final_m = calc_metrics_df(final_test, target_col, final_mask, args)
    _, selection_mon = monthly_summary_from_mask(selection, target_col, selection_mask, args)
    rows = []
    for s in build_forward_splits(df_all, args.fw_splits, args.fw_min_train_months):
        g = df_all[df_all["month"] == s["valid_month"]].copy().reset_index(drop=True)
        if len(g) == 0:
            continue
        m = calc_metrics_df(g, target_col, or_mask(g, rules), args)
        m["valid_month"] = s["valid_month"]
        rows.append(m)
    fw_df = pd.DataFrame(rows)
    if len(fw_df) == 0:
        fw_s = {"fw_n_splits": 0, "fw_n_usable_splits": 0, "fw_mean_precision": np.nan, "fw_min_precision": np.nan, "fw_mean_return": np.nan, "fw_min_return": np.nan, "fw_total_count": 0, "fw_pass_split_rate": 0.0, "fw_crash_split_count": 999}
    else:
        usable = fw_df[fw_df["selected_count"] >= args.fw_min_count].copy()
        if len(usable) == 0:
            fw_s = {"fw_n_splits": len(fw_df), "fw_n_usable_splits": 0, "fw_mean_precision": np.nan, "fw_min_precision": np.nan, "fw_mean_return": np.nan, "fw_min_return": np.nan, "fw_total_count": 0, "fw_pass_split_rate": 0.0, "fw_crash_split_count": len(fw_df)}
        else:
            pass_mask = (usable["precision"] >= args.fw_min_precision) & (usable["lift"] >= args.fw_min_lift) & (usable["avg_return"] >= args.fw_min_avg_return)
            crash_mask = (usable["precision"] < args.fw_crash_precision) | (usable["avg_return"] < args.fw_crash_avg_return)
            fw_s = {"fw_n_splits": len(fw_df), "fw_n_usable_splits": len(usable), "fw_mean_precision": float(usable["precision"].mean()), "fw_min_precision": float(usable["precision"].min()), "fw_mean_return": float(usable["avg_return"].mean()), "fw_min_return": float(usable["avg_return"].min()), "fw_total_count": int(usable["selected_count"].sum()), "fw_pass_split_rate": float(pass_mask.mean()), "fw_crash_split_count": int(crash_mask.sum())}
    return train_m, selection_m, final_m, selection_mon, fw_s


def evaluate_or_rule_sets(rules, train, selection, final_test, df_all, args):
    if len(rules) == 0:
        return pd.DataFrame()
    candidate_rules = rules[:args.or_top_rules]
    rows = []
    for k in range(2, args.or_max_size + 1):
        for combo in itertools.combinations(candidate_rules, k):
            if not has_enough_incremental_coverage(selection, combo, args.or_min_incremental_selection_count):
                continue
            train_m, selection_m, final_m, selection_mon, fw_s = eval_or_rule_set(train, selection, final_test, df_all, list(combo), args.target, args)
            if not np.isfinite(selection_m["precision"]) or not np.isfinite(final_m["precision"]):
                continue
            if selection_m["precision"] < args.or_min_selection_precision:
                continue
            if selection_m["selected_count"] < args.or_min_selection_count:
                continue
            if selection_m["lift"] < args.or_min_selection_lift:
                continue
            if selection_m["avg_return"] < args.or_min_selection_avg_return:
                continue
            if final_m["selected_count"] < args.or_min_test_count:
                continue
            if final_m["precision"] < args.or_min_test_precision:
                continue
            if final_m["lift"] < args.or_min_test_lift:
                continue
            if final_m["avg_return"] < args.or_min_test_avg_return:
                continue
            if np.isfinite(fw_s.get("fw_mean_precision", np.nan)) and fw_s["fw_mean_precision"] < args.or_min_fw_mean_precision:
                continue
            if np.isfinite(fw_s.get("fw_min_precision", np.nan)) and fw_s["fw_min_precision"] < args.or_min_fw_min_precision:
                continue
            if np.isfinite(fw_s.get("fw_mean_return", np.nan)) and fw_s["fw_mean_return"] < args.or_min_fw_mean_return:
                continue
            if fw_s.get("fw_pass_split_rate", 0.0) < args.or_min_fw_pass_split_rate:
                continue
            if fw_s.get("fw_crash_split_count", 999) > args.or_max_fw_crash_splits:
                continue
            score = (
                selection_m["precision"] * 95
                + final_m["precision"] * 115
                + safe_num(selection_m.get("avg_return")) * 35
                + safe_num(final_m.get("avg_return")) * 50
                + math.log1p(selection_m["selected_count"]) * args.or_count_weight
                + math.log1p(final_m["selected_count"]) * args.or_test_count_weight
                + safe_num(fw_s.get("fw_mean_precision")) * 55
                + safe_num(fw_s.get("fw_min_precision")) * 45
                + safe_num(fw_s.get("fw_mean_return")) * 25
                - fw_s.get("fw_crash_split_count", 999) * 25
            )
            row = {"or_size": k, "selection_count": selection_m["selected_count"], "selection_precision": selection_m["precision"], "selection_precision_lcb": selection_m["precision_lcb"], "selection_lift": selection_m["lift"], "selection_avg_return": selection_m["avg_return"], "selection_coverage": selection_m["coverage"], "final_count": final_m["selected_count"], "final_precision": final_m["precision"], "final_precision_lcb": final_m["precision_lcb"], "final_lift": final_m["lift"], "final_avg_return": final_m["avg_return"], "final_coverage": final_m["coverage"], "or_score": score, "rules": " || ".join(r.name() for r in combo)}
            row.update(fw_s)
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["or_score", "final_avg_return", "final_precision", "final_count", "selection_precision"], ascending=[False, False, False, False, False])
    return out



# =============================================================================
# Auto export selected mined rules into buy-rule Python module
# =============================================================================

RULE_EXPORT_LABELS = [
    ("rule_001_precision_high", "HIGH_PRECISION", "고확률형: final precision / 기대수익률 우선 룰"),
    ("rule_002_stable_forward", "STABLE_FORWARD", "안정형: forward 안정성 / crash 최소화 우선 룰"),
    ("rule_003_coverage_expand", "COVERAGE_EXPAND", "커버리지형: final count / coverage 확장 우선 룰"),
]


def _as_float_safe(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _as_int_safe(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _rule_features_from_string(rule: str) -> List[str]:
    feats = []
    for part in str(rule).split(" AND "):
        part = part.strip()
        if " >= " in part:
            feats.append(part.split(" >= ", 1)[0].strip())
        elif " <= " in part:
            feats.append(part.split(" <= ", 1)[0].strip())
    return feats


def _format_threshold_for_code(x) -> str:
    v = _as_float_safe(x)
    if not np.isfinite(v):
        return "np.nan"
    # Short but stable. Avoid scientific notation for normal trading thresholds.
    if abs(v) >= 1e-4 and abs(v) < 1e6:
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    return f"{v:.10g}"


def _parse_rule_to_condition_lines(rule: str) -> List[str]:
    lines = []
    for raw in str(rule).split(" AND "):
        part = raw.strip()
        if " >= " in part:
            feat, th = part.split(" >= ", 1)
            op = ">="
        elif " <= " in part:
            feat, th = part.split(" <= ", 1)
            op = "<="
        else:
            raise ValueError(f"Cannot parse rule atom: {part}")
        feat = feat.strip()
        th_s = _format_threshold_for_code(th.strip())
        lines.append(f'(df["{feat}"] {op} {th_s})')
    return lines


def _choose_first_unique(df: pd.DataFrame, used_rules: Set[str]) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    for _, row in df.iterrows():
        rule = str(row.get("rule", ""))
        if rule and rule not in used_rules:
            used_rules.add(rule)
            return row
    return None


def select_rules_for_export(final_df: pd.DataFrame, strict_df: pd.DataFrame, rules_df: pd.DataFrame, args) -> pd.DataFrame:
    """Pick exactly up to three representative rules: high precision, stable, coverage.

    Priority source:
    1) final_test_filtered_rules_final50_ret2
    2) strict50_ret2
    3) selected rules satisfying final precision/return filter
    """
    frames = []
    for frame in [final_df, strict_df, rules_df]:
        if frame is not None and len(frame):
            frames.append(frame.copy())
    if not frames:
        return pd.DataFrame()

    base = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["rule"]).copy()

    # Ensure forward columns exist even if source is rules_df only.
    defaults = {
        "fw_pass_split_rate": 0.0,
        "fw_crash_split_count": 999,
        "fw_min_return": np.nan,
        "fw_mean_return": np.nan,
        "fw_min_precision": np.nan,
        "fw_mean_precision": np.nan,
        "fw_total_count": 0,
    }
    for c, v in defaults.items():
        if c not in base.columns:
            base[c] = v

    required_cols = ["final_precision", "final_avg_return", "final_selected_count"]
    for c in required_cols:
        if c not in base.columns:
            base[c] = np.nan

    candidates = base[
        (base["final_precision"] >= args.buy_rules_min_final_precision)
        & (base["final_avg_return"] >= args.buy_rules_min_final_avg_return)
        & (base["final_selected_count"] >= args.buy_rules_min_final_count)
        & (base["fw_crash_split_count"].fillna(999) <= args.buy_rules_max_fw_crash_splits)
    ].copy()

    if len(candidates) == 0:
        candidates = base[
            (base["final_precision"] >= args.buy_rules_min_final_precision)
            & (base["final_avg_return"] >= args.buy_rules_min_final_avg_return)
        ].copy()

    if len(candidates) == 0:
        return pd.DataFrame()

    used: Set[str] = set()
    selected_rows = []

    high_precision = candidates.sort_values(
        ["final_precision", "final_avg_return", "fw_pass_split_rate", "final_selected_count"],
        ascending=[False, False, False, False],
    )
    r = _choose_first_unique(high_precision, used)
    if r is not None:
        selected_rows.append((RULE_EXPORT_LABELS[0], r))

    stable = candidates.sort_values(
        ["fw_crash_split_count", "fw_pass_split_rate", "fw_min_return", "fw_min_precision", "final_avg_return", "final_precision"],
        ascending=[True, False, False, False, False, False],
    )
    r = _choose_first_unique(stable, used)
    if r is not None:
        selected_rows.append((RULE_EXPORT_LABELS[1], r))

    coverage = candidates.sort_values(
        ["final_selected_count", "final_precision", "final_avg_return", "fw_pass_split_rate"],
        ascending=[False, False, False, False],
    )
    r = _choose_first_unique(coverage, used)
    if r is not None:
        selected_rows.append((RULE_EXPORT_LABELS[2], r))

    rows = []
    for (rule_name, style, description), row in selected_rows:
        d = row.to_dict()
        d["export_rule_name"] = rule_name
        d["export_style"] = style
        d["export_description"] = description
        rows.append(d)
    return pd.DataFrame(rows)


def write_buy_rules_module(selected_df: pd.DataFrame, out_path: str, args, source_script_name: str) -> None:
    if selected_df is None or len(selected_df) == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# No rules selected for export.\n")
        return

    rule_names = list(selected_df["export_rule_name"])
    required_features: Set[str] = set()
    for rule in selected_df["rule"]:
        required_features.update(_rule_features_from_string(rule))
    required_features = sorted(required_features)

    lines = []
    lines.append("# auto-generated: lowscan good buy rules")
    lines.append(f"# source: {source_script_name}")
    lines.append(f"# feature_set: {args.feature_set}")
    lines.append("# split: train / selection_valid / final_test applied")
    lines.append(f"# target: {args.target} == 1")
    lines.append("# final filter: precision >= %.2f, avg_return >= %.2f" % (args.buy_rules_min_final_precision, args.buy_rules_min_final_avg_return))
    lines.append("# excluded features: today_pct, price_power_value, body_value_power" + (", intraday_return" if args.feature_set == "all_no_today_no_intraday" else ""))
    lines.append("# usage:")
    lines.append("#    import lowscan_auto_buy_rules as lowscan_rules")
    lines.append("#    buy_conditions = lowscan_rules.build_conditions(df)")
    lines.append("#    buy_mask = lowscan_rules.build_mask(df)")
    lines.append("#    df_buy = df[buy_mask].copy()")
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("RULE_NAMES = [")
    for name in rule_names:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")
    excluded = ["today_pct", "price_power_value", "body_value_power"]
    if args.feature_set == "all_no_today_no_intraday":
        excluded.append("intraday_return")
    lines.append("EXCLUDED_FEATURES = {")
    for f in excluded:
        lines.append(f'    "{f}",')
    lines.append("}")
    lines.append("")
    lines.append("REQUIRED_COLUMNS = [")
    for f in required_features:
        lines.append(f'    "{f}",')
    lines.append("]")
    lines.append("")
    lines.append("RULE_META = {")
    for _, row in selected_df.iterrows():
        name = row["export_rule_name"]
        lines.append(f'    "{name}": {{')
        lines.append(f'        "style": "{row["export_style"]}",')
        lines.append(f'        "description": "{row["export_description"]}",')
        lines.append(f'        "target": "{args.target}",')
        for k in [
            "selection_selected_count", "selection_precision", "selection_avg_return",
            "final_selected_count", "final_precision", "final_avg_return",
            "fw_mean_precision", "fw_min_precision", "fw_mean_return", "fw_min_return",
            "fw_pass_split_rate", "fw_crash_split_count",
        ]:
            if k in row.index:
                val = row[k]
                if k.endswith("count") or k in {"selection_selected_count", "final_selected_count", "fw_crash_split_count"}:
                    lines.append(f'        "{k}": {_as_int_safe(val)},')
                else:
                    lines.append(f'        "{k}": {_format_threshold_for_code(val)},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("def _require_columns(df, columns=REQUIRED_COLUMNS):")
    lines.append("    missing = [c for c in columns if c not in df.columns]")
    lines.append("    if missing:")
    lines.append("        raise KeyError(f'Missing required columns for lowscan rules: {missing}')")
    lines.append("")
    lines.append("def build_conditions(df, validate: bool = True):")
    lines.append("    if validate:")
    lines.append("        _require_columns(df)")
    lines.append("    conditions = {")
    for _, row in selected_df.iterrows():
        name = row["export_rule_name"]
        style = row["export_style"]
        rule = row["rule"]
        lines.append(f"        # {style}")
        lines.append(f"        # final_count={_as_int_safe(row.get('final_selected_count'))}, final_precision={_as_float_safe(row.get('final_precision')):.2%}, final_avg_return={_as_float_safe(row.get('final_avg_return')):+.2f}%")
        if "fw_pass_split_rate" in row.index:
            lines.append(f"        # fw_pass_split_rate={_as_float_safe(row.get('fw_pass_split_rate')):.2%}, fw_crash_split_count={_as_int_safe(row.get('fw_crash_split_count'))}")
        condition_lines = _parse_rule_to_condition_lines(rule)
        lines.append(f'        "{name}":')
        for i, cond_line in enumerate(condition_lines):
            suffix = " &" if i < len(condition_lines) - 1 else ","
            lines.append(f"            {cond_line}{suffix}")
        lines.append("")
    lines.append("    }")
    lines.append("    return conditions")
    lines.append("")
    lines.append("def build_mask(df, validate: bool = True):")
    lines.append("    mask = np.zeros(len(df), dtype=bool)")
    lines.append("    for cond in build_conditions(df, validate=validate).values():")
    lines.append("        mask |= np.asarray(cond, dtype=bool)")
    lines.append("    return mask")
    lines.append("")
    lines.append("def build_rule_name_series(df, sep: str = ',', validate: bool = True):")
    lines.append("    conditions = build_conditions(df, validate=validate)")
    lines.append("    names = []")
    lines.append("    for i in range(len(df)):")
    lines.append("        matched = []")
    lines.append("        for name, cond in conditions.items():")
    lines.append("            if bool(cond.iloc[i] if hasattr(cond, 'iloc') else cond[i]):")
    lines.append("                matched.append(name)")
    lines.append("        names.append(sep.join(matched))")
    lines.append("    return names")
    lines.append("")
    lines.append("def build_rule_count_series(df, validate: bool = True):")
    lines.append("    conditions = build_conditions(df, validate=validate)")
    lines.append("    count = np.zeros(len(df), dtype=int)")
    lines.append("    for cond in conditions.values():")
    lines.append("        count += np.asarray(cond, dtype=bool).astype(int)")
    lines.append("    return count")
    lines.append("")
    lines.append("def build_priority_score(df, validate: bool = True):")
    lines.append("    conditions = build_conditions(df, validate=validate)")
    lines.append("    weights = {")
    lines.append("        'rule_001_precision_high': 3.0,")
    lines.append("        'rule_002_stable_forward': 3.0,")
    lines.append("        'rule_003_coverage_expand': 2.0,")
    lines.append("    }")
    lines.append("    score = np.zeros(len(df), dtype=float)")
    lines.append("    for name, cond in conditions.items():")
    lines.append("        score += np.asarray(cond, dtype=bool).astype(float) * weights.get(name, 1.0)")
    lines.append("    return score")
    lines.append("")
    lines.append("def apply_rules(df, copy: bool = True):")
    lines.append("    out = df.copy() if copy else df")
    lines.append("    out['lowscan_buy_signal'] = build_mask(out)")
    lines.append("    out['lowscan_rule_names'] = build_rule_name_series(out)")
    lines.append("    out['lowscan_rule_count'] = build_rule_count_series(out)")
    lines.append("    out['lowscan_priority_score'] = build_priority_score(out)")
    lines.append("    return out[out['lowscan_buy_signal']].copy() if copy else out")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    print('This module is intended to be imported and applied to a pandas DataFrame.')")
    lines.append("    for name in RULE_NAMES:")
    lines.append("        print(f'- {name}: {RULE_META[name]}')")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def feature_decile_report(df: pd.DataFrame, features: List[str], target_col: str, args) -> pd.DataFrame:
    rows = []
    returns = build_trade_return_series(df, target_col, args)
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        tmp = pd.DataFrame({"x": s, "y": df[target_col].astype(int).values, "ret": returns}).dropna(subset=["x", "y"])
        if tmp["x"].nunique() < 8 or len(tmp) < 50:
            continue
        try:
            tmp["bin"] = pd.qcut(tmp["x"], q=10, duplicates="drop")
        except Exception:
            continue
        base_rate = tmp["y"].mean()
        for b, g in tmp.groupby("bin", observed=True):
            count = len(g)
            precision = g["y"].mean() if count else np.nan
            rows.append({"feature": f, "bin": str(b), "bin_left": float(b.left), "bin_right": float(b.right), "count": int(count), "positive_count": int(g["y"].sum()), "precision": precision, "lift": precision / base_rate if base_rate > 0 else np.nan, "avg_return": float(g["ret"].mean()), "median_return": float(g["ret"].median()), "base_rate": float(base_rate)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["feature", "bin_left"])
    return out


def cumulative_threshold_report(df: pd.DataFrame, features: List[str], target_col: str, q_grid: List[float], args) -> pd.DataFrame:
    rows = []
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = s.dropna()
        if vals.nunique() < 8:
            continue
        thresholds = []
        for q in q_grid:
            v = vals.quantile(q)
            if np.isfinite(v):
                thresholds.append((q, round(float(v), 10)))
        for q, th in sorted(set(thresholds), key=lambda x: x[1]):
            for op in [">=", "<="]:
                mask = (s >= th).fillna(False).to_numpy() if op == ">=" else (s <= th).fillna(False).to_numpy()
                m = calc_metrics_df(df, target_col, mask, args)
                row = {"feature": f, "op": op, "quantile": q, "threshold": th}
                row.update(m)
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["feature", "op", "quantile"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="stable_rule_miner_final_v4_7_no_today_ret2_d5_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--feature-set", default="all_no_today", choices=["core", "core2", "no_today", "balanced", "all", "all_no_today", "all_no_today_no_intraday"])
    parser.add_argument("--features", default=None, help="comma-separated feature list. today_pct, price_power_value, body_value_power are still forcibly excluded.")
    parser.add_argument("--threshold-mode", default="quantile", choices=["quantile"])
    parser.add_argument("--quantile-grid", default="v4", choices=["decile", "v4", "fine"])
    parser.add_argument("--manual-thresholds", default="off", choices=["off", "on"])
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--selection-valid-ratio", type=float, default=0.20)
    parser.add_argument("--final-test-ratio", type=float, default=0.20)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=350)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--target-pct", type=float, default=10.0)
    parser.add_argument("--stop-loss", type=float, default=-6.0)
    parser.add_argument("--same-day-return", type=float, default=-6.0)
    parser.add_argument("--no-hit-return-col", default="validation_close_rate7")

    parser.add_argument("--min-train-count", type=int, default=50)
    parser.add_argument("--min-selection-count", type=int, default=25)
    parser.add_argument("--min-train-precision", type=float, default=0.34)
    parser.add_argument("--min-selection-precision", type=float, default=0.42)
    parser.add_argument("--min-train-lift", type=float, default=1.02)
    parser.add_argument("--min-selection-lift", type=float, default=1.08)
    parser.add_argument("--min-selection-lcb", type=float, default=0.30)
    parser.add_argument("--min-selection-avg-return", type=float, default=0.20)
    parser.add_argument("--max-precision-gap", type=float, default=0.28)
    parser.add_argument("--beam-min-count", type=int, default=45)
    parser.add_argument("--beam-min-lift", type=float, default=1.00)
    parser.add_argument("--fast-min-selection-count", type=int, default=18)
    parser.add_argument("--fast-min-selection-precision", type=float, default=0.34)
    parser.add_argument("--fast-min-selection-lift", type=float, default=1.05)
    parser.add_argument("--fast-min-selection-avg-return", type=float, default=-0.30)
    parser.add_argument("--min-month-count", type=int, default=5)
    parser.add_argument("--month-pass-precision", type=float, default=0.42)
    parser.add_argument("--month-pass-lift", type=float, default=1.05)
    parser.add_argument("--month-pass-avg-return", type=float, default=0.00)
    parser.add_argument("--month-crash-precision", type=float, default=0.30)
    parser.add_argument("--month-crash-avg-return", type=float, default=-1.00)
    parser.add_argument("--min-selection-usable-months", type=int, default=2)
    parser.add_argument("--min-selection-pass-month-rate", type=float, default=0.20)
    parser.add_argument("--min-selection-month-min-precision", type=float, default=0.30)
    parser.add_argument("--min-selection-month-min-return", type=float, default=-2.00)
    parser.add_argument("--max-selection-crash-months", type=int, default=2)
    parser.add_argument("--selection-count-weight", type=float, default=6.0)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--use-corr-pruning", action="store_true")
    parser.add_argument("--dup-count-tol", type=float, default=0.10)
    parser.add_argument("--dup-precision-gain", type=float, default=0.01)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--simplify-precision-drop", type=float, default=0.015)
    parser.add_argument("--simplify-min-count-ratio", type=float, default=0.85)

    parser.add_argument("--fw-top-rules", type=int, default=100)
    parser.add_argument("--fw-splits", type=int, default=6)
    parser.add_argument("--fw-min-train-months", type=int, default=6)
    parser.add_argument("--fw-min-count", type=int, default=5)
    parser.add_argument("--fw-min-precision", type=float, default=0.38)
    parser.add_argument("--fw-min-lift", type=float, default=1.05)
    parser.add_argument("--fw-min-avg-return", type=float, default=0.00)
    parser.add_argument("--fw-crash-precision", type=float, default=0.30)
    parser.add_argument("--fw-crash-avg-return", type=float, default=-1.00)

    parser.add_argument("--final-min-train-precision", type=float, default=0.38)
    parser.add_argument("--final-min-selection-precision", type=float, default=0.60)
    parser.add_argument("--final-min-selection-count", type=int, default=25)
    parser.add_argument("--final-min-selection-lift", type=float, default=1.20)
    parser.add_argument("--final-min-selection-avg-return", type=float, default=1.00)
    parser.add_argument("--final-min-selection-month-min-precision", type=float, default=0.30)
    parser.add_argument("--final-min-selection-month-min-return", type=float, default=-1.50)
    parser.add_argument("--final-max-selection-crash-months", type=int, default=2)
    parser.add_argument("--final-min-test-precision", type=float, default=0.50)
    parser.add_argument("--final-min-test-count", type=int, default=30)
    parser.add_argument("--final-min-test-lift", type=float, default=1.05)
    parser.add_argument("--final-min-test-avg-return", type=float, default=2.00)
    parser.add_argument("--final-max-test-crash-months", type=int, default=3)
    parser.add_argument("--final-min-fw-mean-precision", type=float, default=0.50)
    parser.add_argument("--final-min-fw-min-precision", type=float, default=0.35)
    parser.add_argument("--final-min-fw-mean-return", type=float, default=0.30)
    parser.add_argument("--final-min-fw-min-return", type=float, default=-1.50)
    parser.add_argument("--final-min-fw-pass-split-rate", type=float, default=0.65)
    parser.add_argument("--final-min-fw-total-count", type=int, default=30)
    parser.add_argument("--final-max-fw-crash-splits", type=int, default=2)

    # 별도 출력용: final precision 50% 이상 핵심 룰셋 필터
    parser.add_argument("--strict50-min-selection-precision", type=float, default=0.65)
    parser.add_argument("--strict50-min-selection-count", type=int, default=20)
    parser.add_argument("--strict50-min-selection-avg-return", type=float, default=2.00)
    parser.add_argument("--strict50-min-test-precision", type=float, default=0.50)
    parser.add_argument("--strict50-min-test-count", type=int, default=30)
    parser.add_argument("--strict50-min-test-avg-return", type=float, default=2.00)
    parser.add_argument("--strict50-min-fw-mean-precision", type=float, default=0.50)
    parser.add_argument("--strict50-min-fw-pass-split-rate", type=float, default=0.65)

    parser.add_argument("--or-top-rules", type=int, default=16)
    parser.add_argument("--or-max-size", type=int, default=4)
    parser.add_argument("--or-min-incremental-selection-count", type=int, default=5)
    parser.add_argument("--or-min-selection-precision", type=float, default=0.58)
    parser.add_argument("--or-min-selection-count", type=int, default=50)
    parser.add_argument("--or-min-selection-lift", type=float, default=1.10)
    parser.add_argument("--or-min-selection-avg-return", type=float, default=1.00)
    parser.add_argument("--or-min-test-precision", type=float, default=0.50)
    parser.add_argument("--or-min-test-count", type=int, default=50)
    parser.add_argument("--or-min-test-lift", type=float, default=1.00)
    parser.add_argument("--or-min-test-avg-return", type=float, default=2.00)
    parser.add_argument("--or-min-fw-mean-precision", type=float, default=0.50)
    parser.add_argument("--or-min-fw-min-precision", type=float, default=0.35)
    parser.add_argument("--or-min-fw-mean-return", type=float, default=0.30)
    parser.add_argument("--or-min-fw-pass-split-rate", type=float, default=0.65)
    parser.add_argument("--or-max-fw-crash-splits", type=int, default=2)
    parser.add_argument("--or-count-weight", type=float, default=8.0)
    parser.add_argument("--or-test-count-weight", type=float, default=10.0)

    parser.add_argument("--type-hp-selection-precision", type=float, default=0.60)
    parser.add_argument("--type-hp-final-precision", type=float, default=0.50)
    parser.add_argument("--type-hp-selection-lcb", type=float, default=0.38)
    parser.add_argument("--type-hp-avg-return", type=float, default=2.00)
    parser.add_argument("--type-hc-selection-count", type=int, default=70)
    parser.add_argument("--type-hc-final-count", type=int, default=60)
    parser.add_argument("--type-hc-selection-precision", type=float, default=0.42)
    parser.add_argument("--type-hc-avg-return", type=float, default=0.10)
    parser.add_argument("--type-st-pass-month-rate", type=float, default=0.50)
    parser.add_argument("--type-st-max-crash-months", type=int, default=1)
    parser.add_argument("--type-st-avg-return", type=float, default=0.00)
    parser.add_argument("--type-ag-selection-precision", type=float, default=0.38)
    parser.add_argument("--type-ag-avg-return", type=float, default=0.10)
    parser.add_argument("--type-ag-selection-count", type=int, default=60)
    parser.add_argument("--type-df-selection-precision", type=float, default=0.50)
    parser.add_argument("--type-df-selection-lcb", type=float, default=0.36)
    parser.add_argument("--type-df-max-crash-months", type=int, default=1)
    parser.add_argument("--type-df-avg-return", type=float, default=0.20)

    # Auto-export buy rule module from mined result.
    parser.add_argument("--export-buy-rules", action="store_true", default=True)
    parser.add_argument("--buy-rules-filename", default="lowscan_positive_rules_auto.py")
    parser.add_argument("--buy-rules-min-final-precision", type=float, default=0.50)
    parser.add_argument("--buy-rules-min-final-avg-return", type=float, default=2.00)
    parser.add_argument("--buy-rules-min-final-count", type=int, default=30)
    parser.add_argument("--buy-rules-max-fw-crash-splits", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df_raw = pd.read_csv(args.csv, low_memory=False)
    date_col = args.date_col or find_date_col(df_raw)
    if date_col is None:
        raise ValueError("date column not found. Use --date-col today")
    df = prepare_df(df_raw, args.target, date_col)
    requested_features = choose_features(args.feature_set, args.features)
    features = [f for f in requested_features if f in df.columns]
    missing = [f for f in requested_features if f not in df.columns]
    if missing:
        print("[WARN] missing features:", missing)

    train, selection, final_test, split_info = split_train_selection_test_by_date(df, date_col, args.train_ratio, args.selection_valid_ratio, args.final_test_ratio)
    print("=" * 100)
    print("[INFO] rows:", len(df))
    print("[INFO] target:", args.target)
    print("[INFO] all base_rate:", df[args.target].mean())
    print("[INFO] train rows:", len(train), "base_rate:", train[args.target].mean())
    print("[INFO] selection_valid rows:", len(selection), "base_rate:", selection[args.target].mean())
    print("[INFO] final_test rows:", len(final_test), "base_rate:", final_test[args.target].mean())
    print("[INFO] breakeven precision:", breakeven_precision(args))
    print("[INFO] feature_set:", args.feature_set)
    print("[INFO] features:", features)
    print("[INFO] split_info:", split_info)
    print("=" * 100)

    pd.DataFrame([split_info]).to_csv(os.path.join(args.out, "00_split_info.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features, "allowed_ops": [",".join(ALLOWED_OPS.get(f, [">=", "<="])) for f in features], "is_interval": [f in INTERVAL_FEATURES for f in features]}).to_csv(os.path.join(args.out, "00_features_used.csv"), index=False, encoding="utf-8-sig")
    feature_decile_report(train, features, args.target, args).to_csv(os.path.join(args.out, "00_feature_decile_report_train.csv"), index=False, encoding="utf-8-sig")
    q_grid = quantile_grid(args.quantile_grid)
    cumulative_threshold_report(train, features, args.target, q_grid, args).to_csv(os.path.join(args.out, "00_feature_threshold_report_train.csv"), index=False, encoding="utf-8-sig")

    high_corr_df, corr_pairs = build_corr_pairs(train, features, args.corr_threshold)
    high_corr_df.to_csv(os.path.join(args.out, "00_high_corr_pairs.csv"), index=False, encoding="utf-8-sig")
    print("[HIGH CORR PAIRS]", "none" if len(high_corr_df) == 0 else f"{len(high_corr_df)} pairs")

    extra = default_extra_thresholds() if args.manual_thresholds == "on" else {}
    atoms = make_atoms(train, features, q_grid, extra, ALLOWED_OPS)
    pd.DataFrame([{"feature": a.feature, "op": a.op, "threshold": a.threshold, "source": a.source, "atom": a.name()} for a in atoms]).to_csv(os.path.join(args.out, "01_atoms.csv"), index=False, encoding="utf-8-sig")
    print("[INFO] atoms:", len(atoms))

    rules = search_rules(train, selection, final_test, atoms, corr_pairs, args)
    if args.simplify:
        print("[INFO] simplifying rules")
        simplified = [simplify_rule_by_dropping_atoms(r, train, selection, final_test, args) for r in rules]
        simplified = sorted(simplified, key=lambda r: r.score, reverse=True)
        deduped: List[Rule] = []
        for r in simplified:
            if is_near_duplicate(r, deduped, args):
                continue
            deduped.append(r)
            if len(deduped) >= args.top_k:
                break
        rules = deduped

    rules_df = rules_to_df(rules, args)
    rules_df.to_csv(os.path.join(args.out, "02_selected_rules.csv"), index=False, encoding="utf-8-sig")
    if len(rules_df):
        for col, fn in [
            ("pass_selection_50_n30", "03_pass_selection_50_n30.csv"),
            ("pass_selection_60_n25", "04_pass_selection_60_n25.csv"),
            ("pass_selection_70_n20", "05_pass_selection_70_n20.csv"),
            ("pass_final_40_n40", "06_pass_final_40_n40.csv"),
            ("pass_final_45_n30", "07_pass_final_45_n30.csv"),
            ("pass_final_50_n20", "08_pass_final_50_n20.csv"),
            ("type_high_precision", "09_rules_high_precision.csv"),
            ("type_high_coverage", "10_rules_high_coverage.csv"),
            ("type_stable", "11_rules_stable.csv"),
            ("type_aggressive", "12_rules_aggressive.csv"),
            ("type_defensive", "13_rules_defensive.csv"),
        ]:
            rules_df[rules_df[col]].to_csv(os.path.join(args.out, fn), index=False, encoding="utf-8-sig")
    else:
        for fn in ["03_pass_selection_50_n30.csv", "04_pass_selection_60_n25.csv", "05_pass_selection_70_n20.csv", "06_pass_final_40_n40.csv", "07_pass_final_45_n30.csv", "08_pass_final_50_n20.csv", "09_rules_high_precision.csv", "10_rules_high_coverage.csv", "11_rules_stable.csv", "12_rules_aggressive.csv", "13_rules_defensive.csv"]:
            pd.DataFrame().to_csv(os.path.join(args.out, fn), index=False, encoding="utf-8-sig")

    monthly_df = collect_monthly_details(rules, train, selection, final_test, args)
    monthly_df.to_csv(os.path.join(args.out, "14_monthly_details.csv"), index=False, encoding="utf-8-sig")
    fw_df = fixed_rule_forward_eval(df, args.target, rules[:args.fw_top_rules], args)
    fw_df.to_csv(os.path.join(args.out, "15_fixed_rule_forward_eval.csv"), index=False, encoding="utf-8-sig")
    fw_summary = summarize_forward(fw_df, args)
    fw_summary.to_csv(os.path.join(args.out, "16_fixed_rule_forward_summary.csv"), index=False, encoding="utf-8-sig")
    final_test_filtered = build_final_test_filtered_rules(rules_df, fw_summary, args)
    final_test_filtered.to_csv(os.path.join(args.out, "17_final_test_filtered_rules_final50_ret2.csv"), index=False, encoding="utf-8-sig")
    strict50_filtered = build_strict50_filtered_rules(rules_df, fw_summary, args)
    strict50_filtered.to_csv(os.path.join(args.out, "18_final_test_filtered_rules_strict50_ret2.csv"), index=False, encoding="utf-8-sig")
    # backward-compatible alias
    final_test_filtered.to_csv(os.path.join(args.out, "17_final_test_filtered_rules.csv"), index=False, encoding="utf-8-sig")
    or_df = evaluate_or_rule_sets(rules, train, selection, final_test, df, args)
    or_df.to_csv(os.path.join(args.out, "19_or_rule_sets_final_test.csv"), index=False, encoding="utf-8-sig")

    exported_buy_rules = pd.DataFrame()
    if args.export_buy_rules:
        exported_buy_rules = select_rules_for_export(final_test_filtered, strict50_filtered, rules_df, args)
        exported_buy_rules.to_csv(os.path.join(args.out, "20_auto_buy_rules_selected.csv"), index=False, encoding="utf-8-sig")
        write_buy_rules_module(
            exported_buy_rules,
            os.path.join(args.out, args.buy_rules_filename),
            args,
            source_script_name=os.path.basename(__file__),
        )
        print("[INFO] exported buy rules:", os.path.join(args.out, args.buy_rules_filename))

    summary = {
        "rows": len(df),
        "target": args.target,
        "base_rate_all": df[args.target].mean(),
        "base_rate_train": train[args.target].mean(),
        "base_rate_selection": selection[args.target].mean(),
        "base_rate_final": final_test[args.target].mean(),
        "breakeven_precision": breakeven_precision(args),
        "features_count": len(features),
        "atoms_count": len(atoms),
        "selected_rules": len(rules_df),
        "final_test_filtered_final50_ret2": len(final_test_filtered),
        "final_test_filtered_strict50_ret2": len(strict50_filtered),
        "or_rule_sets_final_test": len(or_df),
        "exported_buy_rules": len(exported_buy_rules),
    }
    if len(rules_df):
        for col in ["pass_selection_50_n30", "pass_selection_60_n25", "pass_selection_70_n20", "pass_final_40_n40", "pass_final_45_n30", "pass_final_50_n20", "pass_final_50_ret2_n30", "type_high_precision", "type_high_coverage", "type_stable", "type_aggressive", "type_defensive"]:
            summary[col] = int(rules_df[col].sum())
        summary["best_selection_precision"] = rules_df["selection_precision"].max()
        summary["best_final_precision"] = rules_df["final_precision"].max()
        summary["best_selection_avg_return"] = rules_df["selection_avg_return"].max()
        summary["best_final_avg_return"] = rules_df["final_avg_return"].max()
        summary["max_selection_count"] = rules_df["selection_selected_count"].max()
        summary["max_final_count"] = rules_df["final_selected_count"].max()
    pd.DataFrame([summary]).to_csv(os.path.join(args.out, "99_run_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if len(rules_df):
        show_cols = ["rank", "rule", "selection_selected_count", "selection_precision", "selection_avg_return", "final_selected_count", "final_precision", "final_avg_return", "precision_gap_selection_final_abs", "selection_month_pass_month_rate", "selection_month_crash_month_count", "pass_final_45_n30", "pass_final_50_ret2_n30", "type_high_precision", "type_stable", "type_defensive"]
        show_cols = [c for c in show_cols if c in rules_df.columns]
        print("\n[TOP RULES]")
        print(rules_df[show_cols].head(40).to_string(index=False))
    print("=" * 100)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 100)


if __name__ == "__main__":
    main()
