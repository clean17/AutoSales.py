#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stable_rule_miner_final_v4.py

v2 = 단일 룰 안정성 검증형
v3 = v2 + 구간형 조건 + OR 룰셋
v4 = v3 구조를 더 엄격하게 만든 3분할 최종검증형

v3 개선점
1) train / selection_valid / final_test 3분할
   - train: atom / rule 후보 생성의 기준
   - selection_valid: 룰 선택, threshold, OR 조합 선택에 사용
   - final_test: 최종 선택된 단일 룰 / OR 룰셋을 1회 평가만 함
   - 즉 final_test는 룰 선택에 사용하지 않음

2) 날짜 단위 split
   - 같은 today 날짜가 train/valid/test에 섞이지 않게 split

3) pass flag 이름 명확화
   - pass_valid_60_n60
   - pass_valid_65_n50
   - pass_valid_70_n30
   - pass_valid_70_n60

4) fixed-rule forward robustness 명칭 정리
   - 기존 walk-forward는 재학습이 아니라 고정 룰을 여러 미래 구간에 적용하는 안정성 평가

5) OR 룰셋에 incremental coverage 필터 추가
   - OR에 새 룰을 추가했을 때 selection_valid에서 새로 잡는 행이 너무 적으면 제외

6) feature-set 옵션 추가
   - core: 결과에서 강했던 핵심 피쳐만 사용
   - balanced: 기본값. core + 일부 보조 피쳐
   - all: 사용자가 제시한 전체 피쳐 사용

추천 실행:
python stable_rule_miner_final_v4.py ^
  --csv csv/low_result_7_desc.csv ^
  --out stable_rule_miner_final_v4_out ^
  --date-col today ^
  --feature-set balanced ^
  --train-ratio 0.60 ^
  --selection-valid-ratio 0.20 ^
  --final-test-ratio 0.20 ^
  --max-depth 6 ^
  --beam-width 1500 ^
  --top-k 150 ^
  --simplify

전체 피쳐 실험:
python stable_rule_miner_final_v4.py ^
  --csv csv/low_result_7_desc.csv ^
  --out stable_rule_miner_final_v4_out_all ^
  --date-col today ^
  --feature-set all ^
  --train-ratio 0.60 ^
  --selection-valid-ratio 0.20 ^
  --final-test-ratio 0.20 ^
  --max-depth 5 ^
  --beam-width 1200 ^
  --top-k 150 ^
  --simplify
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


TARGET_COL = "target_before_stop_7"


# =============================================================================
# Feature sets
# =============================================================================

FEATURE_GROUPS = {
    "vol5": "VOLATILITY",
    "vol_ratio_5_15": "VOLATILITY",
    "today_pct": "PRICE",
    "max_drop_7d": "DROP",
    "gap_pct": "GAP",
    "pct_vs_lastweek": "WEEK_POSITION",
    "dist_to_ma5": "POSITION",
    "ma5_chg_rate": "TREND",
    "today_tr_val_eok": "VOLUME",
    "BB_perc": "BAND",
    "lower_wick_ratio": "CANDLE",
    "upper_wick_ratio": "CANDLE",
    "body_ratio": "CANDLE",
    "intraday_return": "INTRADAY",
    "rebound_from_7d_low": "REBOUND",
    "rebound_vs_prior_drop": "REBOUND",
    "price_power_value": "POWER",
    "body_value_power": "POWER",
    "room_to_20d_high": "HIGH_ROOM",
    "room_to_60d_high": "HIGH_ROOM",
}

CORE_FEATURES = [
    "room_to_20d_high",
    "room_to_60d_high",
    "upper_wick_ratio",
    "today_pct",
    "vol5",
    "max_drop_7d",
    "rebound_vs_prior_drop",
    "market_today_pct",  # 있으면 사용. FEATURE_GROUPS에는 없지만 기존 v3 핵심이라 허용.
    "body_ratio",
    "intraday_return",
]

BALANCED_FEATURES = [
    "room_to_20d_high",
    "room_to_60d_high",
    "upper_wick_ratio",
    "today_pct",
    "vol5",
    "max_drop_7d",
    "rebound_vs_prior_drop",
    "market_today_pct",
    "body_ratio",
    "intraday_return",
    "ma5_chg_rate",
    "today_tr_val_eok",
    "gap_pct",
    "BB_perc",
    "pct_vs_lastweek",
]

ALL_FEATURES = [
    "vol5",
    "vol_ratio_5_15",
    "today_pct",
    "max_drop_7d",
    "gap_pct",
    "pct_vs_lastweek",
    "dist_to_ma5",
    "ma5_chg_rate",
    "today_tr_val_eok",
    "BB_perc",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "intraday_return",
    "rebound_from_7d_low",
    "rebound_vs_prior_drop",
    "price_power_value",
    "body_value_power",
    "room_to_20d_high",
    "room_to_60d_high",
    "market_today_pct",
]

ALLOWED_OPS = {
    "room_to_20d_high": [">=", "<="],
    "room_to_60d_high": [">=", "<="],
    "upper_wick_ratio": ["<="],
    "today_pct": [">="],
    "vol5": [">="],
    "vol_ratio_5_15": [">=", "<="],
    "today_tr_val_eok": [">=", "<="],
    "body_ratio": [">="],
    "lower_wick_ratio": [">=", "<="],
    "rebound_from_7d_low": [">="],
    "intraday_return": [">="],
    "ma5_chg_rate": [">=", "<="],
    "price_power_value": [">="],
    "body_value_power": [">="],
    "rebound_vs_prior_drop": [">=", "<="],
    "max_drop_7d": ["<="],
    "market_today_pct": [">="],
    "gap_pct": [">=", "<="],
    "pct_vs_lastweek": [">=", "<="],
    "dist_to_ma5": [">=", "<="],
    "BB_perc": [">=", "<="],
}

INTERVAL_FEATURES = {
    "room_to_20d_high",
    "room_to_60d_high",
    "rebound_vs_prior_drop",
    "gap_pct",
    "pct_vs_lastweek",
    "dist_to_ma5",
    "BB_perc",
    "vol_ratio_5_15",
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class Atom:
    feature: str
    op: str
    threshold: float

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
        return " AND ".join([a.name() for a in self.atoms])

    def features(self) -> List[str]:
        return sorted(set(a.feature for a in self.atoms))


# =============================================================================
# Basic utilities
# =============================================================================

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "today", "date", "Date", "datetime", "trade_date",
        "trd_date", "일자", "날짜", "기준일", "ymd", "YMD",
    ]:
        if c in df.columns:
            return c
    return None


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


def split_train_selection_test_by_date(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float,
    selection_valid_ratio: float,
    final_test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    total_ratio = train_ratio + selection_valid_ratio + final_test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + selection_valid_ratio + final_test_ratio must be 1.0, got {total_ratio}"
        )

    work = df.sort_values(date_col).reset_index(drop=True).copy()
    dates = list(pd.Series(work[date_col].dropna().unique()).sort_values())

    if len(dates) < 3:
        n = len(work)
        i1 = int(n * train_ratio)
        i2 = int(n * (train_ratio + selection_valid_ratio))
        train = work.iloc[:i1].copy().reset_index(drop=True)
        selection = work.iloc[i1:i2].copy().reset_index(drop=True)
        final = work.iloc[i2:].copy().reset_index(drop=True)
        info = {
            "split_mode": "row_fallback",
            "train_end_date": None,
            "selection_start_date": None,
            "selection_end_date": None,
            "final_start_date": None,
        }
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
        "train_start_date": work[date_col].min(),
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


def calc_metrics(y: np.ndarray, mask: np.ndarray) -> Dict:
    mask = np.asarray(mask).astype(bool)
    y = np.asarray(y).astype(int)

    selected_count = int(mask.sum())
    total_count = int(len(y))
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


def apply_atom(df: pd.DataFrame, atom: Atom) -> np.ndarray:
    s = pd.to_numeric(df[atom.feature], errors="coerce")

    if atom.op == ">=":
        return (s >= atom.threshold).fillna(False).to_numpy()

    if atom.op == "<=":
        return (s <= atom.threshold).fillna(False).to_numpy()

    raise ValueError(f"unknown op: {atom.op}")


def apply_rule(df: pd.DataFrame, atoms: Tuple[Atom, ...]) -> np.ndarray:
    if len(atoms) == 0:
        return np.ones(len(df), dtype=bool)

    mask = np.ones(len(df), dtype=bool)
    for a in atoms:
        mask &= apply_atom(df, a)
    return mask


def mask_hash(mask: np.ndarray) -> str:
    m = np.asarray(mask).astype(bool)
    packed = np.packbits(m)
    return hashlib.md5(packed.tobytes()).hexdigest()


def canonical_atoms_key(atoms: Tuple[Atom, ...]) -> Tuple[Tuple[str, str, float], ...]:
    return tuple(sorted((a.feature, a.op, round(float(a.threshold), 10)) for a in atoms))


def canonicalize_atoms(atoms: Tuple[Atom, ...]) -> Tuple[Atom, ...]:
    return tuple(sorted(atoms, key=lambda a: (a.feature, a.op, a.threshold)))


# =============================================================================
# Feature handling
# =============================================================================

def choose_features(feature_set: str) -> List[str]:
    if feature_set == "core":
        return CORE_FEATURES
    if feature_set == "balanced":
        return BALANCED_FEATURES
    if feature_set == "all":
        return ALL_FEATURES
    raise ValueError(f"unknown feature_set: {feature_set}")


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
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "spearman_corr": c,
                    "abs_corr": abs(c),
                })
                pairs.add(frozenset([a, b]))

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("abs_corr", ascending=False)

    return out, pairs


def has_correlated_pair(used_features: Set[str], new_feature: str, corr_pairs: Set[frozenset]) -> bool:
    for f in used_features:
        if frozenset([f, new_feature]) in corr_pairs:
            return True
    return False


# =============================================================================
# Threshold candidates
# =============================================================================

def default_extra_thresholds() -> Dict[str, List[Tuple[str, float]]]:
    return {
        "room_to_20d_high": [
            (">=", -10.0), (">=", -5.0), (">=", 0.0), (">=", 3.0), (">=", 5.0),
            ("<=", -5.0), ("<=", 0.0), ("<=", 3.0), ("<=", 4.4438), ("<=", 5.0),
            ("<=", 10.0), ("<=", 20.0), ("<=", 30.0),
        ],
        "room_to_60d_high": [
            (">=", -10.0), (">=", -5.0), (">=", 0.0), (">=", 3.0), (">=", 5.0),
            ("<=", -5.0), ("<=", 0.0), ("<=", 5.0), ("<=", 10.0),
            ("<=", 12.2626), ("<=", 18.9896), ("<=", 20.0), ("<=", 30.0),
            ("<=", 50.0),
        ],
        "upper_wick_ratio": [
            ("<=", 0.0), ("<=", 0.02), ("<=", 0.04), ("<=", 0.064),
            ("<=", 0.087), ("<=", 0.1), ("<=", 0.111), ("<=", 0.158),
            ("<=", 0.21), ("<=", 0.31),
        ],
        "lower_wick_ratio": [
            ("<=", 0.0), ("<=", 0.02), ("<=", 0.05), (">=", 0.05),
            (">=", 0.10), (">=", 0.20),
        ],
        "today_pct": [
            (">=", 5.0), (">=", 8.0), (">=", 10.0), (">=", 11.253),
            (">=", 13.0), (">=", 15.0), (">=", 15.86), (">=", 18.0),
            (">=", 20.0), (">=", 22.2708), (">=", 25.0),
        ],
        "vol5": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0), (">=", 6.507),
            (">=", 8.0), (">=", 8.467), (">=", 10.0), (">=", 10.5478),
            (">=", 10.5491), (">=", 13.4364),
        ],
        "vol_ratio_5_15": [
            ("<=", 0.5), ("<=", 0.7), ("<=", 0.9), ("<=", 1.0),
            (">=", 1.1), (">=", 1.3), (">=", 1.5), (">=", 1.64),
            (">=", 1.8), (">=", 2.0),
        ],
        "today_tr_val_eok": [
            (">=", 3.0), (">=", 5.0), (">=", 8.0), (">=", 10.0),
            (">=", 12.4428), (">=", 15.3325), (">=", 20.0),
            (">=", 23.4034), (">=", 50.0), (">=", 100.0),
            ("<=", 300.0), ("<=", 500.0), ("<=", 841.482), ("<=", 1391.67),
        ],
        "body_ratio": [
            (">=", 0.6), (">=", 0.7), (">=", 0.706), (">=", 0.8),
            (">=", 0.839), (">=", 0.9),
        ],
        "rebound_from_7d_low": [
            (">=", 25.0), (">=", 30.0), (">=", 32.9122), (">=", 35.0),
            (">=", 40.0), (">=", 40.7563), (">=", 45.0), (">=", 50.0),
            (">=", 55.0), (">=", 60.0),
        ],
        "intraday_return": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0), (">=", 8.0),
            (">=", 10.0), (">=", 15.0), (">=", 18.6578),
        ],
        "ma5_chg_rate": [
            (">=", 3.0), (">=", 5.0), (">=", 5.8628), (">=", 8.0),
            (">=", 10.0), ("<=", -3.0), ("<=", -5.0),
        ],
        "price_power_value": [
            (">=", 10.0), (">=", 20.0), (">=", 40.0), (">=", 60.0),
            (">=", 80.0), (">=", 100.0),
        ],
        "body_value_power": [
            (">=", 5.0), (">=", 10.0), (">=", 15.0), (">=", 20.0),
            (">=", 30.0),
        ],
        "rebound_vs_prior_drop": [
            (">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 3.315),
            (">=", 5.0), ("<=", 10.0), ("<=", 20.0),
        ],
        "max_drop_7d": [
            ("<=", -2.5962), ("<=", -3.0), ("<=", -3.541), ("<=", -3.7934),
            ("<=", -5.0), ("<=", -7.0), ("<=", -10.0),
        ],
        "market_today_pct": [
            (">=", -3.0), (">=", -2.0), (">=", -1.0), (">=", -0.51),
            (">=", 0.0), (">=", 0.5), (">=", 1.0),
        ],
        "gap_pct": [
            ("<=", -2.0), ("<=", -1.0), ("<=", 0.0), (">=", 0.0),
            (">=", 1.0), (">=", 2.0),
        ],
        "pct_vs_lastweek": [
            ("<=", -10.0), ("<=", -5.0), ("<=", -3.0), ("<=", 0.0),
            (">=", 0.0), (">=", 3.0), (">=", 5.0),
        ],
        "dist_to_ma5": [
            ("<=", -2.0), ("<=", -1.0), ("<=", 0.0), (">=", 0.0),
            (">=", 3.0), (">=", 5.0), (">=", 8.0),
        ],
        "BB_perc": [
            ("<=", 0.1), ("<=", 0.2), ("<=", 0.3), (">=", 0.8),
            (">=", 1.0), (">=", 1.1),
        ],
    }


def make_atoms(
    train: pd.DataFrame,
    features: List[str],
    quantiles: List[float],
    extra_thresholds: Dict[str, List[Tuple[str, float]]],
    allowed_ops: Dict[str, List[str]],
) -> List[Atom]:
    atoms = []

    for f in features:
        if f not in train.columns:
            continue

        s = pd.to_numeric(train[f], errors="coerce")
        vals = s.replace([np.inf, -np.inf], np.nan).dropna()

        if vals.nunique() < 8:
            continue

        candidates = []

        for q in quantiles:
            v = vals.quantile(q)
            if np.isfinite(v):
                candidates.append(float(v))

        for _, th in extra_thresholds.get(f, []):
            candidates.append(float(th))

        candidates = sorted(set(round(x, 10) for x in candidates if np.isfinite(x)))

        ops = allowed_ops.get(f, [">=", "<="])
        extra_by_op = set(extra_thresholds.get(f, []))

        for th in candidates:
            for op in ops:
                if (op, th) in extra_by_op or len(extra_by_op) == 0 or True:
                    atoms.append(Atom(feature=f, op=op, threshold=float(th)))

    # dedup
    seen = set()
    out = []
    for a in atoms:
        key = (a.feature, a.op, round(a.threshold, 10))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)

    return out


# =============================================================================
# Rule constraints
# =============================================================================

def can_add_atom(base_atoms: Tuple[Atom, ...], atom: Atom, corr_pairs: Set[frozenset], args) -> bool:
    used_features = set(a.feature for a in base_atoms)

    same_feature_atoms = [a for a in base_atoms if a.feature == atom.feature]

    if atom.feature in used_features and atom.feature not in INTERVAL_FEATURES:
        return False

    if atom.feature in INTERVAL_FEATURES:
        same_ops = [a.op for a in same_feature_atoms]
        if atom.op in same_ops:
            return False
        if len(same_feature_atoms) >= 2:
            return False

        if len(same_feature_atoms) == 1:
            old = same_feature_atoms[0]
            if old.op == ">=" and atom.op == "<=":
                if old.threshold > atom.threshold:
                    return False
            if old.op == "<=" and atom.op == ">=":
                if atom.threshold > old.threshold:
                    return False

    else:
        if atom.feature in used_features:
            return False

    if args.use_corr_pruning and atom.feature not in used_features:
        if has_correlated_pair(used_features, atom.feature, corr_pairs):
            return False

    return True


# =============================================================================
# Monthly evaluation
# =============================================================================

def monthly_summary_from_mask(
    df: pd.DataFrame,
    target_col: str,
    mask: np.ndarray,
    min_month_count: int,
    pass_precision: float,
    pass_lift: float,
    crash_precision: float,
):
    df = df.reset_index(drop=True).copy()
    y_all = df[target_col].values
    mask_all = np.asarray(mask).astype(bool)

    rows = []
    for month, g in df.groupby("month", sort=True):
        idx = g.index.to_numpy()
        m = calc_metrics(y_all[idx], mask_all[idx])
        m["month"] = month
        rows.append(m)

    monthly_df = pd.DataFrame(rows)
    usable = monthly_df[monthly_df["selected_count"] >= min_month_count].copy()

    if len(usable) == 0:
        return monthly_df, {
            "n_usable_months": 0,
            "mean_month_precision": np.nan,
            "min_month_precision": np.nan,
            "std_month_precision": np.nan,
            "mean_month_count": np.nan,
            "total_month_count": int(monthly_df["selected_count"].sum()) if len(monthly_df) else 0,
            "pass_month_rate": 0.0,
            "bad_month_count": len(monthly_df),
            "crash_month_count": len(monthly_df),
        }

    pass_mask = (usable["precision"] >= pass_precision) & (usable["lift"] >= pass_lift)
    crash_mask = usable["precision"] < crash_precision

    return monthly_df, {
        "n_usable_months": len(usable),
        "mean_month_precision": usable["precision"].mean(),
        "min_month_precision": usable["precision"].min(),
        "std_month_precision": usable["precision"].std(ddof=0),
        "mean_month_count": usable["selected_count"].mean(),
        "total_month_count": int(monthly_df["selected_count"].sum()),
        "pass_month_rate": pass_mask.mean(),
        "bad_month_count": int((~pass_mask).sum()),
        "crash_month_count": int(crash_mask.sum()),
    }


def eval_rule(df: pd.DataFrame, target_col: str, atoms: Tuple[Atom, ...], args):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics(df[target_col].values, mask)
    monthly_df, mon = monthly_summary_from_mask(
        df=df,
        target_col=target_col,
        mask=mask,
        min_month_count=args.min_month_count,
        pass_precision=args.month_pass_precision,
        pass_lift=args.month_pass_lift,
        crash_precision=args.month_crash_precision,
    )
    return metrics, mon, monthly_df, mask


def eval_rule_fast(df: pd.DataFrame, target_col: str, atoms: Tuple[Atom, ...]):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics(df[target_col].values, mask)
    return metrics, mask


# =============================================================================
# Scoring
# =============================================================================

def beam_score_fast(metrics, args) -> float:
    count = metrics["selected_count"]
    precision = metrics["precision"]
    lift = metrics["lift"]

    if count < args.beam_min_count:
        return -1e18
    if not np.isfinite(precision) or not np.isfinite(lift):
        return -1e18
    if lift < args.beam_min_lift:
        return -1e18

    return precision * 55 + lift * 16 + math.log1p(count) * 2.0


def final_score(train_m, selection_m, train_mon, selection_mon, args):
    train_count = train_m["selected_count"]
    selection_count = selection_m["selected_count"]

    train_p = train_m["precision"]
    selection_p = selection_m["precision"]
    train_lift = train_m["lift"]
    selection_lift = selection_m["lift"]

    if train_count < args.min_train_count:
        return -1e18
    if selection_count < args.min_selection_count:
        return -1e18
    if not all(np.isfinite(x) for x in [train_p, selection_p, train_lift, selection_lift]):
        return -1e18
    if train_lift < args.min_train_lift:
        return -1e18
    if selection_lift < args.min_selection_lift:
        return -1e18
    if train_p < args.min_train_precision:
        return -1e18
    if selection_p < args.min_selection_precision:
        return -1e18

    gap = abs(train_p - selection_p)
    if gap > args.max_precision_gap:
        return -1e18

    if selection_mon["n_usable_months"] < args.min_selection_usable_months:
        return -1e18
    if selection_mon["pass_month_rate"] < args.min_selection_pass_month_rate:
        return -1e18
    if np.isfinite(selection_mon["min_month_precision"]):
        if selection_mon["min_month_precision"] < args.min_selection_month_min_precision:
            return -1e18
    if selection_mon.get("crash_month_count", 0) > args.max_selection_crash_months:
        return -1e18

    selection_lcb = selection_m["precision_lcb"]
    if np.isfinite(selection_lcb) and selection_lcb < args.min_selection_lcb:
        return -1e18

    selection_std = selection_mon.get("std_month_precision", 0.0)
    if not np.isfinite(selection_std):
        selection_std = 0.0

    selection_min_p = selection_mon["min_month_precision"]
    if not np.isfinite(selection_min_p):
        selection_min_p = 0.0

    precision_bonus_70 = max(0.0, selection_p - 0.70) * 250
    precision_bonus_65 = max(0.0, selection_p - 0.65) * 120
    count_bonus = math.log1p(selection_count) * args.selection_count_weight

    return (
        selection_p * 140
        + selection_lcb * 80
        + selection_lift * 25
        + selection_mon["mean_month_precision"] * 70
        + selection_min_p * 55
        + selection_mon["pass_month_rate"] * 70
        + train_p * 20
        + count_bonus
        + precision_bonus_70
        + precision_bonus_65
        - gap * 90
        - selection_std * 60
        - selection_mon["bad_month_count"] * 4
        - selection_mon.get("crash_month_count", 0) * 25
    )


# =============================================================================
# Search / dedup
# =============================================================================

def is_near_duplicate(new_rule: Rule, selected: List[Rule], args) -> bool:
    for r in selected:
        if new_rule.train_mask_key == r.train_mask_key:
            return True
        if new_rule.selection_mask_key == r.selection_mask_key:
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


def search_rules(train, selection, final_test, atoms, corr_pairs, args):
    beam = [tuple()]
    selected_candidates = []
    seen_rules = set()
    seen_train_masks = set()

    for depth in range(1, args.max_depth + 1):
        print(f"[INFO] searching depth={depth}")
        beam_candidates = []

        for base_atoms in beam:
            for atom in atoms:
                if not can_add_atom(base_atoms, atom, corr_pairs, args):
                    continue

                new_atoms = canonicalize_atoms(tuple(list(base_atoms) + [atom]))
                key = canonical_atoms_key(new_atoms)

                if key in seen_rules:
                    continue

                seen_rules.add(key)

                # 1차 fast train 평가
                train_fast_m, train_mask = eval_rule_fast(train, args.target, new_atoms)
                bscore_fast = beam_score_fast(train_fast_m, args)
                if bscore_fast <= -1e17:
                    continue

                train_key = mask_hash(train_mask)
                if train_key in seen_train_masks and len(new_atoms) > 1:
                    continue
                seen_train_masks.add(train_key)

                # 2차 fast selection 평가
                selection_fast_m, selection_mask = eval_rule_fast(selection, args.target, new_atoms)
                if selection_fast_m["selected_count"] < args.fast_min_selection_count:
                    continue
                if not np.isfinite(selection_fast_m["precision"]):
                    continue
                if selection_fast_m["precision"] < args.fast_min_selection_precision:
                    continue
                if not np.isfinite(selection_fast_m["lift"]) or selection_fast_m["lift"] < args.fast_min_selection_lift:
                    continue

                # 통과 후보만 monthly 평가
                train_m, train_mon, _, train_mask = eval_rule(train, args.target, new_atoms, args)
                selection_m, selection_mon, _, selection_mask = eval_rule(selection, args.target, new_atoms, args)
                final_m, final_mon, _, final_mask = eval_rule(final_test, args.target, new_atoms, args)

                fscore = final_score(train_m, selection_m, train_mon, selection_mon, args)
                score = fscore if fscore > -1e17 else bscore_fast

                rule = Rule(
                    atoms=new_atoms,
                    train_metrics=train_m,
                    selection_metrics=selection_m,
                    final_metrics=final_m,
                    train_monthly=train_mon,
                    selection_monthly=selection_mon,
                    final_monthly=final_mon,
                    score=score,
                    train_mask_key=train_key,
                    selection_mask_key=mask_hash(selection_mask),
                )

                beam_candidates.append(rule)
                if fscore > -1e17:
                    selected_candidates.append(rule)

        if not beam_candidates:
            print(f"[WARN] no beam candidates at depth={depth}")
            break

        beam_candidates = sorted(beam_candidates, key=lambda r: r.score, reverse=True)
        beam = [r.atoms for r in beam_candidates[:args.beam_width]]

        print(
            f"[INFO] depth={depth}, beam_candidates={len(beam_candidates)}, "
            f"selected_candidates={len(selected_candidates)}, kept_beam={len(beam)}"
        )

    selected_candidates = sorted(selected_candidates, key=lambda r: r.score, reverse=True)
    deduped = []

    for r in selected_candidates:
        if is_near_duplicate(r, deduped, args):
            continue
        deduped.append(r)
        if len(deduped) >= args.top_k:
            break

    return deduped


# =============================================================================
# Simplify
# =============================================================================

def simplify_rule_by_dropping_atoms(rule: Rule, train, selection, final_test, args) -> Rule:
    current_rule = rule
    improved = True

    while improved and len(current_rule.atoms) > 1:
        improved = False
        best_candidate = None
        best_tuple = None

        for i in range(len(current_rule.atoms)):
            new_atoms = tuple(a for j, a in enumerate(current_rule.atoms) if j != i)
            new_atoms = canonicalize_atoms(new_atoms)

            train_m, train_mon, _, train_mask = eval_rule(train, args.target, new_atoms, args)
            selection_m, selection_mon, _, selection_mask = eval_rule(selection, args.target, new_atoms, args)
            final_m, final_mon, _, final_mask = eval_rule(final_test, args.target, new_atoms, args)

            fs = final_score(train_m, selection_m, train_mon, selection_mon, args)
            if fs <= -1e17:
                continue

            old_sel_p = current_rule.selection_metrics["precision"]
            new_sel_p = selection_m["precision"]
            old_sel_count = current_rule.selection_metrics["selected_count"]
            new_sel_count = selection_m["selected_count"]

            if new_sel_p + args.simplify_precision_drop < old_sel_p:
                continue

            if new_sel_count < old_sel_count * args.simplify_min_count_ratio:
                continue

            tup = (
                len(new_atoms),
                fs,
                new_sel_p,
                new_sel_count,
            )

            if best_tuple is None or tup > best_tuple:
                best_tuple = tup
                best_candidate = Rule(
                    atoms=new_atoms,
                    train_metrics=train_m,
                    selection_metrics=selection_m,
                    final_metrics=final_m,
                    train_monthly=train_mon,
                    selection_monthly=selection_mon,
                    final_monthly=final_mon,
                    score=fs,
                    train_mask_key=mask_hash(train_mask),
                    selection_mask_key=mask_hash(selection_mask),
                )

        if best_candidate is not None:
            current_rule = best_candidate
            improved = True

    return current_rule


# =============================================================================
# DataFrames
# =============================================================================

def flatten_metrics(prefix: str, metrics: Dict) -> Dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def flatten_monthly(prefix: str, mon: Dict) -> Dict:
    return {f"{prefix}_month_{k}": v for k, v in mon.items()}


def rules_to_df(rules: List[Rule]) -> pd.DataFrame:
    rows = []

    for rank, r in enumerate(rules, start=1):
        row = {
            "rank": rank,
            "rule": r.name(),
            "n_atoms": len(r.atoms),
            "features": ",".join(r.features()),
            "score": r.score,
        }
        row.update(flatten_metrics("train", r.train_metrics))
        row.update(flatten_metrics("selection", r.selection_metrics))
        row.update(flatten_metrics("final", r.final_metrics))
        row.update(flatten_monthly("train", r.train_monthly))
        row.update(flatten_monthly("selection", r.selection_monthly))
        row.update(flatten_monthly("final", r.final_monthly))

        row["precision_gap_train_selection_abs"] = abs(row["train_precision"] - row["selection_precision"])
        row["precision_gap_selection_final_abs"] = abs(row["selection_precision"] - row["final_precision"])

        row["pass_selection_60_n60"] = (
            row["selection_precision"] >= 0.60
            and row["selection_lift"] >= 1.45
            and row["selection_selected_count"] >= 60
        )
        row["pass_selection_65_n50"] = (
            row["selection_precision"] >= 0.65
            and row["selection_lift"] >= 1.55
            and row["selection_selected_count"] >= 50
        )
        row["pass_selection_70_n30"] = (
            row["selection_precision"] >= 0.70
            and row["selection_lift"] >= 1.65
            and row["selection_selected_count"] >= 30
        )
        row["pass_selection_70_n60"] = (
            row["selection_precision"] >= 0.70
            and row["selection_lift"] >= 1.65
            and row["selection_selected_count"] >= 60
        )
        row["pass_train_selection_70_n30"] = (
            row["train_precision"] >= 0.70
            and row["selection_precision"] >= 0.70
            and row["selection_selected_count"] >= 30
        )
        row["pass_final_60_n30"] = (
            row["final_precision"] >= 0.60
            and row["final_lift"] >= 1.25
            and row["final_selected_count"] >= 30
        )
        row["pass_final_65_n30"] = (
            row["final_precision"] >= 0.65
            and row["final_lift"] >= 1.35
            and row["final_selected_count"] >= 30
        )
        row["pass_final_70_n20"] = (
            row["final_precision"] >= 0.70
            and row["final_lift"] >= 1.45
            and row["final_selected_count"] >= 20
        )

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

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Fixed-rule forward robustness
# =============================================================================

def build_forward_splits(df: pd.DataFrame, date_col: str, n_splits: int, min_train_months: int):
    months = sorted(df["month"].unique())
    splits = []

    if len(months) <= min_train_months + 1:
        return splits

    valid_months = months[min_train_months:]

    if len(valid_months) <= n_splits:
        chosen = valid_months
    else:
        idxs = np.linspace(0, len(valid_months) - 1, n_splits).round().astype(int)
        chosen = [valid_months[i] for i in idxs]

    for m in chosen:
        train_months = [x for x in months if x < m]
        splits.append({"valid_month": m, "train_months": train_months})

    return splits


def fixed_rule_forward_eval(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    rules: List[Rule],
    n_splits: int,
    min_train_months: int,
) -> pd.DataFrame:
    splits = build_forward_splits(df, date_col, n_splits, min_train_months)
    rows = []

    for rank, r in enumerate(rules, start=1):
        for s in splits:
            valid_month = s["valid_month"]
            test_df = df[df["month"] == valid_month].copy().reset_index(drop=True)

            if len(test_df) == 0:
                continue

            mask = apply_rule(test_df, r.atoms)
            m = calc_metrics(test_df[target_col].values, mask)

            row = {
                "rank": rank,
                "rule": r.name(),
                "valid_month": valid_month,
                "n_train_months_before": len(s["train_months"]),
            }
            row.update(m)
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_forward(
    forward_df: pd.DataFrame,
    min_count: int,
    min_precision: float,
    min_lift: float,
    crash_precision: float,
) -> pd.DataFrame:
    if len(forward_df) == 0:
        return pd.DataFrame()

    rows = []

    for (rank, rule), g in forward_df.groupby(["rank", "rule"], sort=False):
        usable = g[g["selected_count"] >= min_count].copy()

        if len(usable) == 0:
            rows.append({
                "rank": rank,
                "rule": rule,
                "fw_n_splits": len(g),
                "fw_n_usable_splits": 0,
                "fw_mean_precision": np.nan,
                "fw_min_precision": np.nan,
                "fw_total_count": 0,
                "fw_pass_split_rate": 0.0,
                "fw_crash_split_count": len(g),
            })
            continue

        pass_mask = (usable["precision"] >= min_precision) & (usable["lift"] >= min_lift)
        crash_mask = usable["precision"] < crash_precision

        rows.append({
            "rank": rank,
            "rule": rule,
            "fw_n_splits": len(g),
            "fw_n_usable_splits": len(usable),
            "fw_mean_precision": usable["precision"].mean(),
            "fw_min_precision": usable["precision"].min(),
            "fw_total_count": int(usable["selected_count"].sum()),
            "fw_pass_split_rate": pass_mask.mean(),
            "fw_crash_split_count": int(crash_mask.sum()),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Final filters
# =============================================================================

def build_final_test_filtered_rules(rules_df: pd.DataFrame, forward_summary: pd.DataFrame, args) -> pd.DataFrame:
    if len(rules_df) == 0:
        return pd.DataFrame()

    if len(forward_summary):
        merged = rules_df.merge(forward_summary, on=["rank", "rule"], how="left")
    else:
        merged = rules_df.copy()
        merged["fw_mean_precision"] = np.nan
        merged["fw_min_precision"] = np.nan
        merged["fw_total_count"] = 0
        merged["fw_pass_split_rate"] = 0.0
        merged["fw_crash_split_count"] = 999

    out = merged[
        (merged["train_precision"] >= args.final_min_train_precision)
        & (merged["selection_precision"] >= args.final_min_selection_precision)
        & (merged["selection_selected_count"] >= args.final_min_selection_count)
        & (merged["selection_lift"] >= args.final_min_selection_lift)
        & (merged["selection_month_min_month_precision"] >= args.final_min_selection_month_min_precision)
        & (merged["selection_month_crash_month_count"] <= args.final_max_selection_crash_months)
        & (merged["final_precision"] >= args.final_min_test_precision)
        & (merged["final_selected_count"] >= args.final_min_test_count)
        & (merged["final_lift"] >= args.final_min_test_lift)
        & (merged["final_month_crash_month_count"] <= args.final_max_test_crash_months)
        & (merged["fw_mean_precision"] >= args.final_min_fw_mean_precision)
        & (merged["fw_min_precision"] >= args.final_min_fw_min_precision)
        & (merged["fw_pass_split_rate"] >= args.final_min_fw_pass_split_rate)
        & (merged["fw_total_count"] >= args.final_min_fw_total_count)
        & (merged["fw_crash_split_count"] <= args.final_max_fw_crash_splits)
    ].copy()

    if len(out):
        out = out.sort_values(
            ["final_precision", "final_selected_count", "selection_precision", "selection_selected_count"],
            ascending=[False, False, False, False],
        )

    return out


# =============================================================================
# OR rule sets
# =============================================================================

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


def eval_or_rule_set(train, selection, final_test, df_all, date_col, rules: List[Rule], target_col: str, args):
    train_mask = or_mask(train, rules)
    selection_mask = or_mask(selection, rules)
    final_mask = or_mask(final_test, rules)

    train_m = calc_metrics(train[target_col].values, train_mask)
    selection_m = calc_metrics(selection[target_col].values, selection_mask)
    final_m = calc_metrics(final_test[target_col].values, final_mask)

    _, selection_mon = monthly_summary_from_mask(
        df=selection,
        target_col=target_col,
        mask=selection_mask,
        min_month_count=args.min_month_count,
        pass_precision=args.month_pass_precision,
        pass_lift=args.month_pass_lift,
        crash_precision=args.month_crash_precision,
    )

    # fixed forward on all data, but for reporting only.
    rows = []
    splits = build_forward_splits(df_all, date_col, args.fw_splits, args.fw_min_train_months)
    for s in splits:
        mth = s["valid_month"]
        g = df_all[df_all["month"] == mth].copy().reset_index(drop=True)
        if len(g) == 0:
            continue
        m = calc_metrics(g[target_col].values, or_mask(g, rules))
        m["valid_month"] = mth
        rows.append(m)

    fw_df = pd.DataFrame(rows)
    if len(fw_df) == 0:
        fw_s = {
            "fw_n_splits": 0,
            "fw_n_usable_splits": 0,
            "fw_mean_precision": np.nan,
            "fw_min_precision": np.nan,
            "fw_total_count": 0,
            "fw_pass_split_rate": 0.0,
            "fw_crash_split_count": 999,
        }
    else:
        usable = fw_df[fw_df["selected_count"] >= args.fw_min_count].copy()
        if len(usable) == 0:
            fw_s = {
                "fw_n_splits": len(fw_df),
                "fw_n_usable_splits": 0,
                "fw_mean_precision": np.nan,
                "fw_min_precision": np.nan,
                "fw_total_count": 0,
                "fw_pass_split_rate": 0.0,
                "fw_crash_split_count": len(fw_df),
            }
        else:
            pass_mask = (usable["precision"] >= args.fw_min_precision) & (usable["lift"] >= args.fw_min_lift)
            crash_mask = usable["precision"] < args.fw_crash_precision
            fw_s = {
                "fw_n_splits": len(fw_df),
                "fw_n_usable_splits": len(usable),
                "fw_mean_precision": usable["precision"].mean(),
                "fw_min_precision": usable["precision"].min(),
                "fw_total_count": int(usable["selected_count"].sum()),
                "fw_pass_split_rate": pass_mask.mean(),
                "fw_crash_split_count": int(crash_mask.sum()),
            }

    return train_m, selection_m, final_m, selection_mon, fw_s


def evaluate_or_rule_sets(rules, train, selection, final_test, df_all, date_col, args):
    if len(rules) == 0:
        return pd.DataFrame()

    candidate_rules = rules[:args.or_top_rules]
    rows = []

    for k in range(2, args.or_max_size + 1):
        for combo in itertools.combinations(candidate_rules, k):
            if not has_enough_incremental_coverage(selection, combo, args.or_min_incremental_selection_count):
                continue

            train_m, selection_m, final_m, selection_mon, fw_s = eval_or_rule_set(
                train=train,
                selection=selection,
                final_test=final_test,
                df_all=df_all,
                date_col=date_col,
                rules=list(combo),
                target_col=args.target,
                args=args,
            )

            selection_p = selection_m["precision"]
            selection_count = selection_m["selected_count"]
            final_p = final_m["precision"]
            final_count = final_m["selected_count"]

            if not np.isfinite(selection_p) or not np.isfinite(final_p):
                continue

            # OR selection uses selection_valid and only filters final_test for reporting-grade safety.
            if selection_p < args.or_min_selection_precision:
                continue
            if selection_count < args.or_min_selection_count:
                continue
            if selection_m["lift"] < args.or_min_selection_lift:
                continue
            if final_count < args.or_min_test_count:
                continue
            if final_p < args.or_min_test_precision:
                continue
            if final_m["lift"] < args.or_min_test_lift:
                continue

            fw_mean = fw_s.get("fw_mean_precision", np.nan)
            fw_min = fw_s.get("fw_min_precision", np.nan)
            fw_total = fw_s.get("fw_total_count", 0)
            fw_pass = fw_s.get("fw_pass_split_rate", 0.0)
            fw_crash = fw_s.get("fw_crash_split_count", 999)

            if np.isfinite(fw_mean) and fw_mean < args.or_min_fw_mean_precision:
                continue
            if np.isfinite(fw_min) and fw_min < args.or_min_fw_min_precision:
                continue
            if fw_pass < args.or_min_fw_pass_split_rate:
                continue
            if fw_crash > args.or_max_fw_crash_splits:
                continue

            score = (
                selection_p * 120
                + final_p * 160
                + math.log1p(selection_count) * args.or_count_weight
                + math.log1p(final_count) * args.or_test_count_weight
                + (fw_mean if np.isfinite(fw_mean) else 0) * 90
                + (fw_min if np.isfinite(fw_min) else 0) * 50
                + math.log1p(fw_total) * 5
                - fw_crash * 30
            )

            rows.append({
                "or_size": k,
                "selection_count": selection_count,
                "selection_precision": selection_p,
                "selection_precision_lcb": selection_m["precision_lcb"],
                "selection_lift": selection_m["lift"],
                "selection_coverage": selection_m["coverage"],
                "final_count": final_count,
                "final_precision": final_p,
                "final_precision_lcb": final_m["precision_lcb"],
                "final_lift": final_m["lift"],
                "final_coverage": final_m["coverage"],
                "fw_mean_precision": fw_mean,
                "fw_min_precision": fw_min,
                "fw_total_count": fw_total,
                "fw_pass_split_rate": fw_pass,
                "fw_crash_split_count": fw_crash,
                "or_score": score,
                "rules": " || ".join([r.name() for r in combo]),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["or_score", "final_precision", "final_count", "selection_precision"],
            ascending=[False, False, False, False],
        )
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="stable_rule_miner_final_v4_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)

    parser.add_argument("--feature-set", default="balanced", choices=["core", "balanced", "all"])

    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--selection-valid-ratio", type=float, default=0.20)
    parser.add_argument("--final-test-ratio", type=float, default=0.20)

    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=1500)
    parser.add_argument("--top-k", type=int, default=150)

    parser.add_argument("--min-train-count", type=int, default=60)
    parser.add_argument("--min-selection-count", type=int, default=35)
    parser.add_argument("--min-train-precision", type=float, default=0.55)
    parser.add_argument("--min-selection-precision", type=float, default=0.62)
    parser.add_argument("--min-train-lift", type=float, default=1.20)
    parser.add_argument("--min-selection-lift", type=float, default=1.45)
    parser.add_argument("--min-selection-lcb", type=float, default=0.48)
    parser.add_argument("--max-precision-gap", type=float, default=0.22)

    parser.add_argument("--beam-min-count", type=int, default=50)
    parser.add_argument("--beam-min-lift", type=float, default=1.00)

    parser.add_argument("--fast-min-selection-count", type=int, default=20)
    parser.add_argument("--fast-min-selection-precision", type=float, default=0.52)
    parser.add_argument("--fast-min-selection-lift", type=float, default=1.15)

    parser.add_argument("--min-month-count", type=int, default=5)
    parser.add_argument("--month-pass-precision", type=float, default=0.55)
    parser.add_argument("--month-pass-lift", type=float, default=1.15)
    parser.add_argument("--month-crash-precision", type=float, default=0.45)

    parser.add_argument("--min-selection-usable-months", type=int, default=2)
    parser.add_argument("--min-selection-pass-month-rate", type=float, default=0.20)
    parser.add_argument("--min-selection-month-min-precision", type=float, default=0.52)
    parser.add_argument("--max-selection-crash-months", type=int, default=1)

    parser.add_argument("--selection-count-weight", type=float, default=7.0)

    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--use-corr-pruning", action="store_true")

    parser.add_argument("--dup-count-tol", type=float, default=0.10)
    parser.add_argument("--dup-precision-gain", type=float, default=0.01)

    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--simplify-precision-drop", type=float, default=0.015)
    parser.add_argument("--simplify-min-count-ratio", type=float, default=0.85)

    # Fixed-rule forward robustness
    parser.add_argument("--fw-top-rules", type=int, default=150)
    parser.add_argument("--fw-splits", type=int, default=8)
    parser.add_argument("--fw-min-train-months", type=int, default=6)
    parser.add_argument("--fw-min-count", type=int, default=5)
    parser.add_argument("--fw-min-precision", type=float, default=0.58)
    parser.add_argument("--fw-min-lift", type=float, default=1.15)
    parser.add_argument("--fw-crash-precision", type=float, default=0.45)

    # Final test filters for single rules
    parser.add_argument("--final-min-train-precision", type=float, default=0.58)
    parser.add_argument("--final-min-selection-precision", type=float, default=0.66)
    parser.add_argument("--final-min-selection-count", type=int, default=35)
    parser.add_argument("--final-min-selection-lift", type=float, default=1.45)
    parser.add_argument("--final-min-selection-month-min-precision", type=float, default=0.52)
    parser.add_argument("--final-max-selection-crash-months", type=int, default=1)
    parser.add_argument("--final-min-test-precision", type=float, default=0.58)
    parser.add_argument("--final-min-test-count", type=int, default=20)
    parser.add_argument("--final-min-test-lift", type=float, default=1.15)
    parser.add_argument("--final-max-test-crash-months", type=int, default=2)
    parser.add_argument("--final-min-fw-mean-precision", type=float, default=0.62)
    parser.add_argument("--final-min-fw-min-precision", type=float, default=0.50)
    parser.add_argument("--final-min-fw-pass-split-rate", type=float, default=0.60)
    parser.add_argument("--final-min-fw-total-count", type=int, default=40)
    parser.add_argument("--final-max-fw-crash-splits", type=int, default=2)

    # OR rule sets
    parser.add_argument("--or-top-rules", type=int, default=18)
    parser.add_argument("--or-max-size", type=int, default=4)
    parser.add_argument("--or-min-incremental-selection-count", type=int, default=5)
    parser.add_argument("--or-min-selection-precision", type=float, default=0.66)
    parser.add_argument("--or-min-selection-count", type=int, default=60)
    parser.add_argument("--or-min-selection-lift", type=float, default=1.40)
    parser.add_argument("--or-min-test-precision", type=float, default=0.56)
    parser.add_argument("--or-min-test-count", type=int, default=25)
    parser.add_argument("--or-min-test-lift", type=float, default=1.10)
    parser.add_argument("--or-min-fw-mean-precision", type=float, default=0.62)
    parser.add_argument("--or-min-fw-min-precision", type=float, default=0.50)
    parser.add_argument("--or-min-fw-pass-split-rate", type=float, default=0.55)
    parser.add_argument("--or-max-fw-crash-splits", type=int, default=2)
    parser.add_argument("--or-count-weight", type=float, default=10.0)
    parser.add_argument("--or-test-count-weight", type=float, default=12.0)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    date_col = args.date_col or find_date_col(df)
    if date_col is None:
        raise ValueError("date column not found. Use --date-col today")

    df = prepare_df(df=df, target_col=args.target, date_col=date_col)

    requested_features = choose_features(args.feature_set)
    features = [f for f in requested_features if f in df.columns]
    missing = [f for f in requested_features if f not in df.columns]
    if missing:
        print("[WARN] missing features:", missing)

    train, selection, final_test, split_info = split_train_selection_test_by_date(
        df=df,
        date_col=date_col,
        train_ratio=args.train_ratio,
        selection_valid_ratio=args.selection_valid_ratio,
        final_test_ratio=args.final_test_ratio,
    )

    print("=" * 80)
    print("[INFO] rows:", len(df))
    print("[INFO] train rows:", len(train), "base_rate:", train[args.target].mean())
    print("[INFO] selection_valid rows:", len(selection), "base_rate:", selection[args.target].mean())
    print("[INFO] final_test rows:", len(final_test), "base_rate:", final_test[args.target].mean())
    print("[INFO] date_col:", date_col)
    print("[INFO] feature_set:", args.feature_set)
    print("[INFO] features:", features)
    print("[INFO] split_info:", split_info)
    print("[INFO] final_test is not used for rule selection; only final reporting/filtering")
    print("=" * 80)

    pd.DataFrame([split_info]).to_csv(os.path.join(args.out, "00_split_info.csv"), index=False, encoding="utf-8-sig")

    high_corr_df, corr_pairs = build_corr_pairs(train, features, args.corr_threshold)
    high_corr_df.to_csv(os.path.join(args.out, "00_high_corr_pairs.csv"), index=False, encoding="utf-8-sig")

    if len(high_corr_df):
        print("\n[HIGH CORR PAIRS]")
        print(high_corr_df.to_string(index=False))
    else:
        print("\n[HIGH CORR PAIRS] none")

    quantiles = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
        0.975,
    ]

    atoms = make_atoms(
        train=train,
        features=features,
        quantiles=quantiles,
        extra_thresholds=default_extra_thresholds(),
        allowed_ops=ALLOWED_OPS,
    )

    pd.DataFrame([
        {"feature": a.feature, "op": a.op, "threshold": a.threshold, "atom": a.name()}
        for a in atoms
    ]).to_csv(os.path.join(args.out, "01_atoms.csv"), index=False, encoding="utf-8-sig")

    print("[INFO] atoms:", len(atoms))

    rules = search_rules(
        train=train,
        selection=selection,
        final_test=final_test,
        atoms=atoms,
        corr_pairs=corr_pairs,
        args=args,
    )

    if args.simplify:
        simplified = []
        for r in rules:
            sr = simplify_rule_by_dropping_atoms(r, train, selection, final_test, args)
            simplified.append(sr)

        simplified = sorted(simplified, key=lambda r: r.score, reverse=True)
        deduped = []
        for r in simplified:
            if is_near_duplicate(r, deduped, args):
                continue
            deduped.append(r)
            if len(deduped) >= args.top_k:
                break
        rules = deduped

    rules_df = rules_to_df(rules)
    rules_df.to_csv(os.path.join(args.out, "02_selected_rules.csv"), index=False, encoding="utf-8-sig")

    if len(rules_df):
        rules_df[rules_df["pass_selection_60_n60"]].to_csv(os.path.join(args.out, "03_pass_selection_60_n60.csv"), index=False, encoding="utf-8-sig")
        rules_df[rules_df["pass_selection_65_n50"]].to_csv(os.path.join(args.out, "04_pass_selection_65_n50.csv"), index=False, encoding="utf-8-sig")
        rules_df[rules_df["pass_selection_70_n30"]].to_csv(os.path.join(args.out, "05_pass_selection_70_n30.csv"), index=False, encoding="utf-8-sig")
        rules_df[rules_df["pass_selection_70_n60"]].to_csv(os.path.join(args.out, "06_pass_selection_70_n60.csv"), index=False, encoding="utf-8-sig")
        rules_df[rules_df["pass_final_60_n30"]].to_csv(os.path.join(args.out, "07_pass_final_60_n30.csv"), index=False, encoding="utf-8-sig")
        rules_df[rules_df["pass_final_65_n30"]].to_csv(os.path.join(args.out, "08_pass_final_65_n30.csv"), index=False, encoding="utf-8-sig")
        rules_df[rules_df["pass_final_70_n20"]].to_csv(os.path.join(args.out, "09_pass_final_70_n20.csv"), index=False, encoding="utf-8-sig")
    else:
        for fn in [
            "03_pass_selection_60_n60.csv", "04_pass_selection_65_n50.csv", "05_pass_selection_70_n30.csv",
            "06_pass_selection_70_n60.csv", "07_pass_final_60_n30.csv", "08_pass_final_65_n30.csv",
            "09_pass_final_70_n20.csv",
        ]:
            pd.DataFrame().to_csv(os.path.join(args.out, fn), index=False, encoding="utf-8-sig")

    monthly_df = collect_monthly_details(rules, train, selection, final_test, args)
    monthly_df.to_csv(os.path.join(args.out, "10_monthly_details.csv"), index=False, encoding="utf-8-sig")

    fw_df = fixed_rule_forward_eval(
        df=df,
        date_col=date_col,
        target_col=args.target,
        rules=rules[:args.fw_top_rules],
        n_splits=args.fw_splits,
        min_train_months=args.fw_min_train_months,
    )
    fw_df.to_csv(os.path.join(args.out, "11_fixed_rule_forward_eval.csv"), index=False, encoding="utf-8-sig")

    fw_summary = summarize_forward(
        forward_df=fw_df,
        min_count=args.fw_min_count,
        min_precision=args.fw_min_precision,
        min_lift=args.fw_min_lift,
        crash_precision=args.fw_crash_precision,
    )
    fw_summary.to_csv(os.path.join(args.out, "12_fixed_rule_forward_summary.csv"), index=False, encoding="utf-8-sig")

    final_test_filtered = build_final_test_filtered_rules(rules_df, fw_summary, args)
    final_test_filtered.to_csv(os.path.join(args.out, "13_final_test_filtered_rules.csv"), index=False, encoding="utf-8-sig")

    or_df = evaluate_or_rule_sets(
        rules=rules,
        train=train,
        selection=selection,
        final_test=final_test,
        df_all=df,
        date_col=date_col,
        args=args,
    )
    or_df.to_csv(os.path.join(args.out, "14_or_rule_sets_final_test.csv"), index=False, encoding="utf-8-sig")

    print("\n[SELECTED RULES]")
    show_cols = [
        "rank", "rule",
        "train_selected_count", "train_precision", "train_lift",
        "selection_selected_count", "selection_precision", "selection_precision_lcb", "selection_lift",
        "final_selected_count", "final_precision", "final_precision_lcb", "final_lift",
        "precision_gap_train_selection_abs", "precision_gap_selection_final_abs",
        "selection_month_min_month_precision", "selection_month_crash_month_count",
        "final_month_min_month_precision", "final_month_crash_month_count",
        "pass_selection_70_n30", "pass_final_60_n30", "pass_final_65_n30", "pass_final_70_n20",
    ]
    show_cols = [c for c in show_cols if c in rules_df.columns]

    if len(rules_df):
        print(rules_df[show_cols].head(50).to_string(index=False))
    else:
        print("No selected rules found.")

    print("\n[FINAL TEST FILTERED RULES]")
    if len(final_test_filtered):
        final_cols = [
            "rank", "rule", "selection_precision", "selection_selected_count",
            "final_precision", "final_selected_count", "final_lift",
            "fw_mean_precision", "fw_min_precision", "fw_total_count", "fw_pass_split_rate", "fw_crash_split_count",
        ]
        final_cols = [c for c in final_cols if c in final_test_filtered.columns]
        print(final_test_filtered[final_cols].head(30).to_string(index=False))
    else:
        print("No final-test-filtered rules passed.")

    print("\n[OR RULE SETS FINAL TEST]")
    if len(or_df):
        or_cols = [
            "or_size", "selection_count", "selection_precision", "selection_lift",
            "final_count", "final_precision", "final_lift",
            "fw_mean_precision", "fw_min_precision", "fw_total_count", "fw_pass_split_rate", "fw_crash_split_count",
            "or_score", "rules",
        ]
        or_cols = [c for c in or_cols if c in or_df.columns]
        print(or_df[or_cols].head(30).to_string(index=False))
    else:
        print("No OR rule sets passed final-test filters.")

    print("\n[SUMMARY]")
    print("selected rules:", len(rules_df))
    if len(rules_df):
        print("pass_selection_60_n60:", int(rules_df["pass_selection_60_n60"].sum()))
        print("pass_selection_65_n50:", int(rules_df["pass_selection_65_n50"].sum()))
        print("pass_selection_70_n30:", int(rules_df["pass_selection_70_n30"].sum()))
        print("pass_selection_70_n60:", int(rules_df["pass_selection_70_n60"].sum()))
        print("pass_final_60_n30:", int(rules_df["pass_final_60_n30"].sum()))
        print("pass_final_65_n30:", int(rules_df["pass_final_65_n30"].sum()))
        print("pass_final_70_n20:", int(rules_df["pass_final_70_n20"].sum()))
        print("best_selection_precision:", rules_df["selection_precision"].max())
        print("best_final_precision:", rules_df["final_precision"].max())
        print("max_selection_count:", rules_df["selection_selected_count"].max())
        print("max_final_count:", rules_df["final_selected_count"].max())
    print("final_test_filtered:", len(final_test_filtered))
    print("or_rule_sets_final_test:", len(or_df))
    print("=" * 80)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()
