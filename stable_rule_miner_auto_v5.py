#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stable_rule_miner_auto_v5.py

Purpose
-------
low_result_7_v2_desc.csv 기준 자동 분위수 룰 마이닝.

주요 특징
---------
1) 기본 target = target_before_stop_10
2) 수동 threshold 기본 OFF
3) 분위수 threshold 기반 atom 생성
4) 피쳐별 10분위 성과 리포트 생성
5) ALLOWED_OPS 방향성:
   - preset: 현재 CSV 분석 기준 추천 방향 사용
   - wide: 모든 피쳐 >=, <= 허용
   - auto: train 데이터에서 best >= / <= 비교 후 자동 추정
6) INTERVAL_FEATURES:
   - 같은 피쳐에 하한/상한을 동시에 허용할 피쳐 지정
7) score:
   - precision
   - Wilson LCB
   - lift
   - expected return
   - win/loss ratio
   - monthly stability
   - fixed forward stability
   를 함께 사용
8) 결과를 성격별로 분리:
   - high_precision
   - high_coverage
   - stable
   - aggressive
   - defensive

추천 실행
---------
python stable_rule_miner_auto_v5.py ^
  --csv csv/low_result_7_v2_desc.csv ^
  --out stable_rule_auto_v5_out ^
  --target target_before_stop_10 ^
  --threshold-mode quantile ^
  --ops-mode preset ^
  --max-depth 5 ^
  --beam-width 1200 ^
  --top-k 200 ^
  --simplify

더 넓게 탐색
------------
python stable_rule_miner_auto_v5.py ^
  --csv csv/low_result_7_v2_desc.csv ^
  --out stable_rule_auto_v5_wide ^
  --target target_before_stop_10 ^
  --threshold-mode quantile ^
  --ops-mode wide ^
  --max-depth 6 ^
  --beam-width 1800 ^
  --top-k 300 ^
  --simplify

자동 방향 추정
--------------
python stable_rule_miner_auto_v5.py ^
  --csv csv/low_result_7_v2_desc.csv ^
  --out stable_rule_auto_v5_auto_ops ^
  --target target_before_stop_10 ^
  --threshold-mode quantile ^
  --ops-mode auto ^
  --max-depth 5 ^
  --beam-width 1200 ^
  --top-k 200 ^
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


# =============================================================================
# Defaults for the uploaded CSV
# =============================================================================

DEFAULT_TARGET = "target_before_stop_10"

DEFAULT_FEATURES = [
    "vol5",
    "vol15",
    "rebound_from_7d_low",
    "today_pct",
    "price_power_value",
    "dist_to_ma5",
    "intraday_return",
    "tr_value_ratio_5d",
    "max_drop_7d",
    "body_value_power",
    "BB_perc",
    "gap_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "room_to_60d_high",
    "ma5_chg_rate",
    "pct_vs_lastweek",
    "dist_to_ma20",
    "ATR_pct",
]

# 이번 CSV를 10분위로 본 뒤의 1차 추천 방향.
# 처음에는 수동 threshold는 쓰지 않고, 방향성만 제한한다.
PRESET_ALLOWED_OPS = {
    # 클수록 좋은 경향
    "vol5": [">="],
    "vol15": [">="],
    "rebound_from_7d_low": [">="],
    "today_pct": [">="],
    "price_power_value": [">="],
    "intraday_return": [">="],
    "tr_value_ratio_5d": [">="],
    "body_value_power": [">="],
    "ATR_pct": [">="],

    # 작을수록 좋은 경향
    "max_drop_7d": ["<="],

    # 양방향 / 구간형 가능성
    "ma5_chg_rate": [">=", "<="],
    "dist_to_ma20": [">=", "<="],
    "dist_to_ma5": [">=", "<="],
    "pct_vs_lastweek": [">=", "<="],
    "BB_perc": [">=", "<="],
    "room_to_60d_high": [">=", "<="],
    "gap_pct": [">=", "<="],
    "lower_wick_ratio": [">=", "<="],

    # 직관은 <= 이지만, 이번 데이터에서는 애매하므로 1차 확인용 양방향
    "upper_wick_ratio": ["<="],
}

PRESET_INTERVAL_FEATURES = {
    "ma5_chg_rate",
    "dist_to_ma20",
    "dist_to_ma5",
    "pct_vs_lastweek",
    "BB_perc",
    "room_to_60d_high",
    "gap_pct",
    "lower_wick_ratio",
}

OUTCOME_PREFIXES = (
    "target_before_stop_",
    "stop_before_target_",
    "target_stop_same_day_",
    "no_target_no_stop_",
    "fast_success_",
    "slow_success_",
    "fail_success_",
    "day_to_",
    "validation_",
)

EXCLUDE_COLS = {
    "ticker",
    "stock_name",
    "stock_market",
    "sector_code",
    "today",
    "idx",
    "stop_loss",
    "target_pct",
    "target_class",
    "stop_day",
    "month",
    "quarter",
    "year",
}


# =============================================================================
# Dataclasses
# =============================================================================

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
    fw_metrics: Dict
    score: float
    score_high_precision: float
    score_high_coverage: float
    score_stable: float
    score_aggressive: float
    score_defensive: float
    train_mask_key: str
    selection_mask_key: str

    def name(self) -> str:
        return " AND ".join([a.name() for a in self.atoms])

    def features(self) -> List[str]:
        return sorted(set(a.feature for a in self.atoms))


# =============================================================================
# Utility
# =============================================================================

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["today", "date", "Date", "datetime", "trade_date", "trd_date", "일자", "날짜", "ymd", "YMD"]:
        if c in df.columns:
            return c
    return None


def target_suffix(target_col: str) -> Optional[str]:
    if target_col.startswith("target_before_stop_"):
        return target_col.replace("target_before_stop_", "")
    return None


def quantile_grid(name: str) -> List[float]:
    if name == "decile":
        return [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    if name == "v4":
        return [
            0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.40, 0.50, 0.60, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.95,
            0.975,
        ]

    if name == "fine":
        return [
            0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175,
            0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
            0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975,
        ]

    raise ValueError(f"unknown quantile grid: {name}")


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
        raise ValueError(f"split ratios must sum to 1.0, got {total_ratio}")

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
            "train_start_date": train[date_col].min() if len(train) else None,
            "train_end_date": train[date_col].max() if len(train) else None,
            "selection_start_date": selection[date_col].min() if len(selection) else None,
            "selection_end_date": selection[date_col].max() if len(selection) else None,
            "final_start_date": final[date_col].min() if len(final) else None,
            "final_end_date": final[date_col].max() if len(final) else None,
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


def mask_hash(mask: np.ndarray) -> str:
    m = np.asarray(mask).astype(bool)
    packed = np.packbits(m)
    return hashlib.md5(packed.tobytes()).hexdigest()


def canonicalize_atoms(atoms: Tuple[Atom, ...]) -> Tuple[Atom, ...]:
    return tuple(sorted(atoms, key=lambda a: (a.feature, a.op, a.threshold)))


def canonical_atoms_key(atoms: Tuple[Atom, ...]) -> Tuple[Tuple[str, str, float], ...]:
    return tuple(sorted((a.feature, a.op, round(float(a.threshold), 10)) for a in atoms))


# =============================================================================
# Feature selection and reports
# =============================================================================

def infer_numeric_features(df: pd.DataFrame, target_col: str) -> List[str]:
    features = []

    for c in df.columns:
        if c == target_col:
            continue
        if c in EXCLUDE_COLS:
            continue
        if any(c.startswith(p) for p in OUTCOME_PREFIXES):
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = s.notna().mean()
        nunique = s.nunique(dropna=True)

        if valid_ratio >= 0.80 and nunique >= 8:
            features.append(c)

    return features


def choose_features(df: pd.DataFrame, mode: str, custom_features: Optional[str], target_col: str) -> List[str]:
    if custom_features:
        fs = [x.strip() for x in custom_features.split(",") if x.strip()]
        return [f for f in fs if f in df.columns]

    if mode == "preset":
        return [f for f in DEFAULT_FEATURES if f in df.columns]

    if mode == "auto":
        return infer_numeric_features(df, target_col)

    raise ValueError(f"unknown feature mode: {mode}")


def calc_basic_metrics(y: np.ndarray, mask: np.ndarray) -> Dict:
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask).astype(bool)

    total_count = int(len(y))
    base_rate = float(y.mean()) if total_count else np.nan
    selected_count = int(mask.sum())

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


def build_trade_return_series(df: pd.DataFrame, target_col: str, args) -> pd.Series:
    """
    룰에 선택된 종목의 기대수익률 계산용.

    기본:
    - target_before_stop_x == 1: target_pct 수익
    - stop_before_target_x == 1: stop_loss 손실
    - target_stop_same_day_x == 1: 보수적으로 same_day_return 사용
    - no_target_no_stop_x == 1: validation_close_rate7 있으면 사용, 없으면 0
      단, stop_loss ~ target_pct 범위로 clip
    """
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
            r = r.clip(lower=lo, upper=hi)
            ret[none_hit] = r[none_hit]

    return ret.astype(float)


def add_return_metrics(metrics: Dict, returns: pd.Series, mask: np.ndarray) -> Dict:
    out = dict(metrics)
    mask = np.asarray(mask).astype(bool)

    if mask.sum() == 0:
        out.update({
            "avg_return": np.nan,
            "median_return": np.nan,
            "sum_return": 0.0,
            "win_rate_return": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "win_loss_ratio": np.nan,
            "profit_factor": np.nan,
        })
        return out

    r = pd.to_numeric(returns[mask], errors="coerce").dropna()

    if len(r) == 0:
        out.update({
            "avg_return": np.nan,
            "median_return": np.nan,
            "sum_return": 0.0,
            "win_rate_return": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "win_loss_ratio": np.nan,
            "profit_factor": np.nan,
        })
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


def calc_metrics(df: pd.DataFrame, target_col: str, mask: np.ndarray, args) -> Dict:
    basic = calc_basic_metrics(df[target_col].values, mask)
    returns = build_trade_return_series(df, target_col, args)
    return add_return_metrics(basic, returns, mask)


def feature_decile_report(df: pd.DataFrame, features: List[str], target_col: str, args) -> pd.DataFrame:
    rows = []
    y = df[target_col].astype(int).values
    returns = build_trade_return_series(df, target_col, args)

    for f in features:
        s = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        tmp = pd.DataFrame({"x": s, "y": y, "ret": returns})
        tmp = tmp.dropna(subset=["x", "y"]).copy()

        if tmp["x"].nunique() < 8 or len(tmp) < 50:
            continue

        try:
            tmp["bin"] = pd.qcut(tmp["x"], q=10, duplicates="drop")
        except Exception:
            continue

        base_rate = tmp["y"].mean()

        for b, g in tmp.groupby("bin", observed=True):
            count = len(g)
            pos = int(g["y"].sum())
            precision = pos / count if count else np.nan
            rows.append({
                "feature": f,
                "bin": str(b),
                "bin_left": float(b.left),
                "bin_right": float(b.right),
                "count": int(count),
                "positive_count": pos,
                "precision": precision,
                "lift": precision / base_rate if base_rate > 0 else np.nan,
                "avg_return": float(g["ret"].mean()),
                "median_return": float(g["ret"].median()),
                "base_rate": float(base_rate),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["feature", "bin_left"]).reset_index(drop=True)
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
                thresholds.append((q, float(v)))

        thresholds = sorted(set((q, round(v, 10)) for q, v in thresholds), key=lambda x: x[1])

        for q, th in thresholds:
            for op in [">=", "<="]:
                if op == ">=":
                    mask = (s >= th).fillna(False).to_numpy()
                else:
                    mask = (s <= th).fillna(False).to_numpy()

                m = calc_metrics(df, target_col, mask, args)
                row = {
                    "feature": f,
                    "op": op,
                    "quantile": q,
                    "threshold": th,
                }
                row.update(m)
                rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["feature", "op", "quantile"]).reset_index(drop=True)
    return out


def infer_ops_from_threshold_report(th_report: pd.DataFrame, args) -> Tuple[Dict[str, List[str]], Set[str], pd.DataFrame]:
    rows = []
    allowed_ops: Dict[str, List[str]] = {}
    interval_features: Set[str] = set()

    if len(th_report) == 0:
        return allowed_ops, interval_features, pd.DataFrame()

    for f, g in th_report.groupby("feature"):
        usable = g[g["selected_count"] >= args.auto_ops_min_count].copy()
        if len(usable) == 0:
            allowed_ops[f] = [">=", "<="]
            rows.append({
                "feature": f,
                "decision": "both_low_count",
                "best_ge_precision": np.nan,
                "best_le_precision": np.nan,
                "best_ge_avg_return": np.nan,
                "best_le_avg_return": np.nan,
            })
            continue

        ge = usable[usable["op"] == ">="].copy()
        le = usable[usable["op"] == "<="].copy()

        best_ge = ge.sort_values(["avg_return", "precision_lcb", "precision", "selected_count"], ascending=False).head(1)
        best_le = le.sort_values(["avg_return", "precision_lcb", "precision", "selected_count"], ascending=False).head(1)

        ge_score = -1e18
        le_score = -1e18
        ge_p = np.nan
        le_p = np.nan
        ge_ret = np.nan
        le_ret = np.nan

        if len(best_ge):
            r = best_ge.iloc[0]
            ge_p = r["precision"]
            ge_ret = r["avg_return"]
            ge_score = (
                    r["avg_return"] * 3.0
                    + r["precision_lcb"] * 5.0
                    + r["lift"] * 0.5
                    + math.log1p(r["selected_count"]) * 0.1
            )

        if len(best_le):
            r = best_le.iloc[0]
            le_p = r["precision"]
            le_ret = r["avg_return"]
            le_score = (
                    r["avg_return"] * 3.0
                    + r["precision_lcb"] * 5.0
                    + r["lift"] * 0.5
                    + math.log1p(r["selected_count"]) * 0.1
            )

        diff = abs(ge_score - le_score)
        best_score = max(ge_score, le_score)

        if best_score <= -1e17:
            ops = [">=", "<="]
            decision = "both_no_score"
        elif diff <= args.auto_ops_both_margin:
            ops = [">=", "<="]
            interval_features.add(f)
            decision = "both_interval_candidate"
        elif ge_score > le_score:
            ops = [">="]
            decision = "ge"
        else:
            ops = ["<="]
            decision = "le"

        allowed_ops[f] = ops
        rows.append({
            "feature": f,
            "decision": decision,
            "allowed_ops": ",".join(ops),
            "best_ge_score": ge_score,
            "best_le_score": le_score,
            "best_ge_precision": ge_p,
            "best_le_precision": le_p,
            "best_ge_avg_return": ge_ret,
            "best_le_avg_return": le_ret,
        })

    return allowed_ops, interval_features, pd.DataFrame(rows)


# =============================================================================
# Atoms and rules
# =============================================================================

def make_atoms(
        train: pd.DataFrame,
        features: List[str],
        q_grid: List[float],
        allowed_ops: Dict[str, List[str]],
        threshold_mode: str,
) -> List[Atom]:
    atoms: List[Atom] = []

    if threshold_mode != "quantile":
        raise ValueError("this v5 script currently supports threshold_mode='quantile' only")

    for f in features:
        if f not in train.columns:
            continue

        s = pd.to_numeric(train[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = s.dropna()

        if vals.nunique() < 8:
            continue

        candidates = []
        for q in q_grid:
            v = vals.quantile(q)
            if np.isfinite(v):
                candidates.append((q, float(v)))

        seen = set()
        for q, th in candidates:
            key_th = round(th, 10)
            if key_th in seen:
                continue
            seen.add(key_th)

            for op in allowed_ops.get(f, [">=", "<="]):
                atoms.append(Atom(feature=f, op=op, threshold=float(key_th), source=f"q{q:.3f}"))

    # dedup
    out = []
    seen_atom = set()
    for a in atoms:
        key = (a.feature, a.op, round(a.threshold, 10))
        if key in seen_atom:
            continue
        seen_atom.add(key)
        out.append(a)

    return out


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


def can_add_atom(base_atoms: Tuple[Atom, ...], atom: Atom, interval_features: Set[str]) -> bool:
    used_features = set(a.feature for a in base_atoms)
    same_feature_atoms = [a for a in base_atoms if a.feature == atom.feature]

    if atom.feature not in interval_features:
        if atom.feature in used_features:
            return False
        return True

    # interval feature는 같은 피쳐를 최대 2번 허용하되,
    # 같은 방향 반복은 금지하고, 상하한이 모순이면 금지한다.
    if len(same_feature_atoms) == 0:
        return True

    if len(same_feature_atoms) >= 2:
        return False

    old = same_feature_atoms[0]
    if old.op == atom.op:
        return False

    if old.op == ">=" and atom.op == "<=":
        if old.threshold > atom.threshold:
            return False

    if old.op == "<=" and atom.op == ">=":
        if atom.threshold > old.threshold:
            return False

    return True


def monthly_summary_from_mask(
        df: pd.DataFrame,
        target_col: str,
        mask: np.ndarray,
        args,
) -> Tuple[pd.DataFrame, Dict]:
    df = df.reset_index(drop=True).copy()
    mask = np.asarray(mask).astype(bool)

    rows = []
    for month, g in df.groupby("month", sort=True):
        idx = g.index.to_numpy()
        m = calc_metrics(g.reset_index(drop=True), target_col, mask[idx], args)
        m["month"] = month
        rows.append(m)

    monthly_df = pd.DataFrame(rows)

    if len(monthly_df) == 0:
        return monthly_df, {
            "n_months": 0,
            "n_usable_months": 0,
            "mean_month_precision": np.nan,
            "min_month_precision": np.nan,
            "std_month_precision": np.nan,
            "mean_month_return": np.nan,
            "min_month_return": np.nan,
            "std_month_return": np.nan,
            "pass_month_rate": 0.0,
            "crash_month_count": 999,
        }

    usable = monthly_df[monthly_df["selected_count"] >= args.min_month_count].copy()

    if len(usable) == 0:
        return monthly_df, {
            "n_months": len(monthly_df),
            "n_usable_months": 0,
            "mean_month_precision": np.nan,
            "min_month_precision": np.nan,
            "std_month_precision": np.nan,
            "mean_month_return": np.nan,
            "min_month_return": np.nan,
            "std_month_return": np.nan,
            "pass_month_rate": 0.0,
            "crash_month_count": len(monthly_df),
        }

    pass_mask = (
            (usable["precision"] >= args.month_pass_precision)
            & (usable["avg_return"] >= args.month_pass_avg_return)
    )
    crash_mask = (
            (usable["precision"] < args.month_crash_precision)
            | (usable["avg_return"] < args.month_crash_avg_return)
    )

    return monthly_df, {
        "n_months": len(monthly_df),
        "n_usable_months": len(usable),
        "mean_month_precision": float(usable["precision"].mean()),
        "min_month_precision": float(usable["precision"].min()),
        "std_month_precision": float(usable["precision"].std(ddof=0)),
        "mean_month_return": float(usable["avg_return"].mean()),
        "min_month_return": float(usable["avg_return"].min()),
        "std_month_return": float(usable["avg_return"].std(ddof=0)),
        "pass_month_rate": float(pass_mask.mean()),
        "crash_month_count": int(crash_mask.sum()),
    }


def eval_rule(df: pd.DataFrame, target_col: str, atoms: Tuple[Atom, ...], args):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics(df, target_col, mask, args)
    monthly_df, monthly = monthly_summary_from_mask(df, target_col, mask, args)
    return metrics, monthly, monthly_df, mask


def eval_rule_fast(df: pd.DataFrame, target_col: str, atoms: Tuple[Atom, ...], args):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics(df, target_col, mask, args)
    return metrics, mask


# =============================================================================
# Scoring
# =============================================================================

def safe_num(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if np.isfinite(x):
            return float(x)
        return default
    except Exception:
        return default


def breakeven_precision(args) -> float:
    target = abs(float(args.target_pct))
    stop = abs(float(args.stop_loss))
    if target + stop == 0:
        return 0.5
    return stop / (target + stop)


def beam_score_fast(metrics: Dict, args) -> float:
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

    return (
            safe_num(avg_return) * 25
            + precision * 45
            + safe_num(metrics.get("precision_lcb")) * 30
            + lift * 8
            + math.log1p(count) * 2
    )


def hard_filter_candidate(train_m, selection_m, train_mon, selection_mon, args) -> bool:
    if train_m["selected_count"] < args.min_train_count:
        return False
    if selection_m["selected_count"] < args.min_selection_count:
        return False

    vals = [
        train_m["precision"], selection_m["precision"],
        train_m["lift"], selection_m["lift"],
        selection_m.get("avg_return", np.nan),
    ]
    if not all(np.isfinite(x) for x in vals):
        return False

    if train_m["precision"] < args.min_train_precision:
        return False
    if selection_m["precision"] < args.min_selection_precision:
        return False
    if train_m["lift"] < args.min_train_lift:
        return False
    if selection_m["lift"] < args.min_selection_lift:
        return False
    if selection_m["avg_return"] < args.min_selection_avg_return:
        return False

    gap = abs(train_m["precision"] - selection_m["precision"])
    if gap > args.max_precision_gap:
        return False

    if selection_mon["n_usable_months"] < args.min_selection_usable_months:
        return False
    if selection_mon["pass_month_rate"] < args.min_selection_pass_month_rate:
        return False
    if selection_mon["crash_month_count"] > args.max_selection_crash_months:
        return False

    return True


def base_rule_score(train_m, selection_m, final_m, train_mon, selection_mon, final_mon, args) -> float:
    gap_ts = abs(train_m["precision"] - selection_m["precision"])
    gap_sf = abs(selection_m["precision"] - final_m["precision"]) if np.isfinite(final_m["precision"]) else 0.0

    return (
            safe_num(selection_m["avg_return"]) * 45
            + safe_num(selection_m["precision"]) * 120
            + safe_num(selection_m["precision_lcb"]) * 90
            + safe_num(selection_m["lift"]) * 18
            + safe_num(selection_m["profit_factor"]) * 8
            + safe_num(selection_mon["mean_month_precision"]) * 55
            + safe_num(selection_mon["min_month_precision"]) * 40
            + safe_num(selection_mon["mean_month_return"]) * 25
            + safe_num(selection_mon["min_month_return"]) * 18
            + safe_num(selection_mon["pass_month_rate"]) * 50
            + safe_num(final_m["avg_return"]) * 20
            + safe_num(final_m["precision"]) * 25
            + math.log1p(selection_m["selected_count"]) * args.selection_count_weight
            - gap_ts * 90
            - gap_sf * 25
            - safe_num(selection_mon["std_month_precision"]) * 45
            - safe_num(selection_mon["std_month_return"]) * 10
            - selection_mon["crash_month_count"] * 25
    )


def type_scores(train_m, selection_m, final_m, selection_mon, final_mon, fw, args) -> Dict[str, float]:
    be = breakeven_precision(args)

    sel_count = selection_m["selected_count"]
    fin_count = final_m["selected_count"]

    hp = (
            safe_num(selection_m["precision"]) * 180
            + safe_num(selection_m["precision_lcb"]) * 130
            + safe_num(final_m["precision"]) * 80
            + safe_num(selection_m["avg_return"]) * 35
            + math.log1p(sel_count) * 2
            - abs(safe_num(selection_m["precision"]) - safe_num(final_m["precision"])) * 70
            - selection_mon["crash_month_count"] * 25
    )

    hc = (
            math.log1p(sel_count) * 35
            + math.log1p(fin_count) * 25
            + safe_num(selection_m["coverage"]) * 100
            + safe_num(final_m["coverage"]) * 70
            + max(0.0, safe_num(selection_m["precision"]) - be) * 140
            + safe_num(selection_m["avg_return"]) * 45
            - selection_mon["crash_month_count"] * 15
    )

    st = (
            safe_num(selection_mon["pass_month_rate"]) * 120
            + safe_num(selection_mon["min_month_precision"]) * 90
            + safe_num(selection_mon["min_month_return"]) * 35
            + safe_num(fw.get("fw_pass_split_rate")) * 110
            + safe_num(fw.get("fw_min_precision")) * 90
            + safe_num(fw.get("fw_min_return")) * 35
            + safe_num(selection_m["precision_lcb"]) * 70
            - safe_num(selection_mon["std_month_precision"]) * 80
            - safe_num(selection_mon["std_month_return"]) * 20
            - selection_mon["crash_month_count"] * 35
            - safe_num(fw.get("fw_crash_split_count"), 999) * 30
    )

    ag = (
            safe_num(selection_m["avg_return"]) * 80
            + safe_num(final_m["avg_return"]) * 45
            + math.log1p(sel_count) * 22
            + safe_num(selection_m["coverage"]) * 120
            + max(0.0, safe_num(selection_m["precision"]) - be) * 90
            + safe_num(selection_m["lift"]) * 10
            - selection_mon["crash_month_count"] * 12
    )

    df = (
            safe_num(selection_m["precision_lcb"]) * 150
            + safe_num(selection_m["precision"]) * 120
            + safe_num(final_m["precision_lcb"]) * 100
            + safe_num(selection_mon["min_month_precision"]) * 90
            + safe_num(fw.get("fw_min_precision")) * 80
            + safe_num(selection_m["avg_return"]) * 30
            - selection_mon["crash_month_count"] * 45
            - safe_num(fw.get("fw_crash_split_count"), 999) * 35
    )

    return {
        "score_high_precision": hp,
        "score_high_coverage": hc,
        "score_stable": st,
        "score_aggressive": ag,
        "score_defensive": df,
    }


# =============================================================================
# Forward robustness
# =============================================================================

def build_forward_splits(df: pd.DataFrame, n_splits: int, min_train_months: int) -> List[Dict]:
    months = sorted(df["month"].dropna().unique())
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
            m = calc_metrics(test_df, target_col, mask, args)

            row = {
                "rank": rank,
                "rule": r.name(),
                "valid_month": valid_month,
                "n_train_months_before": len(s["train_months"]),
            }
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
            rows.append({
                "rank": rank,
                "rule": rule,
                "fw_n_splits": len(g),
                "fw_n_usable_splits": 0,
                "fw_mean_precision": np.nan,
                "fw_min_precision": np.nan,
                "fw_mean_return": np.nan,
                "fw_min_return": np.nan,
                "fw_total_count": 0,
                "fw_pass_split_rate": 0.0,
                "fw_crash_split_count": len(g),
            })
            continue

        pass_mask = (
                (usable["precision"] >= args.fw_min_precision)
                & (usable["avg_return"] >= args.fw_min_avg_return)
        )
        crash_mask = (
                (usable["precision"] < args.fw_crash_precision)
                | (usable["avg_return"] < args.fw_crash_avg_return)
        )

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


# =============================================================================
# Search
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


def make_rule_object(atoms, train, selection, final_test, args) -> Rule:
    train_m, train_mon, _, train_mask = eval_rule(train, args.target, atoms, args)
    selection_m, selection_mon, _, selection_mask = eval_rule(selection, args.target, atoms, args)
    final_m, final_mon, _, final_mask = eval_rule(final_test, args.target, atoms, args)

    score = base_rule_score(train_m, selection_m, final_m, train_mon, selection_mon, final_mon, args)

    dummy_fw = {
        "fw_pass_split_rate": 0.0,
        "fw_min_precision": np.nan,
        "fw_min_return": np.nan,
        "fw_crash_split_count": 999,
    }
    ts = type_scores(train_m, selection_m, final_m, selection_mon, final_mon, dummy_fw, args)

    return Rule(
        atoms=atoms,
        train_metrics=train_m,
        selection_metrics=selection_m,
        final_metrics=final_m,
        train_monthly=train_mon,
        selection_monthly=selection_mon,
        final_monthly=final_mon,
        fw_metrics=dummy_fw,
        score=score,
        score_high_precision=ts["score_high_precision"],
        score_high_coverage=ts["score_high_coverage"],
        score_stable=ts["score_stable"],
        score_aggressive=ts["score_aggressive"],
        score_defensive=ts["score_defensive"],
        train_mask_key=mask_hash(train_mask),
        selection_mask_key=mask_hash(selection_mask),
    )


def search_rules(train, selection, final_test, atoms, interval_features, args) -> List[Rule]:
    beam = [tuple()]
    selected_candidates: List[Rule] = []
    seen_rules = set()
    seen_train_masks = set()

    for depth in range(1, args.max_depth + 1):
        print(f"[INFO] searching depth={depth}")
        beam_candidates = []

        for base_atoms in beam:
            for atom in atoms:
                if not can_add_atom(base_atoms, atom, interval_features):
                    continue

                new_atoms = canonicalize_atoms(tuple(list(base_atoms) + [atom]))
                key = canonical_atoms_key(new_atoms)

                if key in seen_rules:
                    continue
                seen_rules.add(key)

                train_fast_m, train_mask = eval_rule_fast(train, args.target, new_atoms, args)
                bscore = beam_score_fast(train_fast_m, args)
                if bscore <= -1e17:
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
                if np.isfinite(selection_fast_m.get("avg_return", np.nan)):
                    if selection_fast_m["avg_return"] < args.fast_min_selection_avg_return:
                        continue

                rule = make_rule_object(new_atoms, train, selection, final_test, args)

                if hard_filter_candidate(
                        rule.train_metrics,
                        rule.selection_metrics,
                        rule.train_monthly,
                        rule.selection_monthly,
                        args,
                ):
                    selected_candidates.append(rule)
                    beam_candidates.append(rule)
                else:
                    # 최종 후보는 아니어도 beam에는 남길 수 있음
                    rule.score = bscore
                    beam_candidates.append(rule)

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

    deduped: List[Rule] = []
    for r in selected_candidates:
        if is_near_duplicate(r, deduped, args):
            continue
        deduped.append(r)
        if len(deduped) >= args.top_k:
            break

    return deduped


def simplify_rule_by_dropping_atoms(rule: Rule, train, selection, final_test, args) -> Rule:
    current = rule
    improved = True

    while improved and len(current.atoms) > 1:
        improved = False
        best_candidate = None
        best_tuple = None

        for i in range(len(current.atoms)):
            new_atoms = tuple(a for j, a in enumerate(current.atoms) if j != i)
            new_atoms = canonicalize_atoms(new_atoms)

            cand = make_rule_object(new_atoms, train, selection, final_test, args)

            if not hard_filter_candidate(
                    cand.train_metrics,
                    cand.selection_metrics,
                    cand.train_monthly,
                    cand.selection_monthly,
                    args,
            ):
                continue

            old_p = current.selection_metrics["precision"]
            new_p = cand.selection_metrics["precision"]
            old_count = current.selection_metrics["selected_count"]
            new_count = cand.selection_metrics["selected_count"]

            if new_p + args.simplify_precision_drop < old_p:
                continue

            if new_count < old_count * args.simplify_min_count_ratio:
                continue

            tup = (
                -len(new_atoms),
                cand.score,
                cand.selection_metrics["avg_return"],
                cand.selection_metrics["precision"],
                cand.selection_metrics["selected_count"],
            )

            if best_tuple is None or tup > best_tuple:
                best_tuple = tup
                best_candidate = cand

        if best_candidate is not None:
            current = best_candidate
            improved = True

    return current


# =============================================================================
# Output tables
# =============================================================================

def flatten(prefix: str, d: Dict) -> Dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def rules_to_df(rules: List[Rule], args) -> pd.DataFrame:
    rows = []
    be = breakeven_precision(args)

    for rank, r in enumerate(rules, start=1):
        row = {
            "rank": rank,
            "rule": r.name(),
            "n_atoms": len(r.atoms),
            "features": ",".join(r.features()),
            "score": r.score,
            "score_high_precision": r.score_high_precision,
            "score_high_coverage": r.score_high_coverage,
            "score_stable": r.score_stable,
            "score_aggressive": r.score_aggressive,
            "score_defensive": r.score_defensive,
            "breakeven_precision": be,
        }

        row.update(flatten("train", r.train_metrics))
        row.update(flatten("selection", r.selection_metrics))
        row.update(flatten("final", r.final_metrics))
        row.update(flatten("train_month", r.train_monthly))
        row.update(flatten("selection_month", r.selection_monthly))
        row.update(flatten("final_month", r.final_monthly))
        row.update(flatten("fw", r.fw_metrics))

        row["precision_gap_train_selection_abs"] = abs(row["train_precision"] - row["selection_precision"])
        row["precision_gap_selection_final_abs"] = abs(row["selection_precision"] - row["final_precision"])

        row["is_high_precision"] = (
                row["selection_precision"] >= args.hp_min_selection_precision
                and row["selection_precision_lcb"] >= args.hp_min_selection_lcb
                and row["selection_selected_count"] >= args.hp_min_selection_count
                and row["selection_avg_return"] >= args.hp_min_avg_return
                and row["final_precision"] >= args.hp_min_final_precision
        )

        row["is_high_coverage"] = (
                row["selection_selected_count"] >= args.hc_min_selection_count
                and row["final_selected_count"] >= args.hc_min_final_count
                and row["selection_precision"] >= args.hc_min_selection_precision
                and row["selection_avg_return"] >= args.hc_min_avg_return
        )

        row["is_stable"] = (
                row["selection_month_pass_month_rate"] >= args.st_min_pass_month_rate
                and row["selection_month_crash_month_count"] <= args.st_max_crash_months
                and row["fw_fw_pass_split_rate"] >= args.st_min_fw_pass_split_rate
                and row["fw_fw_crash_split_count"] <= args.st_max_fw_crash_splits
                and row["selection_avg_return"] >= args.st_min_avg_return
        )

        row["is_aggressive"] = (
                row["selection_precision"] >= args.ag_min_selection_precision
                and row["selection_avg_return"] >= args.ag_min_avg_return
                and row["selection_selected_count"] >= args.ag_min_selection_count
                and row["selection_coverage"] >= args.ag_min_coverage
        )

        row["is_defensive"] = (
                row["selection_precision"] >= args.df_min_selection_precision
                and row["selection_precision_lcb"] >= args.df_min_selection_lcb
                and row["selection_month_crash_month_count"] <= args.df_max_crash_months
                and row["fw_fw_crash_split_count"] <= args.df_max_fw_crash_splits
                and row["selection_avg_return"] >= args.df_min_avg_return
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def collect_monthly_details(rules: List[Rule], train, selection, final_test, args) -> pd.DataFrame:
    dfs = []

    for rank, r in enumerate(rules, start=1):
        for dataset_name, df in [
            ("TRAIN", train),
            ("SELECTION_VALID", selection),
            ("FINAL_TEST", final_test),
        ]:
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
# OR rule sets
# =============================================================================

def or_mask(df: pd.DataFrame, rules: List[Rule]) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    for r in rules:
        mask |= apply_rule(df, r.atoms)
    return mask


def greedy_or_ruleset(
        rules: List[Rule],
        train: pd.DataFrame,
        selection: pd.DataFrame,
        final_test: pd.DataFrame,
        args,
        sort_col: str,
        min_precision: float,
        min_avg_return: float,
        min_incremental_count: int,
        max_size: int,
) -> pd.DataFrame:
    if not rules:
        return pd.DataFrame()

    candidates = sorted(rules, key=lambda r: getattr(r, sort_col), reverse=True)

    selected: List[Rule] = []
    rows = []
    current_mask_sel = np.zeros(len(selection), dtype=bool)

    for r in candidates:
        if len(selected) >= max_size:
            break

        m = apply_rule(selection, r.atoms)
        inc = int((m & ~current_mask_sel).sum())
        if inc < min_incremental_count:
            continue

        trial = selected + [r]
        sel_mask = or_mask(selection, trial)
        sel_m = calc_metrics(selection, args.target, sel_mask, args)

        if sel_m["selected_count"] == 0:
            continue
        if sel_m["precision"] < min_precision:
            continue
        if sel_m["avg_return"] < min_avg_return:
            continue

        selected = trial
        current_mask_sel = sel_mask

        train_m = calc_metrics(train, args.target, or_mask(train, selected), args)
        final_m = calc_metrics(final_test, args.target, or_mask(final_test, selected), args)
        _, sel_mon = monthly_summary_from_mask(selection, args.target, sel_mask, args)

        rows.append({
            "step": len(selected),
            "added_rule": r.name(),
            "or_rules": " || ".join([x.name() for x in selected]),
            "train_count": train_m["selected_count"],
            "train_precision": train_m["precision"],
            "train_avg_return": train_m["avg_return"],
            "selection_count": sel_m["selected_count"],
            "selection_precision": sel_m["precision"],
            "selection_precision_lcb": sel_m["precision_lcb"],
            "selection_avg_return": sel_m["avg_return"],
            "selection_coverage": sel_m["coverage"],
            "selection_month_pass_rate": sel_mon["pass_month_rate"],
            "selection_month_crash_count": sel_mon["crash_month_count"],
            "final_count": final_m["selected_count"],
            "final_precision": final_m["precision"],
            "final_avg_return": final_m["avg_return"],
            "incremental_selection_count": inc,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="stable_rule_auto_v5_out")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--date-col", default=None)

    parser.add_argument("--feature-mode", default="preset", choices=["preset", "auto"])
    parser.add_argument("--features", default=None, help="comma-separated custom feature list")

    parser.add_argument("--threshold-mode", default="quantile", choices=["quantile"])
    parser.add_argument("--quantile-grid", default="v4", choices=["decile", "v4", "fine"])

    parser.add_argument("--ops-mode", default="preset", choices=["preset", "wide", "auto"])
    parser.add_argument("--interval-mode", default="preset", choices=["preset", "none", "auto"])

    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--selection-valid-ratio", type=float, default=0.20)
    parser.add_argument("--final-test-ratio", type=float, default=0.20)

    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=200)

    parser.add_argument("--target-pct", type=float, default=10.0)
    parser.add_argument("--stop-loss", type=float, default=-6.0)
    parser.add_argument("--same-day-return", type=float, default=-6.0)
    parser.add_argument("--no-hit-return-col", default="validation_close_rate7")

    # search filters
    parser.add_argument("--beam-min-count", type=int, default=45)
    parser.add_argument("--beam-min-lift", type=float, default=1.00)

    parser.add_argument("--fast-min-selection-count", type=int, default=18)
    parser.add_argument("--fast-min-selection-precision", type=float, default=0.34)
    parser.add_argument("--fast-min-selection-lift", type=float, default=1.05)
    parser.add_argument("--fast-min-selection-avg-return", type=float, default=-0.30)

    parser.add_argument("--min-train-count", type=int, default=50)
    parser.add_argument("--min-selection-count", type=int, default=25)
    parser.add_argument("--min-train-precision", type=float, default=0.34)
    parser.add_argument("--min-selection-precision", type=float, default=0.38)
    parser.add_argument("--min-train-lift", type=float, default=1.02)
    parser.add_argument("--min-selection-lift", type=float, default=1.08)
    parser.add_argument("--min-selection-avg-return", type=float, default=0.00)
    parser.add_argument("--max-precision-gap", type=float, default=0.25)

    # monthly
    parser.add_argument("--min-month-count", type=int, default=5)
    parser.add_argument("--month-pass-precision", type=float, default=0.38)
    parser.add_argument("--month-pass-avg-return", type=float, default=0.00)
    parser.add_argument("--month-crash-precision", type=float, default=0.30)
    parser.add_argument("--month-crash-avg-return", type=float, default=-1.00)

    parser.add_argument("--min-selection-usable-months", type=int, default=2)
    parser.add_argument("--min-selection-pass-month-rate", type=float, default=0.20)
    parser.add_argument("--max-selection-crash-months", type=int, default=2)

    parser.add_argument("--selection-count-weight", type=float, default=6.0)

    # auto ops
    parser.add_argument("--auto-ops-min-count", type=int, default=100)
    parser.add_argument("--auto-ops-both-margin", type=float, default=0.35)

    # dedup / simplify
    parser.add_argument("--dup-count-tol", type=float, default=0.10)
    parser.add_argument("--dup-precision-gain", type=float, default=0.01)

    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--simplify-precision-drop", type=float, default=0.015)
    parser.add_argument("--simplify-min-count-ratio", type=float, default=0.85)

    # forward
    parser.add_argument("--fw-top-rules", type=int, default=200)
    parser.add_argument("--fw-splits", type=int, default=8)
    parser.add_argument("--fw-min-train-months", type=int, default=6)
    parser.add_argument("--fw-min-count", type=int, default=5)
    parser.add_argument("--fw-min-precision", type=float, default=0.38)
    parser.add_argument("--fw-min-avg-return", type=float, default=0.00)
    parser.add_argument("--fw-crash-precision", type=float, default=0.30)
    parser.add_argument("--fw-crash-avg-return", type=float, default=-1.00)

    # category thresholds
    parser.add_argument("--hp-min-selection-precision", type=float, default=0.50)
    parser.add_argument("--hp-min-selection-lcb", type=float, default=0.38)
    parser.add_argument("--hp-min-selection-count", type=int, default=25)
    parser.add_argument("--hp-min-avg-return", type=float, default=0.50)
    parser.add_argument("--hp-min-final-precision", type=float, default=0.38)

    parser.add_argument("--hc-min-selection-count", type=int, default=80)
    parser.add_argument("--hc-min-final-count", type=int, default=40)
    parser.add_argument("--hc-min-selection-precision", type=float, default=0.40)
    parser.add_argument("--hc-min-avg-return", type=float, default=0.10)

    parser.add_argument("--st-min-pass-month-rate", type=float, default=0.50)
    parser.add_argument("--st-max-crash-months", type=int, default=1)
    parser.add_argument("--st-min-fw-pass-split-rate", type=float, default=0.50)
    parser.add_argument("--st-max-fw-crash-splits", type=int, default=2)
    parser.add_argument("--st-min-avg-return", type=float, default=0.00)

    parser.add_argument("--ag-min-selection-precision", type=float, default=0.38)
    parser.add_argument("--ag-min-avg-return", type=float, default=0.10)
    parser.add_argument("--ag-min-selection-count", type=int, default=70)
    parser.add_argument("--ag-min-coverage", type=float, default=0.03)

    parser.add_argument("--df-min-selection-precision", type=float, default=0.48)
    parser.add_argument("--df-min-selection-lcb", type=float, default=0.36)
    parser.add_argument("--df-max-crash-months", type=int, default=1)
    parser.add_argument("--df-max-fw-crash-splits", type=int, default=1)
    parser.add_argument("--df-min-avg-return", type=float, default=0.20)

    # OR sets
    parser.add_argument("--or-max-size", type=int, default=8)
    parser.add_argument("--or-min-incremental-selection-count", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df_raw = pd.read_csv(args.csv, low_memory=False)
    date_col = args.date_col or find_date_col(df_raw)
    if date_col is None:
        raise ValueError("date column not found. Use --date-col today")

    df = prepare_df(df_raw, args.target, date_col)

    features = choose_features(df, args.feature_mode, args.features, args.target)
    missing_defaults = [f for f in DEFAULT_FEATURES if f not in df.columns]

    train, selection, final_test, split_info = split_train_selection_test_by_date(
        df=df,
        date_col=date_col,
        train_ratio=args.train_ratio,
        selection_valid_ratio=args.selection_valid_ratio,
        final_test_ratio=args.final_test_ratio,
    )

    q_grid = quantile_grid(args.quantile_grid)

    print("=" * 100)
    print("[INFO] rows:", len(df))
    print("[INFO] target:", args.target)
    print("[INFO] target base rate:", df[args.target].mean())
    print("[INFO] target_pct:", args.target_pct, "stop_loss:", args.stop_loss)
    print("[INFO] breakeven precision:", breakeven_precision(args))
    print("[INFO] train:", len(train), "base_rate:", train[args.target].mean())
    print("[INFO] selection:", len(selection), "base_rate:", selection[args.target].mean())
    print("[INFO] final_test:", len(final_test), "base_rate:", final_test[args.target].mean())
    print("[INFO] date_col:", date_col)
    print("[INFO] features:", features)
    print("[INFO] missing default features:", missing_defaults)
    print("[INFO] split_info:", split_info)
    print("[INFO] quantile_grid:", q_grid)
    print("=" * 100)

    pd.DataFrame([split_info]).to_csv(os.path.join(args.out, "00_split_info.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "feature": features,
    }).to_csv(os.path.join(args.out, "00_features_used.csv"), index=False, encoding="utf-8-sig")

    # 1) Feature reports
    decile_report = feature_decile_report(train, features, args.target, args)
    decile_report.to_csv(os.path.join(args.out, "01_feature_decile_report_train.csv"), index=False, encoding="utf-8-sig")

    threshold_report = cumulative_threshold_report(train, features, args.target, q_grid, args)
    threshold_report.to_csv(os.path.join(args.out, "02_feature_threshold_report_train.csv"), index=False, encoding="utf-8-sig")

    # 2) Ops and interval decisions
    if args.ops_mode == "preset":
        allowed_ops = {f: PRESET_ALLOWED_OPS.get(f, [">=", "<="]) for f in features}
        ops_decision_df = pd.DataFrame([
            {"feature": f, "decision": "preset", "allowed_ops": ",".join(allowed_ops[f])}
            for f in features
        ])
        auto_interval_features = set()

    elif args.ops_mode == "wide":
        allowed_ops = {f: [">=", "<="] for f in features}
        ops_decision_df = pd.DataFrame([
            {"feature": f, "decision": "wide", "allowed_ops": ">=,<="}
            for f in features
        ])
        auto_interval_features = set(features)

    elif args.ops_mode == "auto":
        allowed_ops, auto_interval_features, ops_decision_df = infer_ops_from_threshold_report(threshold_report, args)

    else:
        raise ValueError(f"unknown ops_mode: {args.ops_mode}")

    if args.interval_mode == "preset":
        interval_features = {f for f in PRESET_INTERVAL_FEATURES if f in features}
    elif args.interval_mode == "none":
        interval_features = set()
    elif args.interval_mode == "auto":
        if args.ops_mode == "auto":
            interval_features = auto_interval_features
        else:
            interval_features = {f for f in features if len(allowed_ops.get(f, [])) >= 2}
    else:
        raise ValueError(f"unknown interval_mode: {args.interval_mode}")

    ops_decision_df["is_interval_feature"] = ops_decision_df["feature"].isin(interval_features)
    ops_decision_df.to_csv(os.path.join(args.out, "03_allowed_ops_and_interval.csv"), index=False, encoding="utf-8-sig")

    print("[INFO] allowed_ops:")
    for f in features:
        print(" ", f, allowed_ops.get(f, [">=", "<="]), "interval" if f in interval_features else "")

    # 3) Atoms
    atoms = make_atoms(
        train=train,
        features=features,
        q_grid=q_grid,
        allowed_ops=allowed_ops,
        threshold_mode=args.threshold_mode,
    )

    atoms_df = pd.DataFrame([
        {
            "feature": a.feature,
            "op": a.op,
            "threshold": a.threshold,
            "source": a.source,
            "atom": a.name(),
        }
        for a in atoms
    ])
    atoms_df.to_csv(os.path.join(args.out, "04_atoms.csv"), index=False, encoding="utf-8-sig")
    print("[INFO] atoms:", len(atoms))

    # 4) Rule search
    rules = search_rules(train, selection, final_test, atoms, interval_features, args)

    if args.simplify:
        print("[INFO] simplifying rules")
        simplified = []
        for r in rules:
            simplified.append(simplify_rule_by_dropping_atoms(r, train, selection, final_test, args))

        simplified = sorted(simplified, key=lambda r: r.score, reverse=True)

        deduped = []
        for r in simplified:
            if is_near_duplicate(r, deduped, args):
                continue
            deduped.append(r)
            if len(deduped) >= args.top_k:
                break
        rules = deduped

    # 5) Forward evaluation
    fw_df = fixed_rule_forward_eval(df, args.target, rules[:args.fw_top_rules], args)
    fw_df.to_csv(os.path.join(args.out, "05_fixed_rule_forward_eval.csv"), index=False, encoding="utf-8-sig")

    fw_summary = summarize_forward(fw_df, args)
    fw_summary.to_csv(os.path.join(args.out, "06_fixed_rule_forward_summary.csv"), index=False, encoding="utf-8-sig")

    # Merge fw back into rule objects and recompute type scores
    fw_map = {}
    if len(fw_summary):
        for _, row in fw_summary.iterrows():
            fw_map[(int(row["rank"]), row["rule"])] = row.to_dict()

    for idx, r in enumerate(rules, start=1):
        fw = fw_map.get((idx, r.name()), {
            "fw_n_splits": 0,
            "fw_n_usable_splits": 0,
            "fw_mean_precision": np.nan,
            "fw_min_precision": np.nan,
            "fw_mean_return": np.nan,
            "fw_min_return": np.nan,
            "fw_total_count": 0,
            "fw_pass_split_rate": 0.0,
            "fw_crash_split_count": 999,
        })
        r.fw_metrics = fw

        ts = type_scores(
            r.train_metrics,
            r.selection_metrics,
            r.final_metrics,
            r.selection_monthly,
            r.final_monthly,
            fw,
            args,
        )
        r.score_high_precision = ts["score_high_precision"]
        r.score_high_coverage = ts["score_high_coverage"]
        r.score_stable = ts["score_stable"]
        r.score_aggressive = ts["score_aggressive"]
        r.score_defensive = ts["score_defensive"]

    # 6) Save all rules
    rules_df = rules_to_df(rules, args)
    rules_df.to_csv(os.path.join(args.out, "07_selected_rules_all.csv"), index=False, encoding="utf-8-sig")

    monthly_df = collect_monthly_details(rules, train, selection, final_test, args)
    monthly_df.to_csv(os.path.join(args.out, "08_monthly_details.csv"), index=False, encoding="utf-8-sig")

    if len(rules_df):
        hp = rules_df[rules_df["is_high_precision"]].sort_values(
            ["score_high_precision", "selection_precision", "selection_avg_return"],
            ascending=False,
        )
        hc = rules_df[rules_df["is_high_coverage"]].sort_values(
            ["score_high_coverage", "selection_selected_count", "selection_avg_return"],
            ascending=False,
        )
        st = rules_df[rules_df["is_stable"]].sort_values(
            ["score_stable", "fw_fw_pass_split_rate", "selection_month_pass_month_rate"],
            ascending=False,
        )
        ag = rules_df[rules_df["is_aggressive"]].sort_values(
            ["score_aggressive", "selection_avg_return", "selection_selected_count"],
            ascending=False,
        )
        dfv = rules_df[rules_df["is_defensive"]].sort_values(
            ["score_defensive", "selection_precision_lcb", "fw_fw_crash_split_count"],
            ascending=[False, False, True],
        )

        hp.to_csv(os.path.join(args.out, "09_rules_high_precision.csv"), index=False, encoding="utf-8-sig")
        hc.to_csv(os.path.join(args.out, "10_rules_high_coverage.csv"), index=False, encoding="utf-8-sig")
        st.to_csv(os.path.join(args.out, "11_rules_stable.csv"), index=False, encoding="utf-8-sig")
        ag.to_csv(os.path.join(args.out, "12_rules_aggressive.csv"), index=False, encoding="utf-8-sig")
        dfv.to_csv(os.path.join(args.out, "13_rules_defensive.csv"), index=False, encoding="utf-8-sig")
    else:
        for fn in [
            "09_rules_high_precision.csv",
            "10_rules_high_coverage.csv",
            "11_rules_stable.csv",
            "12_rules_aggressive.csv",
            "13_rules_defensive.csv",
        ]:
            pd.DataFrame().to_csv(os.path.join(args.out, fn), index=False, encoding="utf-8-sig")

    # 7) Greedy OR rule sets by type
    if len(rules):
        high_precision_rules = sorted(rules, key=lambda r: r.score_high_precision, reverse=True)[:50]
        high_coverage_rules = sorted(rules, key=lambda r: r.score_high_coverage, reverse=True)[:50]
        stable_rules = sorted(rules, key=lambda r: r.score_stable, reverse=True)[:50]
        aggressive_rules = sorted(rules, key=lambda r: r.score_aggressive, reverse=True)[:50]
        defensive_rules = sorted(rules, key=lambda r: r.score_defensive, reverse=True)[:50]

        or_hp = greedy_or_ruleset(
            high_precision_rules, train, selection, final_test, args,
            sort_col="score_high_precision",
            min_precision=args.hp_min_selection_precision,
            min_avg_return=args.hp_min_avg_return,
            min_incremental_count=args.or_min_incremental_selection_count,
            max_size=args.or_max_size,
        )
        or_hc = greedy_or_ruleset(
            high_coverage_rules, train, selection, final_test, args,
            sort_col="score_high_coverage",
            min_precision=args.hc_min_selection_precision,
            min_avg_return=args.hc_min_avg_return,
            min_incremental_count=args.or_min_incremental_selection_count,
            max_size=args.or_max_size,
        )
        or_st = greedy_or_ruleset(
            stable_rules, train, selection, final_test, args,
            sort_col="score_stable",
            min_precision=args.st_min_avg_return,
            min_avg_return=args.st_min_avg_return,
            min_incremental_count=args.or_min_incremental_selection_count,
            max_size=args.or_max_size,
        )
        or_ag = greedy_or_ruleset(
            aggressive_rules, train, selection, final_test, args,
            sort_col="score_aggressive",
            min_precision=args.ag_min_selection_precision,
            min_avg_return=args.ag_min_avg_return,
            min_incremental_count=args.or_min_incremental_selection_count,
            max_size=args.or_max_size,
        )
        or_dfv = greedy_or_ruleset(
            defensive_rules, train, selection, final_test, args,
            sort_col="score_defensive",
            min_precision=args.df_min_selection_precision,
            min_avg_return=args.df_min_avg_return,
            min_incremental_count=args.or_min_incremental_selection_count,
            max_size=args.or_max_size,
        )

        or_hp.to_csv(os.path.join(args.out, "14_or_high_precision.csv"), index=False, encoding="utf-8-sig")
        or_hc.to_csv(os.path.join(args.out, "15_or_high_coverage.csv"), index=False, encoding="utf-8-sig")
        or_st.to_csv(os.path.join(args.out, "16_or_stable.csv"), index=False, encoding="utf-8-sig")
        or_ag.to_csv(os.path.join(args.out, "17_or_aggressive.csv"), index=False, encoding="utf-8-sig")
        or_dfv.to_csv(os.path.join(args.out, "18_or_defensive.csv"), index=False, encoding="utf-8-sig")

    # 8) Summary
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
        "rules_count": len(rules_df),
        "high_precision_count": int(rules_df["is_high_precision"].sum()) if len(rules_df) else 0,
        "high_coverage_count": int(rules_df["is_high_coverage"].sum()) if len(rules_df) else 0,
        "stable_count": int(rules_df["is_stable"].sum()) if len(rules_df) else 0,
        "aggressive_count": int(rules_df["is_aggressive"].sum()) if len(rules_df) else 0,
        "defensive_count": int(rules_df["is_defensive"].sum()) if len(rules_df) else 0,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out, "99_run_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if len(rules_df):
        show_cols = [
            "rank", "rule", "n_atoms",
            "selection_selected_count", "selection_precision", "selection_precision_lcb",
            "selection_avg_return", "selection_profit_factor",
            "final_selected_count", "final_precision", "final_avg_return",
            "selection_month_pass_month_rate", "selection_month_crash_month_count",
            "fw_fw_pass_split_rate", "fw_fw_crash_split_count",
            "is_high_precision", "is_high_coverage", "is_stable", "is_aggressive", "is_defensive",
        ]
        show_cols = [c for c in show_cols if c in rules_df.columns]
        print("\n[TOP RULES]")
        print(rules_df[show_cols].head(30).to_string(index=False))
    else:
        print("No selected rules found.")

    print("=" * 100)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 100)


if __name__ == "__main__":
    main()

"""
python stable_rule_miner_auto_v5.py ^
  --csv csv/low_result_7_v2_desc.csv ^
  --out stable_rule_auto_v5_fast ^
  --target target_before_stop_10 ^
  --threshold-mode quantile ^
  --quantile-grid decile ^
  --ops-mode preset ^
  --interval-mode preset ^
  --max-depth 3 ^
  --beam-width 300 ^
  --top-k 80 ^
  --fw-top-rules 50 ^
  --fw-splits 4
"""