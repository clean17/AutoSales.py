"""
target_class == 0 no-bounce 회피 룰 마이닝 v5

목적
- no-bounce 후보 target_class == 0 제거 룰 생성
- coverage 5% 달성 가능성 진단
- class23 / class3 손실이 너무 크지 않은 실전형 룰셋 선택
- workers 기반 chunked multiprocessing 지원

추천 실행:
    python low/8-2_find_no_bounce_avoid_rules.py ^
      --mode all ^
      --save-all-scenarios ^
      --workers 8

빠른 테스트:
    python low/8-2_find_no_bounce_avoid_rules.py ^
      --mode all ^
      --save-all-scenarios ^
      --workers 8 ^
      --fast
"""

from __future__ import annotations

import argparse
import heapq
import math
import os, sys
from pathlib import Path

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_features

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from typing import Iterable
import multiprocessing as mp

import numpy as np
import pandas as pd


# =============================================================================
# Path
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

csv_dir = os.path.join(project_root, "csv")
os.makedirs(csv_dir, exist_ok=True)

CSV_PATH = os.path.join(csv_dir, "low_result_7.csv")
OUT_PATH = Path("lowscan_target0_highprob_rules.py")


# =============================================================================
# Target / split
# =============================================================================

TARGET_CLASS = 0
DATE_COL = "today"

# 최근 구간 검증을 강하게 보고 싶으면 0.20 유지
# 과거 결과와 맞추려면 0.10으로 변경
VALID_RATIO = 0.20

TARGET_COVERAGE_5 = 0.050
PRACTICAL_COVERAGE_4 = 0.040


# =============================================================================
# Feature list
# =============================================================================

FEATURE_COLUMNS = [
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

    "price_power_value",
    "body_value_power",
    "room_to_20d_high",
    "room_to_60d_high",
    "rebound_vs_prior_drop",
]


# =============================================================================
# Mining settings
# =============================================================================

N_QUANTILES = 22
MIN_UNIQUE_VALUES = 8

BEAM = 22000
TOP_N = 22000
MIN_CNT = 25
MAX_DEPTH = 5
MAX_RULES = 280

MIN_CLASS0_RATE = 0.32
MIN_LIFT = 0.66
MIN_WILSON_LOW = 0.160

EXPAND_MIN_CLASS0_RATE = [0.25, 0.30, 0.36, 0.42, 0.48]
EXPAND_MIN_LIFT = [0.68, 0.78, 0.90, 1.02, 1.12]

PRECISION_POWER = 2.1
LIFT_POWER = 1.0
WILSON_POWER = 1.25
COVERAGE_POWER = 1.25

CLASS1_PENALTY = 0.35
CLASS2_PENALTY = 3.20
CLASS3_PENALTY = 5.50

USE_FEATURE_GROUP_LIMITS = True


# =============================================================================
# Valid filter tiers
# =============================================================================

VALID_FILTER_TIERS = [
    {
        "name": "micro_zero_class23_pool",
        "min_cnt": 1,
        "min_class0_rate": 0.45,
        "min_lift": 0.85,
        "min_wilson_low": 0.00,
        "max_class1_rate": 1.00,
        "max_class2_rate": 0.00,
        "max_class3_rate": 0.00,
        "max_class23_rate": 0.00,
    },
    {
        "name": "ultra_low_class23_pool",
        "min_cnt": 2,
        "min_class0_rate": 0.34,
        "min_lift": 0.65,
        "min_wilson_low": 0.04,
        "max_class1_rate": 0.92,
        "max_class2_rate": 0.12,
        "max_class3_rate": 0.06,
        "max_class23_rate": 0.14,
    },
    {
        "name": "strict_class23_pool",
        "min_cnt": 3,
        "min_class0_rate": 0.30,
        "min_lift": 0.58,
        "min_wilson_low": 0.03,
        "max_class1_rate": 0.94,
        "max_class2_rate": 0.20,
        "max_class3_rate": 0.10,
        "max_class23_rate": 0.24,
    },
    {
        "name": "reach_low_class23_pool",
        "min_cnt": 3,
        "min_class0_rate": 0.25,
        "min_lift": 0.50,
        "min_wilson_low": 0.02,
        "max_class1_rate": 0.96,
        "max_class2_rate": 0.32,
        "max_class3_rate": 0.18,
        "max_class23_rate": 0.42,
    },
    {
        "name": "diagnostic_pool",
        "min_cnt": 3,
        "min_class0_rate": 0.20,
        "min_lift": 0.42,
        "min_wilson_low": 0.01,
        "max_class1_rate": 0.98,
        "max_class2_rate": 0.50,
        "max_class3_rate": 0.34,
        "max_class23_rate": 0.65,
    },
]


# =============================================================================
# Feature groups
# =============================================================================

def get_feature_groups() -> tuple[dict[str, str], dict[str, int]]:
    feature_groups = {
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

    group_limits = {
        "VOLATILITY": 2,
        "PRICE": 1,
        "DROP": 1,
        "GAP": 1,
        "WEEK_POSITION": 1,
        "POSITION": 1,
        "TREND": 1,
        "VOLUME": 1,
        "BAND": 1,
        "CANDLE": 2,
        "INTRADAY": 1,
        "REBOUND": 2,
        "POWER": 1,
        "HIGH_ROOM": 1,
    }

    return feature_groups, group_limits


# =============================================================================
# Feature / literal utilities
# =============================================================================


def _thresholds_for_series(s: pd.Series, n_quantiles: int) -> list[float]:
    qs = np.linspace(0.02, 0.98, n_quantiles)

    vals = (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if vals.empty:
        return []

    ths = vals.quantile(qs).to_numpy()

    return sorted({round(float(x), 10) for x in ths if np.isfinite(x)})


def default_extra_thresholds() -> dict[str, list[tuple[str, float]]]:
    return {
        "vol5": [
            ("<=", 1.5), ("<=", 2.0), ("<=", 2.5), ("<=", 3.0),
            ("<=", 3.5), ("<=", 4.0), ("<=", 5.0),
            (">=", 4.0), (">=", 5.0), (">=", 6.0), (">=", 8.0),
            (">=", 10.0), (">=", 13.0),
        ],
        "vol_ratio_5_15": [
            ("<=", 0.35), ("<=", 0.45), ("<=", 0.5), ("<=", 0.545),
            ("<=", 0.6), ("<=", 0.7), ("<=", 0.8), ("<=", 0.9),
            ("<=", 1.0),
            (">=", 1.1), (">=", 1.2), (">=", 1.3), (">=", 1.5),
            (">=", 1.64), (">=", 1.8), (">=", 2.0),
        ],
        "today_pct": [
            ("<=", 2.5), ("<=", 3.0), ("<=", 4.0), ("<=", 5.0),
            (">=", 5.0), (">=", 8.0), (">=", 10.0), (">=", 12.0),
            (">=", 15.0), (">=", 18.0), (">=", 20.0),
        ],
        "max_drop_7d": [
            ("<=", -3.0), ("<=", -5.0), ("<=", -7.0),
            ("<=", -10.0), ("<=", -12.0), ("<=", -15.0),
            (">=", -7.0), (">=", -5.0), (">=", -3.0),
        ],
        "gap_pct": [
            ("<=", -2.0), ("<=", -1.0), ("<=", -0.5), ("<=", 0.0),
            ("<=", 0.5), (">=", 1.0), (">=", 2.0), (">=", 3.5),
            (">=", 4.7),
        ],
        "pct_vs_lastweek": [
            ("<=", -10.0), ("<=", -5.0), ("<=", -3.0),
            ("<=", -1.0), ("<=", 0.0), (">=", 3.0), (">=", 5.0),
            (">=", 10.0),
        ],
        "dist_to_ma5": [
            ("<=", -2.0), ("<=", -1.0), ("<=", -0.5),
            ("<=", 0.0), (">=", 3.0), (">=", 5.0), (">=", 8.0),
            (">=", 10.0), (">=", 14.6684), (">=", 16.0),
        ],
        "ma5_chg_rate": [
            ("<=", -5.0), ("<=", -3.0), ("<=", -1.0),
            ("<=", 0.0), (">=", 1.0), (">=", 3.0), (">=", 5.0),
        ],
        "today_tr_val_eok": [
            ("<=", 5.0), ("<=", 10.0), ("<=", 20.0), ("<=", 50.0),
            ("<=", 100.0), ("<=", 300.0), ("<=", 500.0), ("<=", 1000.0),
            (">=", 3.0), (">=", 5.0), (">=", 10.0), (">=", 20.0),
        ],
        "BB_perc": [
            ("<=", 0.05), ("<=", 0.1), ("<=", 0.1765),
            ("<=", 0.25), ("<=", 0.3), ("<=", 0.5),
            (">=", 1.0), (">=", 1.07), (">=", 1.2),
        ],
        "lower_wick_ratio": [
            ("<=", 0.0), ("<=", 0.01), ("<=", 0.02), ("<=", 0.05),
            (">=", 0.1), (">=", 0.2),
        ],
        "upper_wick_ratio": [
            ("<=", 0.0), ("<=", 0.02), ("<=", 0.05),
            ("<=", 0.064), ("<=", 0.1), ("<=", 0.114),
            ("<=", 0.2), ("<=", 0.3),
            (">=", 0.05), (">=", 0.1), (">=", 0.2),
        ],
        "body_ratio": [
            ("<=", 0.3), ("<=", 0.5),
            (">=", 0.6), (">=", 0.7), (">=", 0.8),
            (">=", 0.87), (">=", 0.909), (">=", 0.981),
        ],
        "intraday_return": [
            ("<=", 0.0), ("<=", 1.0), ("<=", 1.5), ("<=", 2.0),
            ("<=", 2.7), ("<=", 3.0), ("<=", 4.0),
            (">=", 3.0), (">=", 4.0), (">=", 5.0), (">=", 6.0),
            (">=", 8.0), (">=", 10.0),
        ],
        "rebound_from_7d_low": [
            ("<=", 5.0), ("<=", 8.661), ("<=", 10.0), ("<=", 12.0),
            ("<=", 15.0),
            (">=", 20.0), (">=", 25.0), (">=", 30.0),
            (">=", 35.0), (">=", 40.0), (">=", 45.0), (">=", 50.0),
        ],
        "price_power_value": [
            ("<=", 3.0), ("<=", 5.0), ("<=", 10.0), ("<=", 20.0),
            (">=", 20.0), (">=", 40.0), (">=", 60.0),
            (">=", 80.0), (">=", 100.0),
        ],
        "body_value_power": [
            ("<=", 1.0), ("<=", 3.0), ("<=", 5.0), ("<=", 10.0),
            (">=", 5.0), (">=", 10.0), (">=", 15.0),
            (">=", 20.0), (">=", 30.0),
        ],
        "room_to_20d_high": [
            ("<=", 5.0), ("<=", 10.0), ("<=", 20.0), ("<=", 30.0),
            ("<=", 50.0),
            (">=", -10.0), (">=", -5.0), (">=", 0.0),
            (">=", 3.0), (">=", 5.0), (">=", 10.0),
        ],
        "room_to_60d_high": [
            ("<=", 5.0), ("<=", 10.0), ("<=", 20.0), ("<=", 30.0),
            ("<=", 50.0),
            (">=", -10.0), (">=", -5.0), (">=", 0.0),
            (">=", 3.0), (">=", 5.0), (">=", 10.0),
            (">=", 60.0), (">=", 65.0),
        ],
        "rebound_vs_prior_drop": [
            ("<=", 1.0), ("<=", 2.0), ("<=", 5.0), ("<=", 10.0),
            (">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 5.0),
        ],
        "market_today_pct": [
            ("<=", -2.0), ("<=", -1.0), ("<=", 0.0),
            (">=", -2.0), (">=", -1.0), (">=", 0.0),
            (">=", 0.5), (">=", 1.0), (">=", 2.0), (">=", 3.0),
        ],
        "market_5d_pct": [
            ("<=", -10.0), ("<=", -5.0), ("<=", -3.0),
            ("<=", -2.0), ("<=", -1.799),
            (">=", 0.0), (">=", 3.0), (">=", 5.0),
        ],
    }


def build_literals(
        df: pd.DataFrame,
        features: Iterable[str],
) -> tuple[list[tuple[str, str, float]], list[np.ndarray]]:
    literals: list[tuple[str, str, float]] = []
    masks: list[np.ndarray] = []

    extra = default_extra_thresholds()

    for feat in features:
        values = pd.to_numeric(df[feat], errors="coerce")
        arr = values.to_numpy()
        finite = np.isfinite(arr)

        raw_thresholds = []

        for th in _thresholds_for_series(values, N_QUANTILES):
            raw_thresholds.append(("<=", th))
            raw_thresholds.append((">=", th))

        for op, th in extra.get(feat, []):
            raw_thresholds.append((op, float(th)))

        unique = {}
        for op, th in raw_thresholds:
            unique[(op, round(float(th), 10))] = (op, float(th))

        for op, th in unique.values():
            literals.append((feat, op, th))

            if op == "<=":
                masks.append(finite & (arr <= th))
            elif op == ">=":
                masks.append(finite & (arr >= th))
            else:
                raise ValueError(op)

    return literals, masks


def make_mask_from_conds(
        df: pd.DataFrame,
        conds: list[tuple[str, str, float]],
) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)

    for feat, op, th in conds:
        arr = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        finite = np.isfinite(arr)

        if op == "<=":
            mask &= finite & (arr <= th)
        elif op == ">=":
            mask &= finite & (arr >= th)
        else:
            raise ValueError(op)

    return mask


def split_train_valid_by_date_ratio(
        df: pd.DataFrame,
        valid_ratio: float = VALID_RATIO,
        date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    if date_col not in df.columns:
        cut = int(len(df) * (1 - valid_ratio))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy(), None

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values(date_col).reset_index(drop=True)

    unique_dates = work[date_col].dropna().sort_values().unique()

    if len(unique_dates) < 2:
        cut = int(len(work) * (1 - valid_ratio))
        return work.iloc[:cut].copy(), work.iloc[cut:].copy(), None

    split_idx = max(1, int(len(unique_dates) * (1 - valid_ratio)))
    split_idx = min(split_idx, len(unique_dates) - 1)

    split_date = pd.Timestamp(unique_dates[split_idx])

    train = work[work[date_col] < split_date].copy()
    valid = work[work[date_col] >= split_date].copy()

    if len(train) == 0 or len(valid) == 0:
        cut = int(len(work) * (1 - valid_ratio))
        train = work.iloc[:cut].copy()
        valid = work.iloc[cut:].copy()
        split_date = None

    return train.reset_index(drop=True), valid.reset_index(drop=True), split_date


# =============================================================================
# Metrics
# =============================================================================

def wilson_lower_bound(success: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0

    p = success / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)

    return max(0.0, (center - margin) / denom)


def eval_mask(df: pd.DataFrame, mask: np.ndarray, label: str = "") -> dict:
    sub = df[mask]
    total = len(df)
    selected = len(sub)

    class0_total = int((df["target_class"] == TARGET_CLASS).sum())
    class0_cnt = int((sub["target_class"] == TARGET_CLASS).sum())

    base_rate = class0_total / total if total else 0.0
    class0_rate = class0_cnt / selected if selected else 0.0

    c = {
        cls: int((sub["target_class"] == cls).sum())
        for cls in [0, 1, 2, 3]
    }

    ct = {
        cls: int((df["target_class"] == cls).sum())
        for cls in [0, 1, 2, 3]
    }

    return {
        "label": label,
        "total_count": total,
        "selected_count": selected,
        "selected_rate": selected / total if total else 0.0,

        "class0_total": class0_total,
        "class0_count": class0_cnt,
        "base_class0_rate": base_rate,
        "class0_rate": class0_rate,
        "class0_lift": class0_rate / base_rate if base_rate else 0.0,
        "class0_coverage": class0_cnt / class0_total if class0_total else 0.0,
        "class0_wilson_low": wilson_lower_bound(class0_cnt, selected),

        "class1_count": c[1],
        "class2_count": c[2],
        "class3_count": c[3],

        "class1_rate": c[1] / selected if selected else 0.0,
        "class2_rate": c[2] / selected if selected else 0.0,
        "class3_rate": c[3] / selected if selected else 0.0,
        "class23_rate": (c[2] + c[3]) / selected if selected else 0.0,

        "class1_loss_rate": c[1] / ct[1] if ct[1] else 0.0,
        "class2_loss_rate": c[2] / ct[2] if ct[2] else 0.0,
        "class3_loss_rate": c[3] / ct[3] if ct[3] else 0.0,
    }


def print_eval(summary: dict):
    print(f"\n[{summary['label']}]")
    print(f"total_count          : {summary['total_count']}")
    print(f"selected_count       : {summary['selected_count']} ({summary['selected_rate'] * 100:.2f}%)")
    print(f"class0_count         : {summary['class0_count']} / {summary['class0_total']}")
    print(f"class0_rate          : {summary['class0_rate'] * 100:.2f}%")
    print(f"base_class0_rate     : {summary['base_class0_rate'] * 100:.2f}%")
    print(f"class0_lift          : {summary['class0_lift']:.3f}x")
    print(f"class0_coverage      : {summary['class0_coverage'] * 100:.2f}%")
    print(f"class0_wilson_low    : {summary['class0_wilson_low'] * 100:.2f}%")
    print(f"class1_rate          : {summary['class1_rate'] * 100:.2f}%")
    print(f"class2_rate          : {summary['class2_rate'] * 100:.2f}%")
    print(f"class3_rate          : {summary['class3_rate'] * 100:.2f}%")
    print(f"class23_rate         : {summary['class23_rate'] * 100:.2f}%")
    print(f"class1_loss_rate     : {summary['class1_loss_rate'] * 100:.2f}%")
    print(f"class2_loss_rate     : {summary['class2_loss_rate'] * 100:.2f}%")
    print(f"class3_loss_rate     : {summary['class3_loss_rate'] * 100:.2f}%")


# =============================================================================
# Rule mining
# =============================================================================

def _get_by_depth(values: list[float], depth: int) -> float:
    return values[min(depth, len(values) - 1)]


def _rule_score(
        cnt: int,
        class0_cnt: int,
        class0_rate: float,
        base_rate: float,
        c1_rate: float,
        c2_rate: float,
        c3_rate: float,
) -> float:
    if cnt <= 0 or base_rate <= 0:
        return 0.0

    lift = class0_rate / base_rate
    wilson = wilson_lower_bound(class0_cnt, cnt)

    penalty = max(
        0.01,
        1.0
        - CLASS1_PENALTY * c1_rate
        - CLASS2_PENALTY * c2_rate
        - CLASS3_PENALTY * c3_rate,
        )

    return (
            (class0_rate ** PRECISION_POWER)
            * (lift ** LIFT_POWER)
            * (max(wilson, 1e-9) ** WILSON_POWER)
            * (np.log1p(class0_cnt) ** COVERAGE_POWER)
            * np.log1p(cnt)
            * penalty
    )


def mine_class0_rules(
        df: pd.DataFrame,
        literals: list[tuple[str, str, float]],
        literal_masks: list[np.ndarray],
        class0: np.ndarray,
        min_count: int = MIN_CNT,
        max_depth: int = MAX_DEPTH,
        beam: int = BEAM,
        top_n: int | None = TOP_N,
        feature_groups: dict[str, str] | None = None,
        group_limits: dict[str, int] | None = None,
):
    beams = [(np.ones(len(df), dtype=bool), [])]
    good = {}

    base_rate = float(class0.mean())
    y = df["target_class"].to_numpy()

    feature_groups = feature_groups or {}
    group_limits = group_limits or {}

    print(
        "\n[CLASS0 HIGH-PROB RULE MINING]",
        "\ntarget_class:", TARGET_CLASS,
        "\nbase_class0_rate:", round(base_rate, 4),
        "\nbeam:", beam,
        "\ntop_n:", top_n,
        "\nmin_count:", min_count,
        "\nmax_depth:", max_depth,
        "\nmin_class0_rate:", MIN_CLASS0_RATE,
        "\nmin_lift:", MIN_LIFT,
        "\nmin_wilson_low:", MIN_WILSON_LOW,
        "\n",
    )

    def get_group(feat_name: str) -> str | None:
        return feature_groups.get(feat_name)

    for depth in range(max_depth):
        print("----------------------------------")
        print("[CLASS0] depth", depth)

        heap = []
        uid = count()

        expand_min_rate = _get_by_depth(EXPAND_MIN_CLASS0_RATE, depth)
        expand_min_lift = _get_by_depth(EXPAND_MIN_LIFT, depth)

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

            group_used: dict[str, int] = {}
            for f in used_feats:
                g = get_group(f)
                if g is not None:
                    group_used[g] = group_used.get(g, 0) + 1

            for lit, lmask in zip(literals, literal_masks):
                feat = lit[0]

                if feat in used_feats:
                    continue

                if USE_FEATURE_GROUP_LIMITS:
                    g = get_group(feat)
                    if g is not None:
                        limit = group_limits.get(g)
                        if limit is not None and group_used.get(g, 0) >= limit:
                            continue

                m = base_mask & lmask
                cnt = int(m.sum())

                if cnt < min_count:
                    continue

                class0_cnt = int((m & class0).sum())
                class0_rate = class0_cnt / cnt
                lift = class0_rate / base_rate if base_rate else 0.0
                wilson = wilson_lower_bound(class0_cnt, cnt)

                c1_rate = int(((y == 1) & m).sum()) / cnt
                c2_rate = int(((y == 2) & m).sum()) / cnt
                c3_rate = int(((y == 3) & m).sum()) / cnt
                c23_rate = c2_rate + c3_rate

                score = _rule_score(
                    cnt=cnt,
                    class0_cnt=class0_cnt,
                    class0_rate=class0_rate,
                    base_rate=base_rate,
                    c1_rate=c1_rate,
                    c2_rate=c2_rate,
                    c3_rate=c3_rate,
                )

                if (
                        class0_rate >= MIN_CLASS0_RATE
                        and lift >= MIN_LIFT
                        and wilson >= MIN_WILSON_LOW
                        and c23_rate <= 0.70
                ):
                    key2 = tuple(
                        sorted(
                            (c[0], c[1], round(float(c[2]), 6))
                            for c in (conds + [lit])
                        )
                    )

                    prev = good.get(key2)
                    if prev is None or score > prev[9]:
                        good[key2] = (
                            cnt,
                            class0_cnt,
                            class0_rate,
                            lift,
                            wilson,
                            c1_rate,
                            c2_rate,
                            c3_rate,
                            conds + [lit],
                            score,
                        )

                if class0_rate >= expand_min_rate and lift >= expand_min_lift:
                    k = (score, wilson, class0_rate, lift, class0_cnt, cnt)
                    item = (k, next(uid), m, conds + [lit], class0_rate, lift, wilson, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    elif k > heap[0][0]:
                        heapq.heapreplace(heap, item)

        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("[CLASS0] new", len(new))

        if not new:
            print("[CLASS0] no expandable candidates; stopping.")
            break

        tail = new[-1]

        print(
            "[CLASS0] tail rate:", round(tail[4], 3),
            "lift:", round(tail[5], 2),
            "wilson:", round(tail[6], 3),
            "cnt:", tail[7],
            "conds:", tail[3],
        )

        beams = [(m, conds) for _, _, m, conds, _, _, _, _ in new]

    def out_key(x):
        (
            cnt,
            class0_cnt,
            class0_rate,
            lift,
            wilson,
            c1_rate,
            c2_rate,
            c3_rate,
            conds,
            score,
        ) = x

        return (
            -score,
            -wilson,
            -class0_rate,
            -lift,
            c3_rate,
            c2_rate,
            -class0_cnt,
            -cnt,
        )

    out = sorted(good.values(), key=out_key)

    if top_n is not None:
        out = out[:top_n]

    return out


# =============================================================================
# Scenario config
# =============================================================================

def get_scenarios() -> list[dict]:
    return [
        {
            "name": "cov4_c3_05_safe",
            "alias": "cov4_c3_05",
            "target_coverage": 0.040,
            "min_acceptable_coverage": 0.035,
            "soft_coverage": 0.030,

            "max_class23": 0.150,
            "max_class3": 0.050,
            "max_class2": 0.120,

            "min_rate_pre": 0.62,
            "min_rate_post": 0.52,

            "min_added_rate": 0.28,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 0.35,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 0.14,

            "coverage_w": 2500.0,
            "class0_w": 8.0,
            "c23_penalty": 380.0,
            "c3_penalty": 520.0,
            "added_c23_penalty": 22.0,
            "diagnostic": False,
        },
        {
            "name": "cov4_c3_06_balanced",
            "alias": "cov4_c3_06",
            "target_coverage": 0.040,
            "min_acceptable_coverage": 0.038,
            "soft_coverage": 0.032,

            "max_class23": 0.160,
            "max_class3": 0.060,
            "max_class2": 0.130,

            "min_rate_pre": 0.58,
            "min_rate_post": 0.48,

            "min_added_rate": 0.22,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 0.42,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 0.18,

            "coverage_w": 3200.0,
            "class0_w": 6.5,
            "c23_penalty": 330.0,
            "c3_penalty": 440.0,
            "added_c23_penalty": 18.0,
            "diagnostic": False,
        },
        {
            "name": "cov5_c23_10_strict",
            "alias": "c23_10",
            "target_coverage": 0.050,
            "min_acceptable_coverage": 0.050,
            "soft_coverage": 0.040,

            "max_class23": 0.100,
            "max_class3": 0.050,
            "max_class2": 0.090,

            "min_rate_pre": 0.66,
            "min_rate_post": 0.56,

            "min_added_rate": 0.36,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 0.18,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 0.10,

            "coverage_w": 2200.0,
            "class0_w": 8.0,
            "c23_penalty": 520.0,
            "c3_penalty": 620.0,
            "added_c23_penalty": 34.0,
            "diagnostic": False,
        },
        {
            "name": "cov5_c23_12_balanced",
            "alias": "c23_12",
            "target_coverage": 0.050,
            "min_acceptable_coverage": 0.050,
            "soft_coverage": 0.040,

            "max_class23": 0.120,
            "max_class3": 0.065,
            "max_class2": 0.110,

            "min_rate_pre": 0.60,
            "min_rate_post": 0.50,

            "min_added_rate": 0.28,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 0.24,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 0.14,

            "coverage_w": 2800.0,
            "class0_w": 7.0,
            "c23_penalty": 430.0,
            "c3_penalty": 540.0,
            "added_c23_penalty": 28.0,
            "diagnostic": False,
        },
        {
            "name": "cov5_c23_15_reach",
            "alias": "c23_15",
            "target_coverage": 0.050,
            "min_acceptable_coverage": 0.050,
            "soft_coverage": 0.040,

            "max_class23": 0.150,
            "max_class3": 0.080,
            "max_class2": 0.140,

            "min_rate_pre": 0.54,
            "min_rate_post": 0.44,

            "min_added_rate": 0.18,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 0.40,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 0.20,

            "coverage_w": 3800.0,
            "class0_w": 5.5,
            "c23_penalty": 300.0,
            "c3_penalty": 390.0,
            "added_c23_penalty": 18.0,
            "diagnostic": False,
        },
        {
            "name": "cov5_c23_18_last_resort",
            "alias": "c23_18",
            "target_coverage": 0.050,
            "min_acceptable_coverage": 0.050,
            "soft_coverage": 0.038,

            "max_class23": 0.180,
            "max_class3": 0.110,
            "max_class2": 0.170,

            "min_rate_pre": 0.48,
            "min_rate_post": 0.38,

            "min_added_rate": 0.08,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 1.00,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 1.00,

            "coverage_w": 5200.0,
            "class0_w": 4.0,
            "c23_penalty": 210.0,
            "c3_penalty": 280.0,
            "added_c23_penalty": 10.0,
            "diagnostic": False,
        },
        {
            "name": "cov5_c23_22_diagnostic",
            "alias": "c23_22",
            "target_coverage": 0.050,
            "min_acceptable_coverage": 0.050,
            "soft_coverage": 0.038,

            "max_class23": 0.220,
            "max_class3": 0.150,
            "max_class2": 0.210,

            "min_rate_pre": 0.42,
            "min_rate_post": 0.34,

            "min_added_rate": 0.00,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 1.00,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 1.00,

            "coverage_w": 7000.0,
            "class0_w": 3.0,
            "c23_penalty": 150.0,
            "c3_penalty": 210.0,
            "added_c23_penalty": 5.0,
            "diagnostic": True,
        },
        {
            "name": "cov5_c23_25_diagnostic",
            "alias": "c23_25",
            "target_coverage": 0.050,
            "min_acceptable_coverage": 0.050,
            "soft_coverage": 0.036,

            "max_class23": 0.250,
            "max_class3": 0.180,
            "max_class2": 0.240,

            "min_rate_pre": 0.38,
            "min_rate_post": 0.30,

            "min_added_rate": 0.00,
            "min_added_rate_fill": 0.00,

            "max_added_c23": 1.00,
            "max_added_c23_fill": 1.00,
            "max_added_c3": 1.00,

            "coverage_w": 9000.0,
            "class0_w": 2.5,
            "c23_penalty": 90.0,
            "c3_penalty": 140.0,
            "added_c23_penalty": 2.0,
            "diagnostic": True,
        },
    ]


# =============================================================================
# Validation / selection
# =============================================================================

def pass_valid_filter(ev: dict, tier: dict) -> bool:
    return (
            ev["selected_count"] >= tier["min_cnt"]
            and ev["class0_rate"] >= tier["min_class0_rate"]
            and ev["class0_lift"] >= tier["min_lift"]
            and ev["class0_wilson_low"] >= tier["min_wilson_low"]
            and ev["class1_rate"] <= tier["max_class1_rate"]
            and ev["class2_rate"] <= tier["max_class2_rate"]
            and ev["class3_rate"] <= tier["max_class3_rate"]
            and ev["class23_rate"] <= tier["max_class23_rate"]
    )


# =============================================================================
# Multiprocessing evaluation
# =============================================================================

_MP_TRAIN = None
_MP_VALID = None


def _init_eval_worker(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    global _MP_TRAIN
    global _MP_VALID

    _MP_TRAIN = train_df
    _MP_VALID = valid_df

    print(f"[WORKER INIT] pid={os.getpid()}")


def _evaluate_one_rule_worker(args):
    (
        i,
        cnt,
        class0_cnt,
        class0_rate,
        lift,
        wilson,
        c1_rate,
        c2_rate,
        c3_rate,
        conds,
        score,
    ) = args

    train = _MP_TRAIN
    valid = _MP_VALID

    name = (
        f"target0_{i:05d}"
        f"_s{score:.2f}"
        f"_tr{class0_rate:.3f}"
        f"_wl{wilson:.3f}"
        f"_n{cnt}"
    )

    train_mask = make_mask_from_conds(train, conds)
    valid_mask = make_mask_from_conds(valid, conds)

    tr = eval_mask(train, train_mask, label=name)
    va = eval_mask(valid, valid_mask, label=name)

    if tr["selected_count"] < MIN_CNT:
        return None

    if (
            tr["class0_rate"] < MIN_CLASS0_RATE
            or tr["class0_lift"] < MIN_LIFT
            or tr["class0_wilson_low"] < MIN_WILSON_LOW
    ):
        return None

    stability_gap = abs(tr["class0_rate"] - va["class0_rate"])

    stable_score = (
            score
            * (1.0 - min(stability_gap, 0.70))
            * max(0.02, 1.0 - va["class23_rate"] * 1.20 - va["class3_rate"] * 1.80)
    )

    return {
        "name": name,
        "conds": conds,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "score": score,
        "stable_score": stable_score,
        "train_eval": tr,
        "valid_eval": va,
    }


def _evaluate_rule_chunk_worker(rule_args_chunk):
    out = []

    for arg in rule_args_chunk:
        result = _evaluate_one_rule_worker(arg)

        if result is not None:
            out.append(result)

    return out


def evaluate_train_pass_rules(
        train: pd.DataFrame,
        valid: pd.DataFrame,
        rules,
        workers: int = 1,
):
    rule_args = []

    for i, (
            cnt,
            class0_cnt,
            class0_rate,
            lift,
            wilson,
            c1_rate,
            c2_rate,
            c3_rate,
            conds,
            score,
    ) in enumerate(rules, start=1):
        rule_args.append(
            (
                i,
                cnt,
                class0_cnt,
                class0_rate,
                lift,
                wilson,
                c1_rate,
                c2_rate,
                c3_rate,
                conds,
                score,
            )
        )

    if workers is None:
        workers = 1

    workers = max(1, int(workers))

    if workers <= 1:
        global _MP_TRAIN
        global _MP_VALID

        _MP_TRAIN = train
        _MP_VALID = valid

        out = []

        for idx, arg in enumerate(rule_args, start=1):
            result = _evaluate_one_rule_worker(arg)

            if result is not None:
                out.append(result)

            if idx % 1000 == 0:
                print(f"[EVAL SERIAL] {idx}/{len(rule_args)} done, passed={len(out)}")

        out.sort(key=lambda x: x["name"])
        return out

    print(f"[EVAL MP CHUNKED] workers={workers}, rules={len(rule_args)}")

    chunk_size = max(100, math.ceil(len(rule_args) / (workers * 8)))

    chunks = [
        rule_args[i:i + chunk_size]
        for i in range(0, len(rule_args), chunk_size)
    ]

    print(
        f"[EVAL MP CHUNKED] chunks={len(chunks)}, "
        f"chunk_size={chunk_size}"
    )

    all_evaluated = []
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_eval_worker,
            initargs=(train, valid),
    ) as executor:
        futures = [
            executor.submit(_evaluate_rule_chunk_worker, chunk)
            for chunk in chunks
        ]

        done_chunks = 0
        done_rules = 0

        for fut in as_completed(futures):
            done_chunks += 1

            try:
                chunk_result = fut.result()
            except Exception as e:
                print(f"[WARN][EVAL MP CHUNKED] worker failed: {e}")
                continue

            all_evaluated.extend(chunk_result)

            done_rules = min(done_rules + chunk_size, len(rule_args))

            print(
                f"[EVAL MP CHUNKED] chunk {done_chunks}/{len(chunks)} done, "
                f"approx_rules={done_rules}/{len(rule_args)}, "
                f"passed={len(all_evaluated)}"
            )

    all_evaluated.sort(key=lambda x: x["name"])

    return all_evaluated


def build_candidates_by_valid_tiers(all_evaluated):
    candidates_by_name = {}

    for tier_idx, tier in enumerate(VALID_FILTER_TIERS):
        tier_candidates = []

        for c in all_evaluated:
            if pass_valid_filter(c["valid_eval"], tier):
                cc = dict(c)
                cc["valid_tier"] = tier["name"]
                cc["valid_tier_idx"] = tier_idx

                tier_candidates.append(cc)

                old = candidates_by_name.get(cc["name"])
                if old is None or tier_idx < old["valid_tier_idx"]:
                    candidates_by_name[cc["name"]] = cc

        print(f"[VALID FILTER:{tier['name']}] passed={len(tier_candidates)} / {len(all_evaluated)}")

    candidates = list(candidates_by_name.values())

    print("[CANDIDATES AFTER MERGED VALID FILTER]", len(candidates))

    return candidates


def scenario_result_score(va: dict, scenario: dict) -> float:
    coverage = va["class0_coverage"]
    class23 = va["class23_rate"]
    class3 = va["class3_rate"]
    rate = va["class0_rate"]

    min_cov = scenario["min_acceptable_coverage"]
    target_cov = scenario["target_coverage"]

    shortfall = max(0.0, min_cov - coverage)
    target_gap = abs(target_cov - coverage)

    score = (
            coverage * 2500000.0
            + rate * 40000.0
            - class23 * 350000.0
            - class3 * 550000.0
            - shortfall * 4500000.0
            - target_gap * 400000.0
            + np.log1p(va["selected_count"]) * 1000.0
    )

    if coverage >= min_cov:
        score += 650000.0

    if coverage >= target_cov:
        score += 250000.0

    if class23 <= 0.10:
        score += 140000.0
    elif class23 <= 0.12:
        score += 90000.0
    elif class23 <= 0.15:
        score += 40000.0

    if class3 <= 0.05:
        score += 160000.0
    elif class3 <= 0.06:
        score += 100000.0
    elif class3 <= 0.08:
        score += 30000.0

    if scenario.get("diagnostic", False):
        score -= 180000.0

    return score


def run_scenario(
        scenario: dict,
        candidates: list[dict],
        train: pd.DataFrame,
        valid: pd.DataFrame,
        max_rules: int | None = MAX_RULES,
        verbose: bool = True,
):
    train_class0 = train["target_class"].to_numpy() == TARGET_CLASS

    valid_y = valid["target_class"].to_numpy()
    valid_class0 = valid_y == TARGET_CLASS
    valid_class2 = valid_y == 2
    valid_class3 = valid_y == 3

    valid_class0_total = int(valid_class0.sum())

    selected: list[tuple[str, list[tuple[str, str, float]]]] = []
    selected_names = set()

    train_mask = np.zeros(len(train), dtype=bool)
    valid_mask = np.zeros(len(valid), dtype=bool)

    best_checkpoint = None

    def checkpoint_score(ev: dict) -> float:
        coverage = ev["class0_coverage"]
        class23 = ev["class23_rate"]
        class3 = ev["class3_rate"]

        shortfall = max(0.0, scenario["min_acceptable_coverage"] - coverage)

        score = (
                coverage * 3600000.0
                + ev["class0_rate"] * 20000.0
                - class23 * scenario["c23_penalty"] * 1000.0
                - class3 * scenario["c3_penalty"] * 1000.0
                - shortfall * 5200000.0
                + np.log1p(ev["selected_count"]) * 1300.0
        )

        if coverage >= scenario["min_acceptable_coverage"]:
            score += 900000.0

        return score

    def maybe_update_checkpoint():
        nonlocal best_checkpoint

        ev = eval_mask(valid, valid_mask, label="VALID")

        if (
                ev["selected_count"] >= 10
                and ev["class23_rate"] <= scenario["max_class23"]
                and ev["class3_rate"] <= scenario["max_class3"]
                and ev["class2_rate"] <= scenario["max_class2"]
                and ev["class0_rate"] >= max(0.30, scenario["min_rate_post"] - 0.12)
        ):
            sc = checkpoint_score(ev)

            if best_checkpoint is None or sc > best_checkpoint["score"]:
                best_checkpoint = {
                    "score": sc,
                    "selected": list(selected),
                    "train_mask": train_mask.copy(),
                    "valid_mask": valid_mask.copy(),
                    "train_eval": eval_mask(train, train_mask, label="TRAIN"),
                    "valid_eval": ev,
                }

    def choose_best(stage: str):
        cur = eval_mask(valid, valid_mask, label="VALID")

        cur_target = cur["class0_count"]
        cur_selected = cur["selected_count"]

        best = None

        for cand in candidates:
            if cand["name"] in selected_names:
                continue

            new_train = cand["train_mask"] & ~train_mask
            new_valid = cand["valid_mask"] & ~valid_mask

            vt = int(new_valid.sum())
            tt = int(new_train.sum())

            if vt < (3 if stage == "primary" else 1):
                continue

            va0 = int((new_valid & valid_class0).sum())
            va2 = int((new_valid & valid_class2).sum())
            va3 = int((new_valid & valid_class3).sum())
            va23 = va2 + va3

            tr0 = int((new_train & train_class0).sum())

            if va0 < (2 if stage == "primary" else 1):
                continue

            valid_added_rate = va0 / vt if vt else 0.0
            train_added_rate = tr0 / tt if tt else 0.0
            added_c23_rate = va23 / vt if vt else 0.0
            added_c3_rate = va3 / vt if vt else 0.0
            added_wilson = wilson_lower_bound(va0, vt)

            min_added = scenario["min_added_rate"] if stage == "primary" else scenario["min_added_rate_fill"]
            max_added_c23 = scenario["max_added_c23"] if stage == "primary" else scenario["max_added_c23_fill"]

            if valid_added_rate < min_added:
                continue

            if train_added_rate < max(0.08, min_added - 0.24):
                continue

            if added_c23_rate > max_added_c23:
                continue

            if added_c3_rate > scenario["max_added_c3"]:
                continue

            next_mask = valid_mask | new_valid

            next_selected = cur_selected + vt
            next_target = cur_target + va0

            next_rate = next_target / next_selected if next_selected else 0.0
            next_cov = next_target / valid_class0_total if valid_class0_total else 0.0

            next_c2 = int((next_mask & valid_class2).sum())
            next_c3 = int((next_mask & valid_class3).sum())
            next_c23 = next_c2 + next_c3

            next_c2_rate = next_c2 / next_selected if next_selected else 0.0
            next_c3_rate = next_c3 / next_selected if next_selected else 0.0
            next_c23_rate = next_c23 / next_selected if next_selected else 0.0

            min_rate = (
                scenario["min_rate_pre"]
                if next_cov < scenario["soft_coverage"]
                else scenario["min_rate_post"]
            )

            if next_rate < min_rate:
                continue

            if next_c23_rate > scenario["max_class23"]:
                continue

            if next_c3_rate > scenario["max_class3"]:
                continue

            if next_c2_rate > scenario["max_class2"]:
                continue

            false_add = vt - va0
            shortfall = max(0.0, scenario["min_acceptable_coverage"] - next_cov)

            reach_bonus = 0.0
            if next_cov >= scenario["min_acceptable_coverage"]:
                reach_bonus += 1400.0
            elif next_cov >= scenario["soft_coverage"]:
                reach_bonus += 350.0

            low_class23_bonus = max(0.0, scenario["max_class23"] - next_c23_rate) * 1800.0
            low_class3_bonus = max(0.0, scenario["max_class3"] - next_c3_rate) * 2600.0

            score = (
                    va0 * (scenario["class0_w"] if stage == "primary" else scenario["class0_w"] + 5.0)
                    + vt * (1.0 if stage == "primary" else 1.7)
                    + valid_added_rate * (10.0 if stage == "primary" else 4.0)
                    + added_wilson * 3.0
                    + next_cov * scenario["coverage_w"]
                    + reach_bonus
                    + low_class23_bonus
                    + low_class3_bonus
                    + tr0 * 0.10
                    - false_add * (1.0 if stage == "primary" else 0.50)
                    - va23 * scenario["added_c23_penalty"]
                    - va3 * (scenario["added_c23_penalty"] + 5.0)
                    - next_c23_rate * scenario["c23_penalty"]
                    - next_c3_rate * scenario["c3_penalty"]
                    - next_c2_rate * (scenario["c23_penalty"] * 0.60)
                    - shortfall * 1800.0
                    + cand["stable_score"] * 0.004
            )

            key = (
                next_cov >= scenario["min_acceptable_coverage"],
                score,
                -shortfall,
                -next_c3_rate,
                -next_c23_rate,
                va0,
                next_cov,
                next_rate,
                -va23,
                cand["stable_score"],
            )

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "score": score,
                    "cand": cand,
                    "stage": stage,
                    "vt": vt,
                    "va0": va0,
                    "va23": va23,
                    "added_c23_rate": added_c23_rate,
                    "valid_added_rate": valid_added_rate,
                    "tr0": tr0,
                    "tt": tt,
                    "train_added_rate": train_added_rate,
                    "next_rate": next_rate,
                    "next_cov": next_cov,
                    "next_c23_rate": next_c23_rate,
                    "next_c3_rate": next_c3_rate,
                }

        return best

    def add_candidate(best, stage: str):
        nonlocal train_mask
        nonlocal valid_mask

        cand = best["cand"]

        train_mask |= cand["train_mask"]
        valid_mask |= cand["valid_mask"]

        selected.append((cand["name"], cand["conds"]))
        selected_names.add(cand["name"])

        maybe_update_checkpoint()

        if verbose:
            ev = eval_mask(valid, valid_mask, label="VALID")

            print(
                f"[SELECT:{scenario['name']}:{stage.upper()}]",
                f"{len(selected):03d}",
                cand["name"],
                f"score={best['score']:.2f}",
                f"valid_add={best['va0']}/{best['vt']} ({best['valid_added_rate'] * 100:.2f}%)",
                f"valid_add_c23={best['va23']}/{best['vt']} ({best['added_c23_rate'] * 100:.2f}%)",
                f"combined_rate={ev['class0_rate'] * 100:.2f}%",
                f"coverage={ev['class0_coverage'] * 100:.2f}%",
                f"target={scenario['target_coverage'] * 100:.1f}%",
                f"class23={ev['class23_rate'] * 100:.2f}%",
                f"class3={ev['class3_rate'] * 100:.2f}%",
            )

    for stage in ["primary", "fill"]:
        while True:
            if max_rules is not None and len(selected) >= max_rules:
                break

            cur = eval_mask(valid, valid_mask, label="VALID")

            if stage == "fill":
                if (
                        cur["class0_coverage"] >= scenario["target_coverage"]
                        and cur["class23_rate"] <= scenario["max_class23"]
                        and cur["class3_rate"] <= scenario["max_class3"]
                ):
                    if verbose:
                        print(f"[{scenario['name']}] [FILL STOP] target achieved.")
                    break

            best = choose_best(stage)

            if best is None:
                if verbose:
                    print(f"[{scenario['name']}] [{stage.upper()} STOP] no acceptable rule.")
                break

            add_candidate(best, stage)

    if verbose:
        print(f"[{scenario['name']}] [REPAIR START]")

    repair_round = 0

    while True:
        if max_rules is not None and len(selected) >= max_rules:
            break

        cur = eval_mask(valid, valid_mask, label="VALID")

        if (
                cur["class0_coverage"] >= scenario["target_coverage"]
                and cur["class23_rate"] <= scenario["max_class23"]
                and cur["class3_rate"] <= scenario["max_class3"]
        ):
            if verbose:
                print(f"[{scenario['name']}] [REPAIR STOP] target achieved.")
            break

        best_repair = None

        for cand in candidates:
            if cand["name"] in selected_names:
                continue

            new_train = cand["train_mask"] & ~train_mask
            new_valid = cand["valid_mask"] & ~valid_mask

            vt = int(new_valid.sum())
            if vt <= 0:
                continue

            va0 = int((new_valid & valid_class0).sum())
            va2 = int((new_valid & valid_class2).sum())
            va3 = int((new_valid & valid_class3).sum())
            va23 = va2 + va3

            if va0 <= 0:
                continue

            next_mask = valid_mask | new_valid
            next_selected = int(next_mask.sum())

            next_c0 = int((next_mask & valid_class0).sum())
            next_c2 = int((next_mask & valid_class2).sum())
            next_c3 = int((next_mask & valid_class3).sum())
            next_c23 = next_c2 + next_c3

            next_cov = next_c0 / valid_class0_total if valid_class0_total else 0.0
            next_rate = next_c0 / next_selected if next_selected else 0.0
            next_c2_rate = next_c2 / next_selected if next_selected else 0.0
            next_c3_rate = next_c3 / next_selected if next_selected else 0.0
            next_c23_rate = next_c23 / next_selected if next_selected else 0.0

            if next_c23_rate > scenario["max_class23"]:
                continue

            if next_c3_rate > scenario["max_class3"]:
                continue

            if next_c2_rate > scenario["max_class2"]:
                continue

            valid_added_rate = va0 / vt
            class23_cost = va23 + va3 * 1.8
            shortfall = max(0.0, scenario["target_coverage"] - next_cov)

            repair_score = (
                    va0 * 150.0
                    + valid_added_rate * 22.0
                    + next_cov * scenario["coverage_w"]
                    - vt * 0.16
                    - class23_cost * 130.0
                    - next_c23_rate * scenario["c23_penalty"] * 4.0
                    - next_c3_rate * scenario["c3_penalty"] * 4.5
                    - shortfall * 1200.0
                    + cand["stable_score"] * 0.001
            )

            key = (
                next_cov >= scenario["min_acceptable_coverage"],
                repair_score,
                va0,
                -va23,
                -next_c3_rate,
                -next_c23_rate,
                next_cov,
                next_rate,
            )

            if best_repair is None or key > best_repair["key"]:
                best_repair = {
                    "key": key,
                    "score": repair_score,
                    "cand": cand,
                    "stage": "repair",
                    "vt": vt,
                    "va0": va0,
                    "va23": va23,
                    "added_c23_rate": va23 / vt if vt else 0.0,
                    "valid_added_rate": valid_added_rate,
                    "tr0": int((new_train & train_class0).sum()),
                    "tt": int(new_train.sum()),
                    "train_added_rate": 0.0,
                    "next_rate": next_rate,
                    "next_cov": next_cov,
                    "next_c23_rate": next_c23_rate,
                    "next_c3_rate": next_c3_rate,
                }

        if best_repair is None:
            if verbose:
                print(f"[{scenario['name']}] [REPAIR STOP] no cap-safe candidate.")
            break

        repair_round += 1
        add_candidate(best_repair, "repair")

        if repair_round >= 150:
            if verbose:
                print(f"[{scenario['name']}] [REPAIR STOP] max repair rounds.")
            break

    tr_eval = eval_mask(train, train_mask, label="TRAIN")
    va_eval = eval_mask(valid, valid_mask, label="VALID")

    final_result = {
        "scenario": scenario,
        "selected": selected,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "train_eval": tr_eval,
        "valid_eval": va_eval,
        "score": scenario_result_score(va_eval, scenario),
    }

    if best_checkpoint is not None:
        b = best_checkpoint["valid_eval"]

        final_safe = (
                va_eval["class23_rate"] <= scenario["max_class23"]
                and va_eval["class3_rate"] <= scenario["max_class3"]
                and va_eval["class2_rate"] <= scenario["max_class2"]
                and va_eval["class0_coverage"] >= b["class0_coverage"]
        )

        if final_safe:
            return final_result

        if verbose:
            print(
                f"[{scenario['name']}] [CHECKPOINT BEST] "
                f"selected={b['selected_count']} "
                f"class0_rate={b['class0_rate'] * 100:.2f}% "
                f"coverage={b['class0_coverage'] * 100:.2f}% "
                f"target={scenario['target_coverage'] * 100:.1f}% "
                f"class23={b['class23_rate'] * 100:.2f}% "
                f"class3={b['class3_rate'] * 100:.2f}% "
                f"checkpoint_score={best_checkpoint['score']:.2f}"
            )

        return {
            "scenario": scenario,
            "selected": best_checkpoint["selected"],
            "train_mask": best_checkpoint["train_mask"],
            "valid_mask": best_checkpoint["valid_mask"],
            "train_eval": best_checkpoint["train_eval"],
            "valid_eval": best_checkpoint["valid_eval"],
            "score": scenario_result_score(best_checkpoint["valid_eval"], scenario),
        }

    return final_result


def is_target_met(result: dict) -> bool:
    va = result["valid_eval"]
    sc = result["scenario"]

    return (
            va["class0_coverage"] >= sc["min_acceptable_coverage"]
            and va["class23_rate"] <= sc["max_class23"]
            and va["class3_rate"] <= sc["max_class3"]
            and va["class2_rate"] <= sc["max_class2"]
    )


def is_practical_met(result: dict) -> bool:
    va = result["valid_eval"]
    sc = result["scenario"]

    return (
            not sc.get("diagnostic", False)
            and va["class0_coverage"] >= 0.035
            and va["class23_rate"] <= 0.16
            and va["class3_rate"] <= 0.06
    )


def select_best_result(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No scenario results.")

    # 1. 진짜 5% 목표 달성 결과가 있으면 그중 class23/class3 낮은 것 선택
    met = [r for r in results if is_target_met(r)]

    if met:
        return sorted(
            met,
            key=lambda r: (
                r["scenario"].get("diagnostic", False),
                r["valid_eval"]["class23_rate"],
                r["valid_eval"]["class3_rate"],
                -r["valid_eval"]["class0_coverage"],
                -r["valid_eval"]["class0_rate"],
                -r["score"],
            ),
        )[0]

    # 2. 5% 미달이면 실전형 우선: class3 낮고 coverage 3.5% 이상인 것
    practical = [r for r in results if is_practical_met(r)]

    if practical:
        return sorted(
            practical,
            key=lambda r: (
                r["valid_eval"]["class3_rate"],
                r["valid_eval"]["class23_rate"],
                -r["valid_eval"]["class0_coverage"],
                -r["valid_eval"]["class0_rate"],
                -r["score"],
            ),
        )[0]

    # 3. 그래도 없으면 diagnostic 제외하고 class3 손실 대비 coverage가 좋은 것
    non_diag = [r for r in results if not r["scenario"].get("diagnostic", False)]

    if non_diag:
        return sorted(
            non_diag,
            key=lambda r: (
                r["valid_eval"]["class3_rate"] > 0.08,
                -r["valid_eval"]["class0_coverage"],
                r["valid_eval"]["class3_rate"],
                r["valid_eval"]["class23_rate"],
                -r["valid_eval"]["class0_rate"],
            ),
        )[0]

    # 4. 마지막으로 전체 중 coverage 최대
    return sorted(
        results,
        key=lambda r: (
            -r["valid_eval"]["class0_coverage"],
            r["valid_eval"]["class23_rate"],
            r["valid_eval"]["class3_rate"],
        ),
    )[0]


def select_rules_with_validation(
        train: pd.DataFrame,
        valid: pd.DataFrame,
        rules,
        mode: str = "all",
        max_rules: int | None = MAX_RULES,
        workers: int = 1,
):
    all_evaluated = evaluate_train_pass_rules(
        train=train,
        valid=valid,
        rules=rules,
        workers=workers,
    )

    print("\n[EVALUATED TRAIN-PASS RULES]", len(all_evaluated))

    candidates = build_candidates_by_valid_tiers(all_evaluated)

    if not candidates:
        print("[STOP] no candidates.")
        empty_train = np.zeros(len(train), dtype=bool)
        empty_valid = np.zeros(len(valid), dtype=bool)
        return [], empty_train, empty_valid, [], None

    valid_class0_total = int((valid["target_class"].to_numpy() == TARGET_CLASS).sum())

    print(
        "[COVERAGE TARGET]",
        f"valid_class0_total={valid_class0_total}",
        f"4pct={int(np.ceil(valid_class0_total * PRACTICAL_COVERAGE_4))}",
        f"5pct={int(np.ceil(valid_class0_total * TARGET_COVERAGE_5))}",
    )

    scenarios = get_scenarios()

    if mode not in ["auto", "all"]:
        scenarios = [
            s for s in scenarios
            if s["alias"] == mode or s["name"] == mode
        ]

        if not scenarios:
            raise ValueError(f"unknown mode: {mode}")

    results = []

    for scenario in scenarios:
        print(f"\n[SCENARIO RUN] {scenario['name']}")

        result = run_scenario(
            scenario=scenario,
            candidates=candidates,
            train=train,
            valid=valid,
            max_rules=max_rules,
            verbose=True,
        )

        results.append(result)

        va = result["valid_eval"]
        target_met = is_target_met(result)
        practical_met = is_practical_met(result)

        print(
            f"[SCENARIO RESULT] {scenario['name']} "
            f"target_met={target_met} "
            f"practical_met={practical_met} "
            f"rules={len(result['selected'])} "
            f"selected={va['selected_count']} "
            f"class0_rate={va['class0_rate'] * 100:.2f}% "
            f"coverage={va['class0_coverage'] * 100:.2f}% "
            f"target={scenario['target_coverage'] * 100:.1f}% "
            f"class23={va['class23_rate'] * 100:.2f}% "
            f"class3={va['class3_rate'] * 100:.2f}% "
            f"score={result['score']:.2f}"
        )

    print("\n[SCENARIO SUMMARY]")
    for r in sorted(results, key=lambda x: x["valid_eval"]["class0_coverage"], reverse=True):
        va = r["valid_eval"]
        sc = r["scenario"]
        target_met = is_target_met(r)
        practical_met = is_practical_met(r)

        print(
            f"{sc['name']:30s} "
            f"target_met={str(target_met):5s} "
            f"practical_met={str(practical_met):5s} "
            f"rules={len(r['selected']):3d} "
            f"selected={va['selected_count']:4d} "
            f"class0_rate={va['class0_rate'] * 100:6.2f}% "
            f"coverage={va['class0_coverage'] * 100:6.2f}% "
            f"target={sc['target_coverage'] * 100:5.1f}% "
            f"class2={va['class2_rate'] * 100:6.2f}% "
            f"class3={va['class3_rate'] * 100:6.2f}% "
            f"class23={va['class23_rate'] * 100:6.2f}% "
            f"diag={str(sc.get('diagnostic', False)):5s} "
            f"score={r['score']:10.2f}"
        )

    best = select_best_result(results)

    print(f"\n[SCENARIO SELECTED] {best['scenario']['name']}")

    return (
        best["selected"],
        best["train_mask"],
        best["valid_mask"],
        results,
        best,
    )


# =============================================================================
# Output
# =============================================================================

def cond_to_python_expr(conds: list[tuple[str, str, float]], df_name: str = "df") -> str:
    parts = []

    for feat, op, th in conds:
        parts.append(f"({df_name}[{feat!r}] {op} {float(th)!r})")

    return " & ".join(parts) if parts else "np.ones(len(df), dtype=bool)"


def write_rule_file(
        out_path: Path,
        selected: list[tuple[str, list[tuple[str, str, float]]]],
        header_comment: str,
):
    lines = [
        header_comment.rstrip(),
        "",
        "import numpy as np",
        "",
        "",
        "TARGET_CLASS = 0",
        "",
        "def build_conditions(df):",
        "    conditions = {}",
    ]

    for name, conds in selected:
        expr = cond_to_python_expr(conds, df_name="df")
        lines.append(f"    conditions[{name!r}] = ({expr}).to_numpy(dtype=bool)")

    lines.extend(
        [
            "    return conditions",
            "",
            "def build_mask(df):",
            "    mask = np.zeros(len(df), dtype=bool)",
            "    for cond in build_conditions(df).values():",
            "        mask |= cond",
            "    return mask",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_selected_rule_report(
        out_path: Path,
        selected,
        train: pd.DataFrame,
        valid: pd.DataFrame,
):
    rows = []

    for name, conds in selected:
        train_mask = make_mask_from_conds(train, conds)
        valid_mask = make_mask_from_conds(valid, conds)

        tr = eval_mask(train, train_mask, label="train")
        va = eval_mask(valid, valid_mask, label="valid")

        rows.append(
            {
                "name": name,
                "conds": " AND ".join(f"{f} {op} {th:.8g}" for f, op, th in conds),

                "train_selected": tr["selected_count"],
                "train_class0_rate": tr["class0_rate"],
                "train_class0_lift": tr["class0_lift"],
                "train_class0_wilson_low": tr["class0_wilson_low"],
                "train_class0_coverage": tr["class0_coverage"],
                "train_class1_rate": tr["class1_rate"],
                "train_class2_rate": tr["class2_rate"],
                "train_class3_rate": tr["class3_rate"],
                "train_class23_rate": tr["class23_rate"],

                "valid_selected": va["selected_count"],
                "valid_class0_rate": va["class0_rate"],
                "valid_class0_lift": va["class0_lift"],
                "valid_class0_wilson_low": va["class0_wilson_low"],
                "valid_class0_coverage": va["class0_coverage"],
                "valid_class1_rate": va["class1_rate"],
                "valid_class2_rate": va["class2_rate"],
                "valid_class3_rate": va["class3_rate"],
                "valid_class23_rate": va["class23_rate"],
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_monthly_report(out_path: Path, df: pd.DataFrame, mask: np.ndarray):
    if DATE_COL not in df.columns:
        return

    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    work["_selected"] = mask
    work["_month"] = work[DATE_COL].dt.to_period("M").astype(str)

    rows = []

    for month, g in work.groupby("_month", dropna=True):
        if month == "NaT":
            continue

        m = g["_selected"].to_numpy(dtype=bool)
        ev = eval_mask(g.drop(columns=["_selected", "_month"]), m, label=month)

        rows.append(
            {
                "month": month,
                "total_count": ev["total_count"],
                "selected_count": ev["selected_count"],
                "selected_rate": ev["selected_rate"],
                "base_class0_rate": ev["base_class0_rate"],
                "class0_rate": ev["class0_rate"],
                "class0_lift": ev["class0_lift"],
                "class0_wilson_low": ev["class0_wilson_low"],
                "class0_coverage": ev["class0_coverage"],
                "class1_rate": ev["class1_rate"],
                "class2_rate": ev["class2_rate"],
                "class3_rate": ev["class3_rate"],
                "class23_rate": ev["class23_rate"],
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_scenario_summary(out_path: Path, results: list[dict]):
    rows = []

    for r in results:
        tr = r["train_eval"]
        va = r["valid_eval"]
        sc = r["scenario"]

        target_met = is_target_met(r)
        practical_met = is_practical_met(r)

        rows.append(
            {
                "scenario": sc["name"],
                "alias": sc["alias"],
                "diagnostic": sc.get("diagnostic", False),
                "target_coverage": sc["target_coverage"],
                "min_acceptable_coverage": sc["min_acceptable_coverage"],
                "max_class23": sc["max_class23"],
                "max_class3": sc["max_class3"],
                "target_met": target_met,
                "practical_met": practical_met,
                "rules": len(r["selected"]),
                "score": r["score"],

                "train_selected": tr["selected_count"],
                "train_class0_rate": tr["class0_rate"],
                "train_class0_lift": tr["class0_lift"],
                "train_class0_coverage": tr["class0_coverage"],
                "train_class23_rate": tr["class23_rate"],

                "valid_selected": va["selected_count"],
                "valid_selected_rate": va["selected_rate"],
                "valid_class0_count": va["class0_count"],
                "valid_class0_rate": va["class0_rate"],
                "valid_class0_lift": va["class0_lift"],
                "valid_class0_wilson_low": va["class0_wilson_low"],
                "valid_class0_coverage": va["class0_coverage"],
                "valid_class1_rate": va["class1_rate"],
                "valid_class2_rate": va["class2_rate"],
                "valid_class3_rate": va["class3_rate"],
                "valid_class23_rate": va["class23_rate"],
                "valid_class1_loss_rate": va["class1_loss_rate"],
                "valid_class2_loss_rate": va["class2_loss_rate"],
                "valid_class3_loss_rate": va["class3_loss_rate"],
            }
        )

    pd.DataFrame(rows).sort_values(
        ["target_met", "practical_met", "diagnostic", "valid_class0_coverage", "valid_class3_rate"],
        ascending=[False, False, True, False, True],
    ).to_csv(out_path, index=False, encoding="utf-8-sig")


def scenario_out_path(base_out_path: Path, scenario_name: str) -> Path:
    stem = base_out_path.stem
    suffix = base_out_path.suffix or ".py"

    return base_out_path.with_name(f"{stem}_{scenario_name}{suffix}")


def write_outputs_for_result(
        result: dict,
        out_path: Path,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        split_date: pd.Timestamp | None,
        is_main: bool = False,
):
    scenario = result["scenario"]
    selected = result["selected"]

    split_comment = (
        f"# split_date: {pd.to_datetime(split_date).date()}\n"
        if split_date is not None
        else "# split_date: row-ratio split\n"
    )

    header = (
            "# auto-generated: lowscan high-probability avoid rules for target_class == 0\n"
            + split_comment
            + f"# scenario: {scenario['name']}\n"
            + f"# alias: {scenario['alias']}\n"
            + f"# target_coverage: {scenario['target_coverage']}\n"
            + f"# max_class23: {scenario['max_class23']}\n"
            + f"# max_class3: {scenario['max_class3']}\n"
            + "# objective: remove target_class 0 while controlling class2/class3 contamination\n"
            + "# use build_mask(df) to get selected target0/no-bounce candidates\n"
    )

    write_rule_file(out_path, selected, header_comment=header)

    save_selected_rule_report(
        out_path.with_suffix(".report.csv"),
        selected,
        train,
        valid,
    )

    save_monthly_report(
        out_path.with_suffix(".monthly_report.csv"),
        valid,
        result["valid_mask"],
    )

    if is_main:
        print("\n[MAIN OUTPUT]")
    else:
        print(f"\n[SCENARIO OUTPUT: {scenario['name']}]")

    print("rule_file:", out_path)
    print("report:", out_path.with_suffix(".report.csv"))
    print("monthly_report:", out_path.with_suffix(".monthly_report.csv"))


def interpret_final_result(valid_eval: dict):
    print("\n[INTERPRETATION]")

    rate = valid_eval["class0_rate"]
    coverage = valid_eval["class0_coverage"]
    selected = valid_eval["selected_count"]
    class2 = valid_eval["class2_rate"]
    class3 = valid_eval["class3_rate"]
    class23 = valid_eval["class23_rate"]

    print(f"valid_selected_count : {selected}")
    print(f"valid_class0_rate    : {rate * 100:.2f}%")
    print(f"valid_class0_coverage: {coverage * 100:.2f}%")
    print(f"valid_class2_rate    : {class2 * 100:.2f}%")
    print(f"valid_class3_rate    : {class3 * 100:.2f}%")
    print(f"valid_class23_rate   : {class23 * 100:.2f}%")

    if coverage >= 0.05 and class23 <= 0.12:
        print("해석: coverage 5% 이상이고 class23도 12% 이하입니다. 매우 좋은 결과입니다.")
    elif coverage >= 0.05 and class23 <= 0.15:
        print("해석: coverage 5% 이상이고 class23 15% 이하입니다. 실전 검토 가능한 결과입니다.")
    elif coverage >= 0.05:
        print("해석: coverage 5%는 달성했지만 class23 손실이 큽니다. 진단용으로 보고 실전 사용은 주의하세요.")
    elif coverage >= 0.038 and class3 <= 0.06 and class23 <= 0.16:
        print("해석: coverage 5%는 미달이지만 class3/class23 방어가 좋은 실전형 후보입니다.")
    elif coverage >= 0.035 and class3 <= 0.06:
        print("해석: coverage는 3.5% 이상이며 class3 방어는 양호합니다. 보수적 실전 후보입니다.")
    else:
        print("해석: coverage 5% 미달입니다. scenario_summary에서 diagnostic c23_22/c23_25를 확인해 한계점을 보세요.")


# =============================================================================
# Main
# =============================================================================

def find_target0_highprob_rules(
        csv_path: str | Path = CSV_PATH,
        out_path: str | Path = OUT_PATH,
        mode: str = "all",
        save_all_scenarios: bool = False,
        workers: int = 1,
):
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    df = pd.read_csv(csv_path, low_memory=False)

    if "target_class" not in df.columns:
        raise ValueError("target_class 컬럼이 없습니다.")

    df = df[df["target_class"].notna()].copy()
    df["target_class"] = df["target_class"].astype(int)

    train, valid, split_date = split_train_valid_by_date_ratio(
        df,
        valid_ratio=VALID_RATIO,
        date_col=DATE_COL,
    )

    print("[DATA]")
    print("csv:", csv_path)
    print("rows:", len(df), "train:", len(train), "valid:", len(valid))
    print("workers:", workers)
    print("valid_ratio:", VALID_RATIO)

    if split_date is not None:
        print("split_date:", pd.to_datetime(split_date).date())

    print("\n[TARGET DISTRIBUTION]")
    print("train")
    print(train["target_class"].value_counts(normalize=True).sort_index().round(4))
    print("valid")
    print(valid["target_class"].value_counts(normalize=True).sort_index().round(4))

    features = get_features(train)
    print("\n[FEATURES]", len(features), features)

    literals, literal_masks = build_literals(train, features)
    print("[LITERALS]", len(literals))

    feature_groups, group_limits = get_feature_groups()
    train_class0 = train["target_class"].to_numpy() == TARGET_CLASS

    rules = mine_class0_rules(
        df=train,
        literals=literals,
        literal_masks=literal_masks,
        class0=train_class0,
        min_count=MIN_CNT,
        max_depth=MAX_DEPTH,
        beam=BEAM,
        top_n=TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    print(f"\n[CLASS0] 후보 룰 개수: {len(rules)}")

    selected, train_mask, valid_mask, results, best = select_rules_with_validation(
        train=train,
        valid=valid,
        rules=rules,
        mode=mode,
        max_rules=MAX_RULES,
        workers=workers,
    )

    print(f"\n[CLASS0] 최종 통과 룰 개수: {len(selected)} / {len(rules)}")

    if results:
        save_scenario_summary(
            out_path.with_suffix(".scenario_summary.csv"),
            results,
        )
        print("scenario_summary:", out_path.with_suffix(".scenario_summary.csv"))

    if best is not None:
        write_outputs_for_result(
            result=best,
            out_path=out_path,
            train=train,
            valid=valid,
            split_date=split_date,
            is_main=True,
        )

    if save_all_scenarios or mode == "all":
        for r in results:
            scenario_path = scenario_out_path(out_path, r["scenario"]["alias"])
            write_outputs_for_result(
                result=r,
                out_path=scenario_path,
                train=train,
                valid=valid,
                split_date=split_date,
                is_main=False,
            )

    print_eval(eval_mask(train, train_mask, label="TRAIN COMBINED"))
    valid_final_eval = eval_mask(valid, valid_mask, label="VALID COMBINED")
    print_eval(valid_final_eval)
    interpret_final_result(valid_final_eval)

    return selected, train_mask, valid_mask, results


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help="input csv path",
    )

    parser.add_argument(
        "--out",
        default=str(OUT_PATH),
        help="output rule python file",
    )

    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "auto",
            "all",
            "cov4_c3_05",
            "cov4_c3_06",
            "c23_10",
            "c23_12",
            "c23_15",
            "c23_18",
            "c23_22",
            "c23_25",
            "cov4_c3_05_safe",
            "cov4_c3_06_balanced",
            "cov5_c23_10_strict",
            "cov5_c23_12_balanced",
            "cov5_c23_15_reach",
            "cov5_c23_18_last_resort",
            "cov5_c23_22_diagnostic",
            "cov5_c23_25_diagnostic",
        ],
        help="scenario mode",
    )

    parser.add_argument(
        "--save-all-scenarios",
        action="store_true",
        help="save scenario-specific rule files and reports",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of worker processes for rule evaluation",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="use faster but rougher settings",
    )

    return parser.parse_args()


def apply_fast_mode():
    global BEAM
    global TOP_N
    global MAX_DEPTH
    global N_QUANTILES
    global MIN_CNT
    global MAX_RULES

    BEAM = 7000
    TOP_N = 7000
    MAX_DEPTH = 4
    N_QUANTILES = 14
    MIN_CNT = 20
    MAX_RULES = 160

    print("[FAST MODE]")
    print("BEAM:", BEAM)
    print("TOP_N:", TOP_N)
    print("MAX_DEPTH:", MAX_DEPTH)
    print("N_QUANTILES:", N_QUANTILES)
    print("MIN_CNT:", MIN_CNT)
    print("MAX_RULES:", MAX_RULES)


if __name__ == "__main__":
    mp.freeze_support()

    args = parse_args()

    if args.fast:
        apply_fast_mode()

    find_target0_highprob_rules(
        csv_path=args.csv,
        out_path=args.out,
        mode=args.mode,
        save_all_scenarios=args.save_all_scenarios,
        workers=args.workers,
    )

    try:
        import winsound

        winsound.Beep(1500, 500)
        winsound.Beep(1000, 500)
    except Exception:
        pass