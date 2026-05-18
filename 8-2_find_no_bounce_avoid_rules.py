"""
target_class == 0 고확률 룰 마이닝 스크립트

목적
- target_class == 0 인 샘플을 높은 확률로 찾는 pick rule 생성
- train 에서 룰 탐색, valid 에서 검증
- 고확률 우선: precision, lift, Wilson lower bound, valid 안정성 반영
- class2/class3 혼입률을 제한해서 좋은 샘플이 섞이는 것을 방지
- utils.py 없이 단독 실행 가능

실행
    python find_target0_highprob_rules.py --csv d0029855-4c84-40b2-8f70-f78c442db855.csv

생성
    lowscan_target0_highprob_rules.py
    lowscan_target0_highprob_rules.report.csv
    lowscan_target0_highprob_rules.monthly_report.csv

사용
    import numpy as np
    import lowscan_target0_highprob_rules

    conditions = lowscan_target0_highprob_rules.build_conditions(df)
    target0_mask = np.zeros(len(df), dtype=bool)
    for cond in conditions.values():
        target0_mask |= cond

상수 조정 가이드
- 더 높은 확률: MIN_CLASS0_RATE, VALID_MIN_CLASS0_RATE, VALID_MIN_WILSON_LOW 를 올림
- 더 많은 신호: 위 세 값을 낮추고, MIN_CNT / VALID_MIN_CNT 를 낮춤
- class2/class3 보호 강화: MAX_VALID_CLASS2_RATE, MAX_VALID_CLASS3_RATE 를 낮춤
"""

from __future__ import annotations

import argparse
import heapq
import math
from itertools import count
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG: 여기 상수만 바꿔서 실험
#
# [REBALANCED VERSION]
# 이전 fast 버전은 train 후보는 나오지만 valid filter 통과가 0개가 될 수 있었음.
# 이 버전은 속도는 원본보다 줄이되, 검증 후보가 살아남도록 탐색/검증 기준을 재균형화.
# - N_QUANTILES: 22 -> 16
# - BEAM       : 10000 -> 5000
# - TOP_N      : 3000 -> 2500
# - MIN_CNT    : 45 -> 60
# - MAX_DEPTH  : 5 유지
# - MAX_RULES  : None -> 35
# - VALID 조건은 strict 기준보다 완화하되, class0_rate 60% 이상을 기본 목표로 유지.
# =============================================================================

CSV_PATH = "csv/low_result_7_desc.csv"
OUT_PATH = Path("lowscan_target0_highprob_rules.py")

TARGET_CLASS = 0

# 최근 날짜 일부를 validation 으로 사용
VALID_RATIO = 0.10
DATE_COL = "today"

# 리터럴 생성: 숫자형 feature별 분위수 threshold 개수
N_QUANTILES = 16
MIN_UNIQUE_VALUES = 8

# Beam search
BEAM = 5000
TOP_N = 2500
MIN_CNT = 60
MAX_DEPTH = 5

# 최종 후보 룰: train 기준
MIN_CLASS0_RATE = 0.66
MIN_LIFT = 1.55
MIN_WILSON_LOW = 0.58

# validation 필터: 과최적화 제거용
VALID_MIN_CNT = 20
VALID_MIN_CLASS0_RATE = 0.60
VALID_MIN_LIFT = 1.30
VALID_MIN_WILSON_LOW = 0.47

# 좋은 class 혼입 제한. 필요 없으면 1.0 으로 완화.
MAX_VALID_CLASS1_RATE = 0.34
MAX_VALID_CLASS2_RATE = 0.24
MAX_VALID_CLASS3_RATE = 0.24
MAX_VALID_CLASS23_RATE = 0.38

# beam 확장 조건: depth별로 점점 엄격하게
# 부족하면 마지막 값을 재사용
EXPAND_MIN_CLASS0_RATE = [0.38, 0.46, 0.54, 0.60, 0.64]
EXPAND_MIN_LIFT = [0.90, 1.08, 1.25, 1.38, 1.48]

# greedy rule selection
MAX_RULES = 35
MIN_NEW_CLASS0_CNT = 4
MIN_NEW_CLASS0_RATE = 0.56

# 점수 가중치
PRECISION_POWER = 3.2
LIFT_POWER = 1.4
WILSON_POWER = 2.2
COVERAGE_POWER = 0.55
CLASS1_PENALTY = 0.8
CLASS2_PENALTY = 1.6
CLASS3_PENALTY = 2.4

# valid strict 조건을 통과하는 후보가 0개일 때를 대비한 완화 tier.
# tier 순서대로만 완화하므로, strict 후보가 있으면 strict만 사용한다.
VALID_FILTER_TIERS = [
    {
        "name": "strict",
        "min_cnt": VALID_MIN_CNT,
        "min_class0_rate": VALID_MIN_CLASS0_RATE,
        "min_lift": VALID_MIN_LIFT,
        "min_wilson_low": VALID_MIN_WILSON_LOW,
        "max_class1_rate": MAX_VALID_CLASS1_RATE,
        "max_class2_rate": MAX_VALID_CLASS2_RATE,
        "max_class3_rate": MAX_VALID_CLASS3_RATE,
        "max_class23_rate": MAX_VALID_CLASS23_RATE,
    },
    {
        "name": "balanced_relaxed",
        "min_cnt": max(15, VALID_MIN_CNT - 5),
        "min_class0_rate": 0.58,
        "min_lift": 1.22,
        "min_wilson_low": 0.44,
        "max_class1_rate": 0.38,
        "max_class2_rate": 0.27,
        "max_class3_rate": 0.27,
        "max_class23_rate": 0.42,
    },
    {
        "name": "coverage_rescue",
        "min_cnt": 12,
        "min_class0_rate": 0.56,
        "min_lift": 1.15,
        "min_wilson_low": 0.40,
        "max_class1_rate": 0.42,
        "max_class2_rate": 0.30,
        "max_class3_rate": 0.30,
        "max_class23_rate": 0.46,
    },
]

# feature group 제한: 유사 feature가 한 룰에 과도하게 중복되는 것을 방지
USE_FEATURE_GROUP_LIMITS = True


# =============================================================================
# Feature / literal utilities
# =============================================================================

def get_exclude_columns(df: pd.DataFrame | None = None) -> set[str]:
    """룰 조건에 쓰면 안 되는 컬럼."""
    exclude = {
        # 식별자 / 메타
        "ticker", "stock_name", "today", "idx",

        # stop / target / label
        "stop_loss", "stop_day", "target_pct", "target_class",

        # 과거 실험용 / raw 후보
        "_close_pos_20d", "_tr_value_ratio", "_tr_value_ratio_5d",
        "_dist_to_high_20d", "_BB_perc", "_UltimateOsc", "_CCI14",
        "_ADX14", "_gap_pct", "_vol_ratio_15_60", "_RSI_rebound",
        "_rebound_power", "_MACD_hist_1d", "_MACD_acc",
        "_MACD_hist_3d_close_norm",
    }

    if df is not None:
        for c in df.columns:
            if (
                    c.startswith("validation_")
                    or c.startswith("day_to_")
                    or c.startswith("target_before_stop_")
                    or c.startswith("stop_before_target_")
                    or c.startswith("target_stop_same_day_")
                    or c.startswith("no_target_no_stop_")
                    or c.startswith("fast_success_")
                    or c.startswith("slow_success_")
                    or c.startswith("fail_success_")
            ):
                exclude.add(c)

    return exclude


def get_features(df: pd.DataFrame) -> list[str]:
    exclude = get_exclude_columns(df)
    features: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().sum() == 0:
            continue
        if df[c].nunique(dropna=True) < MIN_UNIQUE_VALUES:
            continue
        features.append(c)
    return features


def get_feature_groups() -> tuple[dict[str, str], dict[str, int]]:
    feature_groups = {
        "today_pct": "PRICE",
        "max_drop_7d": "DROP",
        "dist_from_low_20d": "POSITION",
        "three_m_cur_max_chg_rate": "POSITION",
        "dist_to_ma5": "POSITION",
        "dist_to_ma20": "POSITION",
        "pct_vs_lastweek": "WEEK_POSITION",
        "ma5_ma20_gap_chg_1d": "TREND",
        "gap_pct": "GAP",
        "today_tr_val_eok": "VOLUME",
        "tr_val_rank_20d": "VOLUME",
        "tr_value_ratio_5d": "VOLUME",
        "MACD_hist_3d": "MACD",
        "vol5": "VOLATILITY",
        "ATR_pct": "VOLATILITY",
        "vol_ratio_5_15": "VOLATILITY",
    }

    group_limits = {
        "PRICE": 1,
        "DROP": 1,
        "POSITION": 2,
        "WEEK_POSITION": 1,
        "TREND": 1,
        "GAP": 1,
        "VOLATILITY": 2,
        "VOLUME": 1,
        "MACD": 1,
    }
    return feature_groups, group_limits


def _thresholds_for_series(s: pd.Series, n_quantiles: int) -> list[float]:
    qs = np.linspace(0.04, 0.96, n_quantiles)
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return []
    ths = vals.quantile(qs).to_numpy()
    return sorted({round(float(x), 10) for x in ths if np.isfinite(x)})


def build_literals(
        df: pd.DataFrame,
        features: Iterable[str],
) -> tuple[list[tuple[str, str, float]], list[np.ndarray]]:
    literals: list[tuple[str, str, float]] = []
    masks: list[np.ndarray] = []

    for feat in features:
        values = pd.to_numeric(df[feat], errors="coerce")
        thresholds = _thresholds_for_series(values, N_QUANTILES)
        arr = values.to_numpy()
        finite = np.isfinite(arr)

        for th in thresholds:
            literals.append((feat, "<=", th))
            masks.append(finite & (arr <= th))
            literals.append((feat, ">=", th))
            masks.append(finite & (arr >= th))

    return literals, masks


def make_mask_from_conds(df: pd.DataFrame, conds: list[tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for feat, op, th in conds:
        arr = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        finite = np.isfinite(arr)
        if op == "<=":
            mask &= finite & (arr <= th)
        elif op == ">=":
            mask &= finite & (arr >= th)
        else:
            raise ValueError(f"unknown operator: {op}")
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
    """Binomial proportion Wilson lower bound."""
    if total <= 0:
        return 0.0
    p = success / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def class_rates(df: pd.DataFrame, mask: np.ndarray) -> dict[int, float]:
    sub = df[mask]
    n = len(sub)
    rates: dict[int, float] = {}
    for cls in [0, 1, 2, 3]:
        rates[cls] = float((sub["target_class"] == cls).mean()) if n else 0.0
    return rates


def eval_mask(df: pd.DataFrame, mask: np.ndarray, label: str = "") -> dict:
    sub = df[mask]
    total = len(df)
    selected = len(sub)

    class0_total = int((df["target_class"] == TARGET_CLASS).sum())
    class0_cnt = int((sub["target_class"] == TARGET_CLASS).sum())
    base_rate = class0_total / total if total else 0.0
    class0_rate = class0_cnt / selected if selected else 0.0

    c = {cls: int((sub["target_class"] == cls).sum()) for cls in [0, 1, 2, 3]}
    ct = {cls: int((df["target_class"] == cls).sum()) for cls in [0, 1, 2, 3]}

    return {
        "label": label,
        "total_count": total,
        "selected_count": selected,
        "selected_rate": selected / total if total else 0.0,
        "target_class": TARGET_CLASS,
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
    penalty = max(0.01, 1.0 - CLASS1_PENALTY * c1_rate - CLASS2_PENALTY * c2_rate - CLASS3_PENALTY * c3_rate)
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
        "\nvalid_min_class0_rate:", VALID_MIN_CLASS0_RATE,
        "\nvalid_min_wilson_low:", VALID_MIN_WILSON_LOW,
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

                score = _rule_score(cnt, class0_cnt, class0_rate, base_rate, c1_rate, c2_rate, c3_rate)

                if (
                        class0_rate >= MIN_CLASS0_RATE
                        and lift >= MIN_LIFT
                        and wilson >= MIN_WILSON_LOW
                        and c2_rate <= MAX_VALID_CLASS2_RATE
                        and c3_rate <= MAX_VALID_CLASS3_RATE
                        and c23_rate <= MAX_VALID_CLASS23_RATE
                ):
                    key2 = tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in (conds + [lit])))
                    prev = good.get(key2)
                    if prev is None or score > prev[9]:
                        good[key2] = (
                            cnt, class0_cnt, class0_rate, lift, wilson,
                            c1_rate, c2_rate, c3_rate, conds + [lit], score,
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
        cnt, class0_cnt, class0_rate, lift, wilson, c1_rate, c2_rate, c3_rate, conds, score = x
        return (-score, -wilson, -class0_rate, -lift, c3_rate, c2_rate, -class0_cnt, -cnt)

    out = sorted(good.values(), key=out_key)
    if top_n is not None:
        out = out[:top_n]
    return out


# =============================================================================
# Selection / output
# =============================================================================

def pass_valid_filter(ev: dict, tier: dict | None = None) -> bool:
    if tier is None:
        tier = VALID_FILTER_TIERS[0]
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


def validation_fail_reasons(ev: dict, tier: dict) -> list[str]:
    reasons = []
    checks = [
        ("cnt", ev["selected_count"], ">=", tier["min_cnt"]),
        ("class0_rate", ev["class0_rate"], ">=", tier["min_class0_rate"]),
        ("lift", ev["class0_lift"], ">=", tier["min_lift"]),
        ("wilson", ev["class0_wilson_low"], ">=", tier["min_wilson_low"]),
        ("class1_rate", ev["class1_rate"], "<=", tier["max_class1_rate"]),
        ("class2_rate", ev["class2_rate"], "<=", tier["max_class2_rate"]),
        ("class3_rate", ev["class3_rate"], "<=", tier["max_class3_rate"]),
        ("class23_rate", ev["class23_rate"], "<=", tier["max_class23_rate"]),
    ]
    for name, value, op, threshold in checks:
        if op == ">=" and value < threshold:
            reasons.append(f"{name} {value:.4f} < {threshold:.4f}")
        elif op == "<=" and value > threshold:
            reasons.append(f"{name} {value:.4f} > {threshold:.4f}")
    return reasons


def select_rules_with_validation(train: pd.DataFrame, valid: pd.DataFrame, rules, max_rules: int | None = MAX_RULES):
    train_class0 = train["target_class"].to_numpy() == TARGET_CLASS
    valid_class0 = valid["target_class"].to_numpy() == TARGET_CLASS
    train_base = train_class0.mean()
    valid_base = valid_class0.mean()

    all_evaluated = []
    for i, (cnt, class0_cnt, class0_rate, lift, wilson, c1_rate, c2_rate, c3_rate, conds, score) in enumerate(rules, start=1):
        name = (
            f"target0_highprob_{i:04d}"
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
            continue
        if tr["class0_rate"] < MIN_CLASS0_RATE or tr["class0_lift"] < MIN_LIFT or tr["class0_wilson_low"] < MIN_WILSON_LOW:
            continue

        stability_gap = abs(tr["class0_rate"] - va["class0_rate"])
        stable_score = score * (1.0 - min(stability_gap, 0.35))

        all_evaluated.append({
            "name": name,
            "conds": conds,
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "score": score,
            "stable_score": stable_score,
            "train_eval": tr,
            "valid_eval": va,
        })

    print("\n[EVALUATED TRAIN-PASS RULES]", len(all_evaluated))

    candidates = []
    selected_tier = None
    for tier in VALID_FILTER_TIERS:
        tier_candidates = [c for c in all_evaluated if pass_valid_filter(c["valid_eval"], tier)]
        print(f"[VALID FILTER:{tier['name']}] passed={len(tier_candidates)} / {len(all_evaluated)}")
        if tier_candidates:
            candidates = tier_candidates
            selected_tier = tier
            break

    if not candidates:
        print("\n[VALID FILTER DIAGNOSTIC] no candidates passed. Top valid candidates by class0_rate:")
        top_debug = sorted(
            all_evaluated,
            key=lambda c: (
                c["valid_eval"]["class0_rate"],
                c["valid_eval"]["class0_wilson_low"],
                c["valid_eval"]["selected_count"],
            ),
            reverse=True,
        )[:20]
        strict = VALID_FILTER_TIERS[0]
        for c in top_debug:
            va = c["valid_eval"]
            reasons = "; ".join(validation_fail_reasons(va, strict))
            print(
                c["name"],
                f"valid_n={va['selected_count']}",
                f"class0={va['class0_rate'] * 100:.2f}%",
                f"wilson={va['class0_wilson_low'] * 100:.2f}%",
                f"lift={va['class0_lift']:.2f}",
                f"c1={va['class1_rate'] * 100:.1f}%",
                f"c2={va['class2_rate'] * 100:.1f}%",
                f"c3={va['class3_rate'] * 100:.1f}%",
                "fail:", reasons,
            )
        return [], np.zeros(len(train), dtype=bool), np.zeros(len(valid), dtype=bool)

    print(f"[VALID FILTER SELECTED TIER] {selected_tier['name']}")
    print("[CANDIDATES AFTER VALID FILTER]", len(candidates))

    selected = []
    combined_train_mask = np.zeros(len(train), dtype=bool)
    combined_valid_mask = np.zeros(len(valid), dtype=bool)

    while True:
        best = None

        for cand in candidates:
            new_train = cand["train_mask"] & ~combined_train_mask
            new_valid = cand["valid_mask"] & ~combined_valid_mask
            if not new_train.any():
                continue

            train_added_total = int(new_train.sum())
            train_added_class0 = int((new_train & train_class0).sum())
            train_added_rate = train_added_class0 / train_added_total if train_added_total else 0.0
            train_added_lift = train_added_rate / train_base if train_base else 0.0
            train_added_wilson = wilson_lower_bound(train_added_class0, train_added_total)

            valid_added_total = int(new_valid.sum())
            valid_added_class0 = int((new_valid & valid_class0).sum())
            valid_added_rate = valid_added_class0 / valid_added_total if valid_added_total else 0.0
            valid_added_lift = valid_added_rate / valid_base if valid_base else 0.0
            valid_added_wilson = wilson_lower_bound(valid_added_class0, valid_added_total)

            if train_added_class0 < MIN_NEW_CLASS0_CNT:
                continue
            if train_added_rate < MIN_NEW_CLASS0_RATE:
                continue

            # 신규 valid가 충분히 있으면 tier 기준으로 확인.
            # 신규 valid가 작으면 전체 rule valid 성능을 보조지표로 쓰되, 신규 class0가 0이면 배제.
            if valid_added_total >= selected_tier["min_cnt"]:
                if (
                        valid_added_rate < max(0.54, selected_tier["min_class0_rate"] - 0.04)
                        or valid_added_wilson < max(0.36, selected_tier["min_wilson_low"] - 0.06)
                ):
                    continue
            elif valid_added_total > 0 and valid_added_class0 == 0:
                continue

            valid_component = valid_added_rate if valid_added_total >= selected_tier["min_cnt"] else cand["valid_eval"]["class0_rate"]
            valid_wilson_component = valid_added_wilson if valid_added_total >= selected_tier["min_cnt"] else cand["valid_eval"]["class0_wilson_low"]

            valid_false = valid_added_total - valid_added_class0
            class3_penalty = cand["valid_eval"]["class3_rate"]

            key = (
                valid_component,
                valid_wilson_component,
                train_added_rate,
                train_added_wilson,
                valid_added_class0,
                -valid_false,
                -class3_penalty,
                train_added_class0,
                cand["stable_score"],
            )

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "cand": cand,
                    "train_added_total": train_added_total,
                    "train_added_class0": train_added_class0,
                    "train_added_rate": train_added_rate,
                    "train_added_lift": train_added_lift,
                    "train_added_wilson": train_added_wilson,
                    "valid_added_total": valid_added_total,
                    "valid_added_class0": valid_added_class0,
                    "valid_added_rate": valid_added_rate,
                    "valid_added_lift": valid_added_lift,
                    "valid_added_wilson": valid_added_wilson,
                }

        if best is None:
            break

        cand = best["cand"]
        combined_train_mask |= cand["train_mask"]
        combined_valid_mask |= cand["valid_mask"]
        selected.append((cand["name"], cand["conds"]))

        combined_valid_eval = eval_mask(valid, combined_valid_mask, label="VALID COMBINED")
        print(
            "[SELECT]",
            f"{len(selected):03d}",
            cand["name"],
            f"train_add={best['train_added_class0']}/{best['train_added_total']} ({best['train_added_rate'] * 100:.2f}%)",
            f"valid_add={best['valid_added_class0']}/{best['valid_added_total']} ({best['valid_added_rate'] * 100:.2f}%)",
            f"valid_rule={cand['valid_eval']['class0_rate'] * 100:.2f}%",
            f"combined_valid_class0_rate={combined_valid_eval['class0_rate'] * 100:.2f}%",
            f"combined_valid_class0_coverage={combined_valid_eval['class0_coverage'] * 100:.2f}%",
            f"combined_valid_class3_rate={combined_valid_eval['class3_rate'] * 100:.2f}%",
        )

        if max_rules is not None and len(selected) >= max_rules:
            break

    return selected, combined_train_mask, combined_valid_mask


def cond_to_python_expr(conds: list[tuple[str, str, float]], df_name: str = "df") -> str:
    parts = []
    for feat, op, th in conds:
        parts.append(f"({df_name}[{feat!r}] {op} {float(th)!r})")
    return " & ".join(parts) if parts else "np.ones(len(df), dtype=bool)"


def write_rule_file(out_path: Path, selected: list[tuple[str, list[tuple[str, str, float]]]], header_comment: str):
    lines = [header_comment.rstrip(), "", "import numpy as np", "", ""]
    lines.append("TARGET_CLASS = 0")
    lines.append("")
    lines.append("def build_conditions(df):")
    lines.append("    conditions = {}")
    for name, conds in selected:
        expr = cond_to_python_expr(conds, df_name="df")
        lines.append(f"    conditions[{name!r}] = ({expr}).to_numpy(dtype=bool)")
    lines.append("    return conditions")
    lines.append("")
    lines.append("def build_mask(df):")
    lines.append("    mask = np.zeros(len(df), dtype=bool)")
    lines.append("    for cond in build_conditions(df).values():")
    lines.append("        mask |= cond")
    lines.append("    return mask")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_selected_rule_report(out_path: Path, selected, train: pd.DataFrame, valid: pd.DataFrame):
    rows = []
    for name, conds in selected:
        train_mask = make_mask_from_conds(train, conds)
        valid_mask = make_mask_from_conds(valid, conds)
        tr = eval_mask(train, train_mask, label="train")
        va = eval_mask(valid, valid_mask, label="valid")
        rows.append({
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
            "valid_selected": va["selected_count"],
            "valid_class0_rate": va["class0_rate"],
            "valid_class0_lift": va["class0_lift"],
            "valid_class0_wilson_low": va["class0_wilson_low"],
            "valid_class0_coverage": va["class0_coverage"],
            "valid_class1_rate": va["class1_rate"],
            "valid_class2_rate": va["class2_rate"],
            "valid_class3_rate": va["class3_rate"],
            "valid_class23_rate": va["class23_rate"],
        })
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
        rows.append({
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
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


# =============================================================================
# Main
# =============================================================================

def find_target0_highprob_rules(csv_path: str | Path = CSV_PATH, out_path: str | Path = OUT_PATH):
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    df = pd.read_csv(csv_path, low_memory=False)
    if "target_class" not in df.columns:
        raise ValueError("target_class 컬럼이 없습니다.")

    df = df[df["target_class"].notna()].copy()
    df["target_class"] = df["target_class"].astype(int)

    train, valid, split_date = split_train_valid_by_date_ratio(df, valid_ratio=VALID_RATIO, date_col=DATE_COL)

    print("[DATA]")
    print("csv:", csv_path)
    print("rows:", len(df), "train:", len(train), "valid:", len(valid))
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

    selected, train_mask, valid_mask = select_rules_with_validation(
        train=train,
        valid=valid,
        rules=rules,
        max_rules=MAX_RULES,
    )

    print(f"\n[CLASS0] 최종 통과 룰 개수: {len(selected)} / {len(rules)}")

    split_comment = f"# split_date: {pd.to_datetime(split_date).date()}\n" if split_date is not None else "# split_date: row-ratio split\n"
    write_rule_file(
        out_path,
        selected,
        header_comment=(
                "# auto-generated: lowscan high-probability pick rules for target_class == 0\n"
                + split_comment
                + "# generated on train, filtered by validation\n"
                + "# use build_mask(df) to get selected target0 candidates\n"
        ),
    )

    report_path = out_path.with_suffix(".report.csv")
    monthly_path = out_path.with_suffix(".monthly_report.csv")
    save_selected_rule_report(report_path, selected, train, valid)
    save_monthly_report(monthly_path, valid, valid_mask)

    print_eval(eval_mask(train, train_mask, label="TRAIN COMBINED"))
    print_eval(eval_mask(valid, valid_mask, label="VALID COMBINED"))

    print("\n[OUTPUT]")
    print("rule_file:", out_path)
    print("report:", report_path)
    print("monthly_report:", monthly_path)

    return selected, train_mask, valid_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH, help="input csv path")
    parser.add_argument("--out", default=str(OUT_PATH), help="output rule python file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    find_target0_highprob_rules(args.csv, args.out)

    try:
        import winsound
        winsound.Beep(1500, 500)
        winsound.Beep(1000, 500)
    except Exception:
        pass
