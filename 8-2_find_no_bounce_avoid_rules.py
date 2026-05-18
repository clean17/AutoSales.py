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
# [CLASS0 SAFE COVERAGE PARETO VERSION]
# 목적:
# - 기존 방식은 class0 후보를 만든 뒤 class23를 사후 패널티로만 처리해서,
#   coverage를 올리면 class23도 같이 올라가고, class23를 낮추면 coverage가 막혔음.
# - 이 버전은 여러 목적을 동시에 평가한다.
#
# 핵심:
# 1) 후보 룰 자체는 class0가 높거나 class23가 낮은 룰을 모두 후보로 둔다.
# 2) 선택 단계에서 다음 4개 시나리오를 모두 실행한다.
#    - precision_safe
#    - class23_safe
#    - balanced_safe_coverage
#    - coverage_push_with_class23_cap
# 3) 최종 결과를 시나리오별로 비교 출력한다.
#
# 중요한 현실:
# - 현재 로그 기준으로 class0_rate 상승 + coverage 상승 + class23_rate 하락을 동시에 만족하는
#   추가 룰이 충분하지 않을 수 있음.
# - 그 경우 이 코드는 억지로 나쁜 룰을 붙이지 않고, Pareto 후보를 보여준다.
#
# =============================================================================

CSV_PATH = "csv/low_result_7_desc.csv"
OUT_PATH = Path("lowscan_target0_highprob_rules.py")

TARGET_CLASS = 0

# 최근 날짜 일부를 validation 으로 사용
VALID_RATIO = 0.10
DATE_COL = "today"

# 리터럴 생성: 숫자형 feature별 분위수 threshold 개수
N_QUANTILES = 18
MIN_UNIQUE_VALUES = 8

# Beam search
BEAM = 9000
TOP_N = 7000
MIN_CNT = 40
MAX_DEPTH = 5

# 최종 후보 룰: train 기준
MIN_CLASS0_RATE = 0.54
MIN_LIFT = 1.08
MIN_WILSON_LOW = 0.40

# validation 필터: 과최적화 제거용
VALID_MIN_CNT = 10
VALID_MIN_CLASS0_RATE = 0.48
VALID_MIN_LIFT = 1.00
VALID_MIN_WILSON_LOW = 0.28

# 좋은 class 혼입 제한. 필요 없으면 1.0 으로 완화.
MAX_VALID_CLASS1_RATE = 0.52
MAX_VALID_CLASS2_RATE = 0.34
MAX_VALID_CLASS3_RATE = 0.28
MAX_VALID_CLASS23_RATE = 0.48

# beam 확장 조건: depth별로 점점 엄격하게
# 부족하면 마지막 값을 재사용
EXPAND_MIN_CLASS0_RATE = [0.38, 0.46, 0.54, 0.60, 0.64]
EXPAND_MIN_LIFT = [0.90, 1.08, 1.25, 1.38, 1.48]

# greedy rule selection
MAX_RULES = 120
MIN_NEW_CLASS0_CNT = 1
MIN_NEW_CLASS0_RATE = 0.42

# 점수 가중치
PRECISION_POWER = 3.2
LIFT_POWER = 1.4
WILSON_POWER = 2.2
COVERAGE_POWER = 0.55
CLASS1_PENALTY = 0.8
CLASS2_PENALTY = 2.2
CLASS3_PENALTY = 3.2

# valid strict 조건을 통과하는 후보가 0개일 때를 대비한 완화 tier.
# tier 순서대로만 완화하므로, strict 후보가 있으면 strict만 사용한다.
VALID_FILTER_TIERS = [
    {
        "name": "precision_safe_pool",
        "min_cnt": 10,
        "min_class0_rate": 0.58,
        "min_lift": 1.12,
        "min_wilson_low": 0.36,
        "max_class1_rate": 0.50,
        "max_class2_rate": 0.32,
        "max_class3_rate": 0.24,
        "max_class23_rate": 0.42,
    },
    {
        "name": "balanced_safe_pool",
        "min_cnt": 7,
        "min_class0_rate": 0.52,
        "min_lift": 1.02,
        "min_wilson_low": 0.28,
        "max_class1_rate": 0.58,
        "max_class2_rate": 0.40,
        "max_class3_rate": 0.32,
        "max_class23_rate": 0.58,
    },
    {
        "name": "coverage_diagnostic_pool",
        "min_cnt": 5,
        "min_class0_rate": 0.46,
        "min_lift": 0.92,
        "min_wilson_low": 0.22,
        "max_class1_rate": 0.68,
        "max_class2_rate": 0.48,
        "max_class3_rate": 0.40,
        "max_class23_rate": 0.68,
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
    """Pareto/scenario selection for class0 coverage with class23 protection.

    이 함수는 단일 결과만 억지로 만들지 않고, 여러 시나리오를 실행한 뒤 가장 균형 좋은 결과를 선택한다.
    """
    train_class0 = train["target_class"].to_numpy() == TARGET_CLASS
    valid_y = valid["target_class"].to_numpy()
    valid_class0 = valid_y == TARGET_CLASS
    valid_class1 = valid_y == 1
    valid_class2 = valid_y == 2
    valid_class3 = valid_y == 3
    valid_class23 = valid_class2 | valid_class3

    all_evaluated = []
    for i, (cnt, class0_cnt, class0_rate, lift, wilson, c1_rate, c2_rate, c3_rate, conds, score) in enumerate(rules, start=1):
        name = (
            f"target0_pareto_{i:04d}"
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
        # class23 penalty is strong, but not a hard exclusion here.
        stable_score = score * (1.0 - min(stability_gap, 0.45)) * max(0.12, 1.0 - va["class23_rate"] * 0.55 - va["class3_rate"] * 0.40)

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

    if not candidates:
        print("[STOP] no candidates.")
        return [], np.zeros(len(train), dtype=bool), np.zeros(len(valid), dtype=bool)

    valid_class0_total = int(valid_class0.sum())
    print(
        "[COVERAGE TARGETS]",
        f"valid_class0_total={valid_class0_total}",
        f"5.0pct={int(np.ceil(valid_class0_total * 0.050))}",
        f"5.5pct={int(np.ceil(valid_class0_total * 0.055))}",
        f"6.0pct={int(np.ceil(valid_class0_total * 0.060))}",
    )

    scenarios = [
        {
            "name": "precision_safe",
            "target_coverage": 0.050,
            "soft_coverage": 0.048,
            "max_class23": 0.180,
            "max_class3": 0.120,
            "min_rate_pre": 0.68,
            "min_rate_post": 0.68,
            "min_added_rate": 0.48,
            "min_added_rate_fill": 0.45,
            "max_added_c23": 0.34,
            "max_added_c23_fill": 0.38,
            "coverage_w": 250.0,
            "class0_w": 18.0,
            "c23_penalty": 105.0,
            "c3_penalty": 82.0,
            "added_c23_penalty": 8.5,
        },
        {
            "name": "class23_safe",
            "target_coverage": 0.053,
            "soft_coverage": 0.050,
            "max_class23": 0.190,
            "max_class3": 0.135,
            "min_rate_pre": 0.65,
            "min_rate_post": 0.66,
            "min_added_rate": 0.44,
            "min_added_rate_fill": 0.40,
            "max_added_c23": 0.38,
            "max_added_c23_fill": 0.44,
            "coverage_w": 330.0,
            "class0_w": 16.0,
            "c23_penalty": 88.0,
            "c3_penalty": 68.0,
            "added_c23_penalty": 6.8,
        },
        {
            "name": "balanced_safe_coverage",
            "target_coverage": 0.055,
            "soft_coverage": 0.052,
            "max_class23": 0.205,
            "max_class3": 0.155,
            "min_rate_pre": 0.62,
            "min_rate_post": 0.63,
            "min_added_rate": 0.40,
            "min_added_rate_fill": 0.36,
            "max_added_c23": 0.44,
            "max_added_c23_fill": 0.52,
            "coverage_w": 430.0,
            "class0_w": 14.0,
            "c23_penalty": 70.0,
            "c3_penalty": 52.0,
            "added_c23_penalty": 5.0,
        },
        {
            "name": "coverage_push_with_cap",
            "target_coverage": 0.060,
            "soft_coverage": 0.055,
            "max_class23": 0.215,
            "max_class3": 0.165,
            "min_rate_pre": 0.58,
            "min_rate_post": 0.60,
            "min_added_rate": 0.36,
            "min_added_rate_fill": 0.33,
            "max_added_c23": 0.50,
            "max_added_c23_fill": 0.60,
            "coverage_w": 560.0,
            "class0_w": 12.0,
            "c23_penalty": 52.0,
            "c3_penalty": 42.0,
            "added_c23_penalty": 3.8,
        },
    ]

    def run_scenario(scenario, verbose=True):
        selected = []
        selected_names = set()
        train_mask = np.zeros(len(train), dtype=bool)
        valid_mask = np.zeros(len(valid), dtype=bool)

        def choose_best(stage):
            cur = eval_mask(valid, valid_mask, label="VALID")
            cur_target = cur["class0_count"]
            cur_selected = cur["selected_count"]
            cur_cov = cur["class0_coverage"]

            best = None
            for cand in candidates:
                if cand["name"] in selected_names:
                    continue

                new_train = cand["train_mask"] & ~train_mask
                new_valid = cand["valid_mask"] & ~valid_mask
                vt = int(new_valid.sum())
                tt = int(new_train.sum())

                if vt < (5 if stage == "primary" else 3):
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
                if train_added_rate < max(0.30, min_added - 0.06):
                    continue
                if added_c23_rate > max_added_c23:
                    continue
                if added_c3_rate > min(0.40, scenario["max_class3"] + 0.18):
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

                min_rate = scenario["min_rate_pre"] if next_cov < scenario["soft_coverage"] else scenario["min_rate_post"]
                if next_rate < min_rate:
                    continue
                if next_c23_rate > scenario["max_class23"]:
                    continue
                if next_c3_rate > scenario["max_class3"]:
                    continue

                false_add = vt - va0
                score = (
                        va0 * (scenario["class0_w"] if stage == "primary" else scenario["class0_w"] + 5.0)
                        + vt * (0.70 if stage == "primary" else 0.90)
                        + valid_added_rate * (24.0 if stage == "primary" else 12.0)
                        + added_wilson * 8.0
                        + next_cov * scenario["coverage_w"]
                        + tr0 * 0.35
                        - false_add * (2.0 if stage == "primary" else 1.25)
                        - va23 * scenario["added_c23_penalty"]
                        - va3 * (scenario["added_c23_penalty"] + 1.5)
                        - next_c23_rate * scenario["c23_penalty"]
                        - next_c3_rate * scenario["c3_penalty"]
                        - next_c2_rate * (scenario["c23_penalty"] * 0.42)
                        + cand["stable_score"] * 0.02
                )

                key = (
                    score,
                    -next_c23_rate,
                    va0,
                    next_cov,
                    next_rate,
                    -next_c3_rate,
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

        for stage in ["primary", "fill"]:
            while True:
                if max_rules is not None and len(selected) >= max_rules:
                    break

                cur = eval_mask(valid, valid_mask, label="VALID")

                if stage == "fill":
                    if cur["class0_coverage"] >= scenario["target_coverage"]:
                        if verbose:
                            print(f"[{scenario['name']}] [FILL STOP] reached target coverage.")
                        break
                    if cur["class0_coverage"] >= scenario["soft_coverage"] and cur["class23_rate"] >= scenario["max_class23"] * 0.96:
                        if verbose:
                            print(f"[{scenario['name']}] [FILL STOP] soft coverage reached and class23 near cap.")
                        break

                best = choose_best(stage)
                if best is None:
                    if verbose:
                        print(f"[{scenario['name']}] [{stage.upper()} STOP] no acceptable rule.")
                    break

                cand = best["cand"]
                train_mask |= cand["train_mask"]
                valid_mask |= cand["valid_mask"]
                selected.append((cand["name"], cand["conds"]))
                selected_names.add(cand["name"])

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
                        f"class23={ev['class23_rate'] * 100:.2f}%",
                        f"class3={ev['class3_rate'] * 100:.2f}%",
                    )

        tr_eval = eval_mask(train, train_mask, label="TRAIN")
        va_eval = eval_mask(valid, valid_mask, label="VALID")
        return selected, train_mask, valid_mask, tr_eval, va_eval

    results = []
    for scenario in scenarios:
        print(f"\n[SCENARIO RUN] {scenario['name']}")
        sel, tm, vm, tr, va = run_scenario(scenario, verbose=True)

        score = (
                va["class0_rate"] * 100000
                + va["class0_coverage"] * 420000
                + np.log1p(va["selected_count"]) * 700
                - va["class23_rate"] * 70000
                - va["class3_rate"] * 50000
        )
        if va["class0_rate"] >= 0.68:
            score += 12000
        if va["class0_coverage"] >= 0.050:
            score += 6000
        if va["class0_coverage"] >= 0.055:
            score += 18000
        if va["class0_coverage"] >= 0.060:
            score += 10000
        if va["class23_rate"] <= 0.198:
            score += 16000
        if va["class23_rate"] <= 0.185:
            score += 8000
        if va["class3_rate"] <= 0.145:
            score += 7000

        results.append({
            "scenario": scenario,
            "selected": sel,
            "train_mask": tm,
            "valid_mask": vm,
            "train_eval": tr,
            "valid_eval": va,
            "score": score,
        })

        print(
            f"[SCENARIO RESULT] {scenario['name']} "
            f"rules={len(sel)} selected={va['selected_count']} "
            f"class0_rate={va['class0_rate'] * 100:.2f}% "
            f"coverage={va['class0_coverage'] * 100:.2f}% "
            f"class23={va['class23_rate'] * 100:.2f}% "
            f"class3={va['class3_rate'] * 100:.2f}% "
            f"score={score:.2f}"
        )

    print("\n[SCENARIO SUMMARY]")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        va = r["valid_eval"]
        print(
            f"{r['scenario']['name']:28s} "
            f"rules={len(r['selected']):3d} "
            f"selected={va['selected_count']:4d} "
            f"class0_rate={va['class0_rate'] * 100:6.2f}% "
            f"coverage={va['class0_coverage'] * 100:6.2f}% "
            f"class2={va['class2_rate'] * 100:6.2f}% "
            f"class3={va['class3_rate'] * 100:6.2f}% "
            f"class23={va['class23_rate'] * 100:6.2f}% "
            f"score={r['score']:10.2f}"
        )

    best = max(results, key=lambda x: x["score"])
    print(f"\n[SCENARIO SELECTED] {best['scenario']['name']}")

    return best["selected"], best["train_mask"], best["valid_mask"]


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
    valid_final_eval = eval_mask(valid, valid_mask, label="VALID COMBINED")
    print_eval(valid_final_eval)
    interpret_final_result(valid_final_eval)

    print("\n[OUTPUT]")
    print("rule_file:", out_path)
    print("report:", report_path)
    print("monthly_report:", monthly_path)

    return selected, train_mask, valid_mask


def interpret_final_result(valid_eval: dict):
    print("\n[INTERPRETATION]")
    rate = valid_eval["class0_rate"]
    coverage = valid_eval["class0_coverage"]
    selected = valid_eval["selected_count"]
    class2 = valid_eval["class2_rate"]
    class3 = valid_eval["class3_rate"]
    class23 = valid_eval["class23_rate"]

    if rate >= 0.68 and coverage >= 0.055 and class23 <= 0.198:
        print("type: preferred balanced candidate")
        print("해석: class0_rate, coverage, class23 보호의 균형이 좋습니다.")
    elif rate >= 0.68 and coverage >= 0.050 and class23 <= 0.185:
        print("type: low-class23 precision candidate")
        print("해석: class23는 낮고 class0_rate는 좋지만 coverage는 중간입니다.")
    elif rate >= 0.62 and coverage >= 0.060 and class23 <= 0.215:
        print("type: coverage-priority candidate")
        print("해석: coverage는 좋지만 class23 위험을 확인해야 합니다.")
    else:
        print("type: needs review")
        print("해석: 현재 feature/rule 조합에서 세 목표를 동시에 만족하기 어렵습니다.")

    print(f"valid_selected_count : {selected}")
    print(f"valid_class0_rate    : {rate * 100:.2f}%")
    print(f"valid_class0_coverage: {coverage * 100:.2f}%")
    print(f"valid_class2_rate    : {class2 * 100:.2f}%")
    print(f"valid_class3_rate    : {class3 * 100:.2f}%")
    print(f"valid_class23_rate   : {class23 * 100:.2f}%")

    if coverage < 0.055:
        print("[NOTE] coverage 5.5% 미만입니다. 커버리지 확대가 부족합니다.")
    if class23 > 0.198:
        print("[WARN] class23_rate가 19.8%를 넘습니다.")
    if class3 > 0.145:
        print("[WARN] class3_rate가 14.5%를 넘습니다.")


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
