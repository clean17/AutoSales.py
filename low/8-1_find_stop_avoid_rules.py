"""
stop_before_target_10 == 1 룰 생성 스크립트 - 적용 버전
“이 종목은 반등할 수도 있지만, 목표가 전에 손절이 먼저 나올 가능성이 높다.”

여러 시나리오와 precision fill을 비교해 coverage를 유지하면서 target_rate를 더 끌어올린다.

기준 결과:
    기존 안정형:
        VALID matched=546, target=336, target_rate=61.54%, target_coverage=9.15%
    적용 시나리오 coverage_keep_rate_loose:
        VALID matched=644, target=403, target_rate=62.58%, target_coverage=10.97%

목표:
    - target_rate를 유지/소폭 개선
    - target_coverage를 확대
    - 너무 얇은 70%대 소표본 룰로 과최적화하지 않음

사용:
    python find_stop_before_target_10_rules_applied.py
    python find_stop_before_target_10_rules_applied.py --csv csv/low_result_7_v2_desc_v2_desc.csv

생성:
    lowscan_stop_before_target_10_rules.py
    lowscan_stop_before_target_10_rule_report.csv
"""

import os, sys
import argparse
import heapq
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_features

# =============================================================================
# 기본 설정
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root

csv_dir = os.path.join(project_root, "csv")
os.makedirs(csv_dir, exist_ok=True)

CSV_PATH = os.path.join(csv_dir, "low_result_7_v2_desc.csv")
OUT_PATH = Path("job/lowscan_stop_before_target_10_rules_formatted.py")
REPORT_PATH = os.path.join(csv_dir, "lowscan_stop_before_target_10_rule_report.csv")
SCENARIO_REPORT_PATH = Path("csv/stop_avoid_10_scenario_report.csv")

TARGET_COL = "stop_before_target_10"
TARGET_VALUE = 1
DATE_COL = "today"

VALID_RATIO = 0.20

# 후보 탐색 파라미터: 기존 안정형 유지
BEAM = 30000
TOP_N = 10000
MIN_CNT = 80
MAX_DEPTH = 4

# 후보 룰 기준: 기존 안정형 유지
MIN_TARGET_RATE = 0.58
MIN_LIFT = 1.25

# valid 후보 통과 기준: 너무 빡세게 올리면 selected=0 위험
VALID_MIN_RATE = 0.60
VALID_MIN_CNT = 35
MAX_RATE_GAP = 0.14
VALID_MIN_LIFT = 1.15

# literal 생성 설정
N_QUANTILES = 14
MAX_UNIQUE_FOR_EQ = 8

# =============================================================================
# 최소수정 적용 피쳐 목록
# =============================================================================
# utils.get_features(train) 자동 전체 피쳐 선택 대신 아래 피쳐만 사용한다.
DEFAULT_FEATURES = [
    "vol5",
    "rebound_from_7d_low",
    # "today_pct",
    # "price_power_value",
    "dist_to_ma5",

    # "intraday_return",
    "tr_value_ratio_5d",
    "max_drop_7d",
    # "body_value_power",

    "upper_wick_ratio",
    "lower_wick_ratio",
    "vol15",
    "ATR_pct",
    "dist_to_ma20",

    "BB_perc",
    "gap_pct",
    "room_to_60d_high",
    "ma5_chg_rate",
    "pct_vs_lastweek",
]


# stability 점수 가중치
VALID_WEIGHT = 1.25
RATE_GAP_PENALTY_POWER = 2.0

# =============================================================================
# 최종 적용 시나리오: rate_up_keep_coverage 자동 선택
# =============================================================================
# 현재 결과가 coverage는 충분히 커졌지만 target_rate가 61% 초반에 머물렀으므로,
# 단일 loose 시나리오 대신 여러 시나리오를 돌린 뒤 coverage floor를 만족하는 조합 중
# valid target_rate가 가장 높은 조합을 최종 적용한다.

APPLIED_SCENARIOS = [
    {
        # depth4 안정형: V2에 가까운 정밀도와 coverage 균형을 노린다.
        "name": "depth4_precision68_stable_balance",
        "max_rules": 18,
        "min_added_total": 14,
        "min_added_target": 9,
        "min_added_valid_total": 9,
        "min_added_valid_target": 6,
        "min_train_added_rate": 0.600,
        "min_valid_added_rate": 0.610,
        "max_added_gap": 0.13,
        "min_next_valid_rate": 0.675,
        "max_rate_drop": 0.004,
        "objective": "rate_then_coverage",
    },
    {
        # depth4 정밀형: 69% 이상을 노리되 표본이 너무 작으면 제외.
        "name": "depth4_precision69_guard",
        "max_rules": 14,
        "min_added_total": 16,
        "min_added_target": 10,
        "min_added_valid_total": 10,
        "min_added_valid_target": 7,
        "min_train_added_rate": 0.615,
        "min_valid_added_rate": 0.630,
        "max_added_gap": 0.12,
        "min_next_valid_rate": 0.685,
        "max_rate_drop": 0.003,
        "objective": "rate_then_coverage",
    },
    {
        # coverage 보존형이지만 65% 아래로 내려가지 않도록 방어.
        "name": "depth4_coverage_keep_rate_guard",
        "max_rules": 20,
        "min_added_total": 12,
        "min_added_target": 7,
        "min_added_valid_total": 8,
        "min_added_valid_target": 5,
        "min_train_added_rate": 0.585,
        "min_valid_added_rate": 0.595,
        "max_added_gap": 0.14,
        "min_next_valid_rate": 0.660,
        "max_rate_drop": 0.006,
        "objective": "rate_then_coverage",
    },
    {
        # V2 개선 목표: 정확도 68% 이상, coverage 13.5% 이상을 노리는 균형형.
        "name": "precision68_v2_improve_balance",
        "max_rules": 20,
        "min_added_total": 12,
        "min_added_target": 8,
        "min_added_valid_total": 7,
        "min_added_valid_target": 5,
        "min_train_added_rate": 0.600,
        "min_valid_added_rate": 0.620,
        "max_added_gap": 0.16,
        "min_next_valid_rate": 0.680,
        "max_rate_drop": 0.004,
        "objective": "rate_then_coverage",
    },
    {
        # 조금 더 정밀한 69% 목표. coverage는 줄어들 수 있음.
        "name": "precision69_strict_balance",
        "max_rules": 16,
        "min_added_total": 14,
        "min_added_target": 9,
        "min_added_valid_total": 8,
        "min_added_valid_target": 6,
        "min_train_added_rate": 0.615,
        "min_valid_added_rate": 0.640,
        "max_added_gap": 0.15,
        "min_next_valid_rate": 0.690,
        "max_rate_drop": 0.003,
        "objective": "rate_then_coverage",
    },
    {
        # 기존 66.89%보다 약간 높은 67.5%를 유지하면서 coverage를 더 확보하는 fallback 균형형.
        "name": "precision675_coverage_keep",
        "max_rules": 24,
        "min_added_total": 10,
        "min_added_target": 6,
        "min_added_valid_total": 6,
        "min_added_valid_target": 4,
        "min_train_added_rate": 0.590,
        "min_valid_added_rate": 0.600,
        "max_added_gap": 0.18,
        "min_next_valid_rate": 0.675,
        "max_rate_drop": 0.005,
        "objective": "rate_then_coverage",
    },
    {
        "name": "rate_up_strict_keep_coverage",
        "max_rules": 12,
        "min_added_total": 18,
        "min_added_target": 11,
        "min_added_valid_total": 12,
        "min_added_valid_target": 7,
        "min_train_added_rate": 0.595,
        "min_valid_added_rate": 0.605,
        "max_added_gap": 0.16,
        "min_next_valid_rate": 0.615,
        "max_rate_drop": 0.004,
        "objective": "rate_then_coverage",
    },
    {
        "name": "rate_up_balanced_keep_coverage",
        "max_rules": 12,
        "min_added_total": 16,
        "min_added_target": 9,
        "min_added_valid_total": 10,
        "min_added_valid_target": 6,
        "min_train_added_rate": 0.585,
        "min_valid_added_rate": 0.590,
        "max_added_gap": 0.18,
        "min_next_valid_rate": 0.610,
        "max_rate_drop": 0.006,
        "objective": "rate_then_coverage",
    },
    {
        "name": "rate_up_soft_keep_coverage",
        "max_rules": 12,
        "min_added_total": 15,
        "min_added_target": 8,
        "min_added_valid_total": 10,
        "min_added_valid_target": 5,
        "min_train_added_rate": 0.575,
        "min_valid_added_rate": 0.580,
        "max_added_gap": 0.19,
        "min_next_valid_rate": 0.605,
        "max_rate_drop": 0.008,
        "objective": "rate_then_coverage",
    },
    {
        "name": "rate_up_balanced_more_rules",
        "max_rules": 18,
        "min_added_total": 14,
        "min_added_target": 8,
        "min_added_valid_total": 8,
        "min_added_valid_target": 5,
        "min_train_added_rate": 0.580,
        "min_valid_added_rate": 0.585,
        "max_added_gap": 0.19,
        "min_next_valid_rate": 0.608,
        "max_rate_drop": 0.006,
        "objective": "rate_then_coverage",
    },
    {
        "name": "rate_up_soft_more_rules",
        "max_rules": 20,
        "min_added_total": 12,
        "min_added_target": 7,
        "min_added_valid_total": 8,
        "min_added_valid_target": 5,
        "min_train_added_rate": 0.570,
        "min_valid_added_rate": 0.575,
        "max_added_gap": 0.20,
        "min_next_valid_rate": 0.605,
        "max_rate_drop": 0.008,
        "objective": "rate_then_coverage",
    },
    {
        # strict로 시작하되 룰 수를 늘려 coverage를 회복하는 시나리오.
        # 목표: balanced_more_rules보다 rate를 올리면서 coverage 90% 안팎 유지.
        "name": "rate_up_strict_more_rules_fill",
        "max_rules": 18,
        "min_added_total": 12,
        "min_added_target": 7,
        "min_added_valid_total": 6,
        "min_added_valid_target": 4,
        "min_train_added_rate": 0.585,
        "min_valid_added_rate": 0.600,
        "max_added_gap": 0.18,
        "min_next_valid_rate": 0.620,
        "max_rate_drop": 0.003,
        "objective": "rate_then_coverage",
    },
    {
        # coverage를 더 유지하되, 후반부 fill 룰의 valid precision을 59.5% 이상으로 제한.
        "name": "rate_up_precision_fill_keep_coverage",
        "max_rules": 22,
        "min_added_total": 10,
        "min_added_target": 6,
        "min_added_valid_total": 6,
        "min_added_valid_target": 4,
        "min_train_added_rate": 0.580,
        "min_valid_added_rate": 0.595,
        "max_added_gap": 0.19,
        "min_next_valid_rate": 0.618,
        "max_rate_drop": 0.0035,
        "objective": "rate_then_coverage",
    },
    {
        # 11.5% 이상 coverage 구간에서 rate를 더 밀어보는 시나리오.
        "name": "rate_up_high_coverage_guard",
        "max_rules": 24,
        "min_added_total": 8,
        "min_added_target": 5,
        "min_added_valid_total": 6,
        "min_added_valid_target": 4,
        "min_train_added_rate": 0.575,
        "min_valid_added_rate": 0.590,
        "max_added_gap": 0.20,
        "min_next_valid_rate": 0.616,
        "max_rate_drop": 0.003,
        "objective": "rate_then_coverage",
    },
    {
        # fallback: 기존 coverage_keep_rate_loose와 거의 동일
        "name": "coverage_keep_rate_loose_fallback",
        "max_rules": 12,
        "min_added_total": 15,
        "min_added_target": 8,
        "min_added_valid_total": 10,
        "min_added_valid_target": 5,
        "min_train_added_rate": 0.57,
        "min_valid_added_rate": 0.565,
        "max_added_gap": 0.20,
        "min_next_valid_rate": 0.595,
        "max_rate_drop": 0.012,
        "objective": "coverage",
    },
]

# 자동 선택 기준.
# 이전 버전은 coverage floor(0.115 / matched 650)를 못 넘으면 fallback을 고르도록 되어 있어
# rate-up 시나리오가 실제로는 버려지는 문제가 있었다.
#
# 이제는 "최대 coverage 조합"을 baseline으로 잡고,
# coverage를 일정 비율 이상 유지하면서 rate가 유의미하게 높은 조합을 우선 선택한다.
# 예: baseline coverage 12.44%라면 0.82 유지 기준은 약 10.20%.
MIN_RATE_GAIN_OVER_COVERAGE_BASELINE = 0.006

# 기본 허용 하한: rate-up 후보로 인정할 최소 coverage/matched 유지율.
MIN_COVERAGE_KEEP_RATIO = 0.82
MIN_MATCHED_KEEP_RATIO = 0.72

# 선호 구간: "coverage는 되도록 유지" 목적에 맞춰,
# 최대 coverage baseline 대비 90% 이상 유지하는 rate-up 후보를 먼저 고른다.
# 이번 scenario_report 기준으로는
#   fallback: 61.42% / coverage 12.44%
#   balanced_more_rules: 62.80% / coverage 11.49% / keep 92.34%
# 이 조합이 여기에 들어온다.
PREFERRED_COVERAGE_KEEP_RATIO = 0.90
PREFERRED_MATCHED_KEEP_RATIO = 0.88

ABS_MIN_VALID_COVERAGE = 0.08
ABS_MIN_VALID_MATCHED = 300

# 너무 작은 최종 결과 방지용 경고 기준
WARN_MIN_VALID_MATCHED = 300
WARN_MIN_VALID_COVERAGE = 0.08
WARN_MIN_VALID_RATE = 0.60


# V2 개선형 선택 기준:
# - depth 5 과적합을 피하기 위해 depth 4를 사용한다.
# - 목표는 현재 66.89%보다 정확도를 올리되, V2 근처 coverage를 유지하는 것.
# - 68% 이상 + coverage 12.5% 이상 후보가 있으면 우선 선택한다.
PREFERRED_MIN_VALID_RATE = 0.68
PREFERRED_MIN_VALID_COVERAGE = 0.125
PREFERRED_MIN_VALID_MATCHED = 550

STRICT_MIN_VALID_RATE = 0.69
STRICT_MIN_VALID_COVERAGE = 0.110
STRICT_MIN_VALID_MATCHED = 500

FALLBACK_MIN_VALID_RATE = 0.675
FALLBACK_MIN_VALID_COVERAGE = 0.100
FALLBACK_MIN_VALID_MATCHED = 450

# =============================================================================
# 데이터 / 피쳐 유틸
# =============================================================================

def get_exclude_columns(df=None):
    exclude = {
        "ticker", "stock_name", "today", "idx", "sector_code",
        "stop_loss", "stop_day", "target_pct", "target_class",
        "_close_pos_20d", "_tr_value_ratio", "_tr_value_ratio_5d",
        "_dist_to_high_20d", "_BB_perc", "_UltimateOsc", "_CCI14",
        "_ADX14", "_gap_pct", "_vol_ratio_15_60", "_RSI_rebound",
        "_rebound_power", "_MACD_hist_1d", "_MACD_acc",
        "_MACD_hist_3d_close_norm",
    }

    if df is not None:
        for c in df.columns:
            if (
                    c == TARGET_COL
                    or c.startswith("validation_")
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


def get_feature_groups():
    """DEFAULT_FEATURES에 포함된 피쳐만 그룹 제한에 반영한다."""
    feature_groups = {
        "vol5": "VOLATILITY",
        "vol15": "VOLATILITY",
        "ATR_pct": "VOLATILITY",

        "rebound_from_7d_low": "REBOUND",

        "dist_to_ma5": "POSITION",
        "dist_to_ma20": "POSITION",

        "tr_value_ratio_5d": "VOLUME",

        "max_drop_7d": "DROP",

        "upper_wick_ratio": "CANDLE",
        "lower_wick_ratio": "CANDLE",

        "BB_perc": "BAND",
        "gap_pct": "GAP",
        "room_to_60d_high": "HIGH_ROOM",
        "ma5_chg_rate": "TREND",
        "pct_vs_lastweek": "WEEK_POSITION",
    }

    group_limits = {
        "VOLATILITY": 3,
        "POSITION": 2,
        "CANDLE": 2,
        "REBOUND": 1,
        "VOLUME": 1,
        "DROP": 1,
        "BAND": 1,
        "GAP": 1,
        "HIGH_ROOM": 1,
        "TREND": 1,
        "WEEK_POSITION": 1,
    }

    return feature_groups, group_limits



def split_train_valid_by_date_ratio(df, valid_ratio=VALID_RATIO, date_col=DATE_COL):
    if date_col not in df.columns:
        raise ValueError(f"{date_col} 컬럼이 없습니다.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    unique_dates = np.array(sorted(out[date_col].dropna().unique()))
    if len(unique_dates) < 2:
        raise ValueError("날짜 종류가 너무 적어서 train/valid 분리를 할 수 없습니다.")

    split_idx = int(len(unique_dates) * (1 - valid_ratio))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = unique_dates[split_idx]

    train = out[out[date_col] < split_date].reset_index(drop=True)
    valid = out[out[date_col] >= split_date].reset_index(drop=True)

    return train, valid, split_date


def select_default_features(df, default_features=DEFAULT_FEATURES):
    """DEFAULT_FEATURES 중 실제 데이터에 있고 숫자로 변환 가능한 피쳐만 사용한다."""
    selected = []
    missing = []
    non_numeric_or_empty = []

    for feat in default_features:
        if feat not in df.columns:
            missing.append(feat)
            continue

        s = pd.to_numeric(df[feat], errors="coerce")
        finite_count = int(np.isfinite(s.to_numpy(dtype=float, na_value=np.nan)).sum())
        unique_count = int(s.replace([np.inf, -np.inf], np.nan).dropna().nunique())

        if finite_count == 0 or unique_count < 2:
            non_numeric_or_empty.append(feat)
            continue

        selected.append(feat)

    if missing:
        print(f"[WARN] DEFAULT_FEATURES missing skipped: {missing}")
    if non_numeric_or_empty:
        print(f"[WARN] DEFAULT_FEATURES non-numeric/empty skipped: {non_numeric_or_empty}")

    return selected


def build_literals(df, features, min_count=MIN_CNT, n_quantiles=N_QUANTILES):
    literals = []
    literal_masks = []

    for feat in features:
        s = pd.to_numeric(df[feat], errors="coerce")
        arr = s.to_numpy()
        notna = ~pd.isna(arr)

        if not notna.any():
            continue

        uniq = np.sort(pd.unique(s.dropna()))
        uniq = uniq[np.isfinite(uniq)]

        if len(uniq) == 0:
            continue

        if len(uniq) <= MAX_UNIQUE_FOR_EQ:
            for v in uniq:
                mask = notna & (arr == v)
                if int(mask.sum()) >= min_count:
                    literals.append((feat, "==", float(v)))
                    literal_masks.append(mask)
            continue

        qs = np.linspace(0.10, 0.90, n_quantiles - 1)
        thresholds = np.unique(np.nanquantile(arr, qs))
        thresholds = thresholds[np.isfinite(thresholds)]

        for t in thresholds:
            le_mask = notna & (arr <= t)
            ge_mask = notna & (arr >= t)

            if int(le_mask.sum()) >= min_count:
                literals.append((feat, "<=", float(t)))
                literal_masks.append(le_mask)

            if int(ge_mask.sum()) >= min_count:
                literals.append((feat, ">=", float(t)))
                literal_masks.append(ge_mask)

    return literals, literal_masks


def make_mask_from_conds(df, conds):
    mask = np.ones(len(df), dtype=bool)

    for feat, op, val in conds:
        if feat not in df.columns:
            mask &= False
            continue

        s = pd.to_numeric(df[feat], errors="coerce")
        arr = s.to_numpy()
        notna = ~pd.isna(arr)

        if op == "<=":
            mask &= notna & (arr <= val)
        elif op == ">=":
            mask &= notna & (arr >= val)
        elif op == "==":
            mask &= notna & (arr == val)
        else:
            raise ValueError(f"지원하지 않는 op: {op}")

    return mask


# =============================================================================
# 룰 파일 출력
# =============================================================================

def _cond_to_code(cond):
    feat, op, val = cond

    if op in ("<=", ">=", "=="):
        return f'(df["{feat}"] {op} {val:.10g})'

    raise ValueError(f"지원하지 않는 op: {op}")


def write_rule_file(path, selected, header_comment=""):
    """RULE_NAMES + build_conditions(df) + build_mask(df) 형식의 룰 파일을 생성한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rule_names = [f"rule_{i:03d}" for i in range(1, len(selected) + 1)]

    lines = []
    lines.append(header_comment.rstrip())
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("RULE_NAMES = [")
    for name in rule_names:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")
    lines.append("def build_conditions(df):")
    lines.append("    conditions = {")

    for rule_name, rule in zip(rule_names, selected):
        conds = rule["conds"]
        lines.append(f'        "{rule_name}":')
        for j, cond in enumerate(conds):
            suffix = " &" if j < len(conds) - 1 else ","
            lines.append(f"            {_cond_to_code(cond)}{suffix}")

    lines.append("    }")
    lines.append("    return conditions")
    lines.append("")
    lines.append("def build_mask(df):")
    lines.append("    mask = np.zeros(len(df), dtype=bool)")
    lines.append("    for cond in build_conditions(df).values():")
    lines.append("        mask |= cond")
    lines.append("    return mask")
    lines.append("")
    lines.append("def build_rule_name_series(df, sep=\",\"):")
    lines.append("    conditions = build_conditions(df)")
    lines.append("    names = []")
    lines.append("    for i in range(len(df)):")
    lines.append("        matched = [")
    lines.append("            name")
    lines.append("            for name, cond in conditions.items()")
    lines.append("            if bool(cond.iloc[i] if hasattr(cond, \"iloc\") else cond[i])")
    lines.append("        ]")
    lines.append("        names.append(sep.join(matched))")
    lines.append("    return names")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# 평가 유틸
# =============================================================================

def eval_mask(df, mask, label=""):
    y = (df[TARGET_COL] == TARGET_VALUE).to_numpy()

    matched_count = int(mask.sum())
    matched_target = int((mask & y).sum())
    total_target = int(y.sum())

    target_rate = matched_target / matched_count if matched_count else 0.0
    target_coverage = matched_target / total_target if total_target else 0.0

    return {
        "label": label,
        "total_count": int(len(df)),
        "total_target": total_target,
        "matched_count": matched_count,
        "matched_target": matched_target,
        "target_rate": target_rate,
        "target_coverage": target_coverage,
    }


def print_eval(row):
    print(
        f"[{row['label']}] "
        f"matched={row['matched_count']}/{row['total_count']} "
        f"target={row['matched_target']}/{row['total_target']} "
        f"target_rate={row['target_rate'] * 100:.2f}% "
        f"target_coverage={row['target_coverage'] * 100:.2f}%"
    )


def combined_mask(df, selected):
    mask = np.zeros(len(df), dtype=bool)

    for rule in selected:
        mask |= make_mask_from_conds(df, rule["conds"])

    return mask


def _rule_key_from_conds(conds):
    return tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in conds))


# =============================================================================
# 룰 탐색
# =============================================================================

def mine_target_rules(
        df,
        literals,
        literal_masks,
        y,
        min_count=MIN_CNT,
        max_depth=MAX_DEPTH,
        beam=BEAM,
        top_n=TOP_N,
        feature_groups=None,
        group_limits=None,
):
    beams = [(np.ones(len(df), dtype=bool), [])]
    good = {}

    base_rate = float(y.mean()) if len(y) else 0.0
    min_rate = max(MIN_TARGET_RATE, base_rate * MIN_LIFT)

    print(
        "\n[TARGET RULE MINING]"
        f"\ntarget: {TARGET_COL} == {TARGET_VALUE}"
        f"\nbase_rate: {base_rate:.4f}"
        f"\nbeam: {beam}"
        f"\ntop_n: {top_n}"
        f"\nmin_count: {min_count}"
        f"\nmax_depth: {max_depth}"
        f"\nmin_target_rate: {MIN_TARGET_RATE}"
        f"\nmin_lift: {MIN_LIFT}"
        f"\neffective_min_rate: {min_rate:.4f}"
        "\n"
    )

    if feature_groups is None:
        feature_groups = {}
    if group_limits is None:
        group_limits = {}

    def get_group(feat_name):
        return feature_groups.get(feat_name)

    for depth in range(max_depth):
        print("----------------------------------")
        print("[TARGET] depth", depth + 1)

        heap = []
        uid = count()

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

            group_used = {}
            for f in used_feats:
                g = get_group(f)
                if g is not None:
                    group_used[g] = group_used.get(g, 0) + 1

            for lit, lmask in zip(literals, literal_masks):
                feat = lit[0]

                if feat in used_feats:
                    continue

                g = get_group(feat)
                if g is not None:
                    limit = group_limits.get(g)
                    if limit is not None and group_used.get(g, 0) >= limit:
                        continue

                m = base_mask & lmask
                cnt = int(m.sum())

                if cnt < min_count:
                    continue

                target_cnt = int((m & y).sum())
                target_rate = target_cnt / cnt
                lift = target_rate / base_rate if base_rate else 0.0

                score = (target_rate ** 2.0) * np.log1p(cnt) * max(lift, 0.01)
                new_conds = conds + [lit]

                if target_rate >= min_rate:
                    key2 = _rule_key_from_conds(new_conds)
                    prev = good.get(key2)
                    if prev is None or score > prev["train_score"]:
                        good[key2] = {
                            "train_count": cnt,
                            "train_target": target_cnt,
                            "train_rate": target_rate,
                            "train_lift": lift,
                            "conds": new_conds,
                            "train_score": score,
                        }

                expand_min_rate = max(
                    base_rate * (1.08 + depth * 0.08),
                    base_rate + 0.04 + depth * 0.04,
                    )
                expand_min_rate = min(expand_min_rate, min_rate)

                if target_rate >= expand_min_rate:
                    k = (score, target_rate, target_cnt, cnt)
                    item = (k, next(uid), m, new_conds, target_rate, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        if k <= heap[0][0]:
                            continue
                        heapq.heapreplace(heap, item)

        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("[TARGET] expandable candidates:", len(new))

        if not new:
            print("[TARGET] no expandable candidates; stopping.")
            break

        tail = new[-1]
        print(
            "[TARGET] tail rate:",
            round(tail[4], 3),
            "cnt:",
            tail[5],
            "conds:",
            tail[3],
        )

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    out = sorted(
        good.values(),
        key=lambda x: (-x["train_score"], -x["train_rate"], -x["train_count"]),
    )

    if top_n is not None:
        out = out[:top_n]

    return out


# =============================================================================
# Valid 평가
# =============================================================================

def attach_validation_metrics(train, valid, rules):
    train_base = float((train[TARGET_COL] == TARGET_VALUE).mean())
    valid_base = float((valid[TARGET_COL] == TARGET_VALUE).mean())

    out = []

    for i, rule in enumerate(rules, start=1):
        conds = rule["conds"]

        train_mask = make_mask_from_conds(train, conds)
        valid_mask = make_mask_from_conds(valid, conds)

        tr = eval_mask(train, train_mask, "train")
        va = eval_mask(valid, valid_mask, "valid")

        train_rate = tr["target_rate"]
        valid_rate = va["target_rate"]
        rate_gap = abs(train_rate - valid_rate)
        generalized_rate = min(train_rate, valid_rate)
        valid_lift = valid_rate / valid_base if valid_base else 0.0
        train_lift = train_rate / train_base if train_base else 0.0

        stability_score = (
                generalized_rate
                * np.log1p(tr["matched_count"])
                * np.log1p(va["matched_count"])
                * max(train_lift, 0.01)
                * max(valid_lift, 0.01) ** VALID_WEIGHT
                * max(1.0 - rate_gap, 0.01) ** RATE_GAP_PENALTY_POWER
        )

        valid_pass = (
                tr["matched_count"] >= MIN_CNT
                and va["matched_count"] >= VALID_MIN_CNT
                and train_rate >= MIN_TARGET_RATE
                and valid_rate >= VALID_MIN_RATE
                and train_lift >= MIN_LIFT
                and valid_lift >= VALID_MIN_LIFT
                and rate_gap <= MAX_RATE_GAP
        )

        name = (
            f"rule_{i:03d}"
            f"_st{stability_score:.2f}"
            f"_tr{train_rate:.3f}"
            f"_va{valid_rate:.3f}"
            f"_n{tr['matched_count']}"
            f"_v{va['matched_count']}"
        )

        out.append({
            "name": name,
            "conds": conds,
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "train_count": tr["matched_count"],
            "train_target": tr["matched_target"],
            "train_rate": train_rate,
            "train_lift": train_lift,
            "train_coverage": tr["target_coverage"],
            "valid_count": va["matched_count"],
            "valid_target": va["matched_target"],
            "valid_rate": valid_rate,
            "valid_lift": valid_lift,
            "valid_coverage": va["target_coverage"],
            "rate_gap": rate_gap,
            "generalized_rate": generalized_rate,
            "stability_score": stability_score,
            "valid_pass": valid_pass,
        })

    out.sort(
        key=lambda x: (
            -x["stability_score"],
            -x["generalized_rate"],
            -x["valid_count"],
        )
    )

    return out


# =============================================================================
# 적용 시나리오 selection
# =============================================================================

def select_rules_applied(candidates, y_train, y_valid, scenario=None, verbose=True):
    if scenario is None:
        scenario = APPLIED_SCENARIOS[0]

    usable = [c for c in candidates if c["valid_pass"]]

    if not usable:
        return []

    n_train = len(usable[0]["train_mask"])
    n_valid = len(usable[0]["valid_mask"])

    train_combined = np.zeros(n_train, dtype=bool)
    valid_combined = np.zeros(n_valid, dtype=bool)

    selected = []
    selected_names = set()

    while True:
        best = None

        current_valid_target = int((valid_combined & y_valid).sum())
        current_valid_total = int(valid_combined.sum())
        current_valid_rate = (
            current_valid_target / current_valid_total
            if current_valid_total
            else 0.0
        )

        for cand in usable:
            if cand["name"] in selected_names:
                continue

            train_new = cand["train_mask"] & ~train_combined
            valid_new = cand["valid_mask"] & ~valid_combined

            train_added_total = int(train_new.sum())
            valid_added_total = int(valid_new.sum())

            if train_added_total < scenario["min_added_total"]:
                continue
            if valid_added_total < scenario["min_added_valid_total"]:
                continue

            train_added_target = int((train_new & y_train).sum())
            valid_added_target = int((valid_new & y_valid).sum())

            if train_added_target < scenario["min_added_target"]:
                continue
            if valid_added_target < scenario["min_added_valid_target"]:
                continue

            train_added_rate = train_added_target / train_added_total
            valid_added_rate = valid_added_target / valid_added_total
            added_gap = abs(train_added_rate - valid_added_rate)

            if train_added_rate < scenario["min_train_added_rate"]:
                continue
            if valid_added_rate < scenario["min_valid_added_rate"]:
                continue
            if added_gap > scenario["max_added_gap"]:
                continue

            next_valid_target = current_valid_target + valid_added_target
            next_valid_total = current_valid_total + valid_added_total
            next_valid_rate = next_valid_target / next_valid_total if next_valid_total else 0.0

            if next_valid_rate < scenario["min_next_valid_rate"]:
                continue
            if selected and next_valid_rate < current_valid_rate - scenario["max_rate_drop"]:
                continue

            train_false = train_added_total - train_added_target
            valid_false = valid_added_total - valid_added_target

            coverage_gain = valid_added_target
            rate_quality = valid_added_rate
            stability = min(train_added_rate, valid_added_rate)
            gap_quality = max(1.0 - added_gap, 0.01)

            if scenario.get("objective") == "rate_then_coverage":
                # target_rate 개선용 점수식.
                # coverage_gain은 유지하되, valid_added_rate / next_valid_rate / false positive 패널티를 더 크게 둔다.
                score = (
                        next_valid_rate * 160.0
                        + rate_quality * 90.0
                        + stability * 35.0
                        + coverage_gain * 1.15
                        + valid_added_total * 0.05
                        - valid_false * 0.90
                        - train_false * 0.14
                        + cand["stability_score"] * 0.05
                )
                score *= gap_quality ** 2.0
            else:
                # coverage_keep_rate_loose fallback 점수식.
                # coverage_gain에 큰 가중치를 주되, 낮은 precision 룰이 들어오지 않도록 rate_quality도 반영.
                score = (
                        coverage_gain * 2.0
                        + valid_added_total * 0.15
                        + rate_quality * 20.0
                        + stability * 10.0
                        - valid_false * 0.35
                        - train_false * 0.10
                        + cand["stability_score"] * 0.08
                )
                score *= gap_quality ** 1.5

            key = (
                score,
                next_valid_rate,
                valid_added_rate,
                valid_added_target,
                -valid_false,
                cand["stability_score"],
            )

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "cand": cand,
                    "train_added_total": train_added_total,
                    "train_added_target": train_added_target,
                    "train_added_rate": train_added_rate,
                    "valid_added_total": valid_added_total,
                    "valid_added_target": valid_added_target,
                    "valid_added_rate": valid_added_rate,
                    "added_gap": added_gap,
                    "next_valid_rate": next_valid_rate,
                }

        if best is None:
            break

        cand = best["cand"]

        train_combined |= cand["train_mask"]
        valid_combined |= cand["valid_mask"]

        selected.append(cand)
        selected_names.add(cand["name"])

        if verbose:
            print(
                f"[SELECT] {len(selected):03d} {cand['name']} "
                f"train_add={best['train_added_target']}/{best['train_added_total']} "
                f"({best['train_added_rate'] * 100:.2f}%) "
                f"valid_add={best['valid_added_target']}/{best['valid_added_total']} "
                f"({best['valid_added_rate'] * 100:.2f}%) "
                f"next_valid_rate={best['next_valid_rate'] * 100:.2f}% "
                f"gap={best['added_gap']:.3f}"
            )

        if len(selected) >= scenario["max_rules"]:
            break

    return selected



def select_best_scenario(candidates, train, valid, y_train, y_valid, scenarios=APPLIED_SCENARIOS):
    """여러 selection 시나리오를 평가하고 최종 조합을 선택한다.

    v4 선택 원칙:
    - 이전 버전은 절대 coverage floor를 만족하지 못하면 fallback에 큰 보너스를 줘서
      target_rate가 높은 시나리오가 있어도 선택하지 못했다.
    - 이제는 실행된 시나리오 중 valid_coverage가 가장 큰 조합을 coverage baseline으로 잡는다.
    - baseline 대비 coverage/matched를 일정 비율 이상 유지하면서 valid_rate가 유의미하게 높으면
      그 시나리오를 우선 선택한다.
    - 즉, "coverage는 되도록 유지"하되 "개선이 없으면 fallback"이 아니라
      실제 rate 개선 조합을 선택한다.
    """
    rows = []
    evaluated = []

    for scenario in scenarios:
        print(f"\n[SCENARIO RUN] {scenario['name']}")
        selected = select_rules_applied(
            candidates=candidates,
            y_train=y_train,
            y_valid=y_valid,
            scenario=scenario,
            verbose=True,
        )

        train_mask = combined_mask(train, selected)
        valid_mask = combined_mask(valid, selected)
        train_eval = eval_mask(train, train_mask, "TRAIN")
        valid_eval = eval_mask(valid, valid_mask, "VALID")

        row = {
            "scenario": scenario["name"],
            "selected_rules": len(selected),
            "train_matched": train_eval["matched_count"],
            "train_target": train_eval["matched_target"],
            "train_rate": train_eval["target_rate"],
            "train_coverage": train_eval["target_coverage"],
            "valid_matched": valid_eval["matched_count"],
            "valid_target": valid_eval["matched_target"],
            "valid_rate": valid_eval["target_rate"],
            "valid_coverage": valid_eval["target_coverage"],
            "coverage_ok": False,  # 아래에서 baseline 기반으로 다시 계산
            "rate_gain_vs_coverage_base": 0.0,
            "coverage_keep_ratio": 0.0,
            "matched_keep_ratio": 0.0,
            "score": 0.0,
        }
        rows.append(row)
        evaluated.append({
            "row": row,
            "selected": selected,
            "scenario": scenario,
            "train_eval": train_eval,
            "valid_eval": valid_eval,
        })

        print_eval(train_eval)
        print_eval(valid_eval)

    if not evaluated:
        return [], None, pd.DataFrame(rows)

    # coverage가 가장 큰 결과를 현재 coverage baseline으로 본다.
    coverage_base = max(
        evaluated,
        key=lambda x: (
            x["valid_eval"]["target_coverage"],
            x["valid_eval"]["matched_count"],
            x["valid_eval"]["target_rate"],
        ),
    )
    base_valid = coverage_base["valid_eval"]
    base_rate = base_valid["target_rate"]
    base_coverage = base_valid["target_coverage"]
    base_matched = base_valid["matched_count"]

    min_keep_coverage = max(ABS_MIN_VALID_COVERAGE, base_coverage * MIN_COVERAGE_KEEP_RATIO)
    min_keep_matched = max(ABS_MIN_VALID_MATCHED, int(base_matched * MIN_MATCHED_KEEP_RATIO))

    print("\n[SCENARIO BASELINE]")
    print(
        f"coverage_baseline={coverage_base['scenario']['name']} "
        f"valid_rate={base_rate * 100:.2f}% "
        f"valid_coverage={base_coverage * 100:.2f}% "
        f"valid_matched={base_matched}"
    )
    print(
        f"rate-up accept if: valid_rate >= {(base_rate + MIN_RATE_GAIN_OVER_COVERAGE_BASELINE) * 100:.2f}% "
        f"and valid_coverage >= {min_keep_coverage * 100:.2f}% "
        f"and valid_matched >= {min_keep_matched}"
    )

    accepted = []
    for item in evaluated:
        valid_eval = item["valid_eval"]
        row = item["row"]

        rate_gain = valid_eval["target_rate"] - base_rate
        coverage_keep_ratio = (
            valid_eval["target_coverage"] / base_coverage
            if base_coverage else 0.0
        )
        matched_keep_ratio = (
            valid_eval["matched_count"] / base_matched
            if base_matched else 0.0
        )

        coverage_ok = (
                valid_eval["target_coverage"] >= min_keep_coverage
                and valid_eval["matched_count"] >= min_keep_matched
        )
        rate_up_ok = rate_gain >= MIN_RATE_GAIN_OVER_COVERAGE_BASELINE

        row["coverage_ok"] = coverage_ok
        row["rate_gain_vs_coverage_base"] = rate_gain
        row["coverage_keep_ratio"] = coverage_keep_ratio
        row["matched_keep_ratio"] = matched_keep_ratio

        # target_rate 개선을 가장 크게 보되, coverage/matched도 보상.
        # coverage baseline 자체는 개선이 없으면 안전 fallback으로 남게 한다.
        score = (
                valid_eval["target_rate"] * 100000.0
                + valid_eval["target_coverage"] * 5000.0
                + np.log1p(valid_eval["matched_count"]) * 100.0
        )

        if coverage_ok:
            score += 50000.0
        else:
            score -= max(0.0, min_keep_coverage - valid_eval["target_coverage"]) * 200000.0
            score -= max(0, min_keep_matched - valid_eval["matched_count"]) * 5.0

        if coverage_ok and rate_up_ok:
            # 실제 개선 후보에는 큰 보너스. 이전 버전과 달리 fallback이 압도하지 못하게 한다.
            score += 200000.0 + rate_gain * 500000.0
            accepted.append(item)

        row["score"] = score

    # -----------------------------------------------------------------
    # V2 개선형 export 선택
    # -----------------------------------------------------------------
    # 현재 66.89% / 17.51%는 coverage는 좋지만 정확도가 V2보다 낮다.
    # 기대수익률 개선 목적이면 68% 이상을 우선 잡고, coverage를 최소 13.5% 이상 유지한다.
    preferred_candidates = []
    strict_candidates = []
    fallback_precision_candidates = []

    for item in evaluated:
        valid_eval = item["valid_eval"]

        if (
                valid_eval["target_rate"] >= PREFERRED_MIN_VALID_RATE
                and valid_eval["target_coverage"] >= PREFERRED_MIN_VALID_COVERAGE
                and valid_eval["matched_count"] >= PREFERRED_MIN_VALID_MATCHED
        ):
            preferred_candidates.append(item)

        if (
                valid_eval["target_rate"] >= STRICT_MIN_VALID_RATE
                and valid_eval["target_coverage"] >= STRICT_MIN_VALID_COVERAGE
                and valid_eval["matched_count"] >= STRICT_MIN_VALID_MATCHED
        ):
            strict_candidates.append(item)

        if (
                valid_eval["target_rate"] >= FALLBACK_MIN_VALID_RATE
                and valid_eval["target_coverage"] >= FALLBACK_MIN_VALID_COVERAGE
                and valid_eval["matched_count"] >= FALLBACK_MIN_VALID_MATCHED
        ):
            fallback_precision_candidates.append(item)

    if preferred_candidates:
        # 1순위: V2 개선형. 68% 이상 정확도 + 13.5% 이상 coverage 중 coverage 최대.
        best = max(
            preferred_candidates,
            key=lambda x: (
                x["valid_eval"]["target_rate"] - abs(x["train_eval"]["target_rate"] - x["valid_eval"]["target_rate"]) * 0.75,
                x["valid_eval"]["target_coverage"],
                x["valid_eval"]["matched_count"],
            ),
        )
        print(
            f"[SCENARIO SELECTED] preferred v2-improve precision>=68 coverage>=13.5: "
            f"{best['scenario']['name']}"
        )

    elif strict_candidates:
        # 2순위: 정밀형. coverage는 조금 낮아도 69% 이상이면 선택.
        best = max(
            strict_candidates,
            key=lambda x: (
                x["valid_eval"]["target_rate"] - abs(x["train_eval"]["target_rate"] - x["valid_eval"]["target_rate"]) * 0.75,
                x["valid_eval"]["target_coverage"],
                x["valid_eval"]["matched_count"],
            ),
        )
        print(
            f"[SCENARIO SELECTED] strict precision>=69: "
            f"{best['scenario']['name']}"
        )

    elif fallback_precision_candidates:
        # 3순위: 67% 이상 후보 중 coverage 최대.
        best = max(
            fallback_precision_candidates,
            key=lambda x: (
                x["valid_eval"]["target_rate"] - abs(x["train_eval"]["target_rate"] - x["valid_eval"]["target_rate"]) * 0.75,
                x["valid_eval"]["target_coverage"],
                x["valid_eval"]["matched_count"],
            ),
        )
        print(
            f"[SCENARIO SELECTED] fallback precision>=67 max coverage: "
            f"{best['scenario']['name']}"
        )

    elif accepted:
        # v4 선택: coverage를 유지하면서 rate를 더 올리기 위해 coverage bucket을 둔다.
        # 1) 92% 이상 유지 후보가 있으면 그 안에서 rate 최우선
        # 2) 없으면 90% 이상 유지 후보
        # 3) 없으면 기존 accept floor 안에서 rate 최우선
        # 이렇게 하면 63%대 후보가 coverage를 너무 많이 깎을 때는 배제하고,
        # 11%대 coverage를 지키는 후보 중 더 높은 precision 조합을 선택한다.
        bucket_rules = [
            (0.94, 0.92, "very high coverage-preserving rate-up"),
            (0.92, 0.90, "high coverage-preserving rate-up"),
            (PREFERRED_COVERAGE_KEEP_RATIO, PREFERRED_MATCHED_KEEP_RATIO, "preferred coverage-preserving rate-up"),
        ]

        best = None
        selected_label = None
        for cov_keep, matched_keep, label in bucket_rules:
            bucket = [
                x for x in accepted
                if x["row"]["coverage_keep_ratio"] >= cov_keep
                   and x["row"]["matched_keep_ratio"] >= matched_keep
            ]
            if bucket:
                best = max(
                    bucket,
                    key=lambda x: (
                        x["valid_eval"]["target_rate"],
                        x["valid_eval"]["target_coverage"],
                        x["valid_eval"]["matched_count"],
                    ),
                )
                selected_label = label
                break

        if best is None:
            best = max(
                accepted,
                key=lambda x: (
                    x["valid_eval"]["target_rate"],
                    x["valid_eval"]["target_coverage"],
                    x["valid_eval"]["matched_count"],
                ),
            )
            selected_label = "rate-up accepted"

        print(f"[SCENARIO SELECTED] {selected_label}: {best['scenario']['name']}")
    else:
        # rate 개선 후보가 없으면 score 기준으로 선택.
        best = max(
            evaluated,
            key=lambda x: (
                x["row"]["score"],
                x["valid_eval"]["target_rate"],
                x["valid_eval"]["target_coverage"],
            ),
        )
        print(f"[SCENARIO SELECTED] fallback by score: {best['scenario']['name']}")

    summary = pd.DataFrame(rows).sort_values(
        ["score", "valid_rate", "valid_coverage"],
        ascending=[False, False, False],
    )
    return best["selected"], best["scenario"], summary

# =============================================================================
# 리포트
# =============================================================================

def make_report_df(candidates, selected):
    selected_names = {r["name"] for r in selected}
    rows = []

    for c in candidates:
        rows.append({
            "selected": c["name"] in selected_names,
            "valid_pass": c["valid_pass"],
            "name": c["name"],
            "train_count": c["train_count"],
            "train_target": c["train_target"],
            "train_rate": c["train_rate"],
            "train_lift": c["train_lift"],
            "train_coverage": c["train_coverage"],
            "valid_count": c["valid_count"],
            "valid_target": c["valid_target"],
            "valid_rate": c["valid_rate"],
            "valid_lift": c["valid_lift"],
            "valid_coverage": c["valid_coverage"],
            "rate_gap": c["rate_gap"],
            "generalized_rate": c["generalized_rate"],
            "stability_score": c["stability_score"],
            "conds": repr(c["conds"]),
        })

    return pd.DataFrame(rows)


def warn_if_needed(valid_eval):
    warnings = []
    if valid_eval["matched_count"] < WARN_MIN_VALID_MATCHED:
        warnings.append(f"valid matched가 작습니다: {valid_eval['matched_count']} < {WARN_MIN_VALID_MATCHED}")
    if valid_eval["target_rate"] < WARN_MIN_VALID_RATE:
        warnings.append(f"valid target_rate가 낮습니다: {valid_eval['target_rate'] * 100:.2f}%")
    if valid_eval["target_coverage"] < WARN_MIN_VALID_COVERAGE:
        warnings.append(f"valid target_coverage가 낮습니다: {valid_eval['target_coverage'] * 100:.2f}%")

    for w in warnings:
        print(f"[WARN] {w}")


# =============================================================================
# 메인
# =============================================================================

def find_stop_before_target_10_rules(csv_path=CSV_PATH, out_path=OUT_PATH, report_path=REPORT_PATH):
    df = pd.read_csv(csv_path, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 없습니다.")

    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    train, valid, split_date = split_train_valid_by_date_ratio(df, valid_ratio=VALID_RATIO)

    y_train = (train[TARGET_COL] == TARGET_VALUE).to_numpy()
    y_valid = (valid[TARGET_COL] == TARGET_VALUE).to_numpy()

    print(f"[DATA] total={len(df)}, train={len(train)}, valid={len(valid)}")
    print(f"[DATA] split_date={pd.to_datetime(split_date).date()}")
    print(f"[DATA] train target rate={y_train.mean() * 100:.2f}%")
    print(f"[DATA] valid target rate={y_valid.mean() * 100:.2f}%")
    print("[APPLIED_SCENARIOS]", ", ".join(s["name"] for s in APPLIED_SCENARIOS))

    features = select_default_features(train)
    print(f"[FEATURES] {len(features)} selected DEFAULT_FEATURES")
    print(features)

    literals, literal_masks = build_literals(train, features, min_count=MIN_CNT)
    print(f"[LITERALS] {len(literals)} literals")

    feature_groups, group_limits = get_feature_groups()

    rules = mine_target_rules(
        df=train,
        literals=literals,
        literal_masks=literal_masks,
        y=y_train,
        min_count=MIN_CNT,
        max_depth=MAX_DEPTH,
        beam=BEAM,
        top_n=TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    print(f"[RULES] mined={len(rules)}")

    candidates = attach_validation_metrics(train, valid, rules)
    passed = [c for c in candidates if c["valid_pass"]]
    print(f"[VALID FILTER] passed={len(passed)} / {len(candidates)}")

    print("\n[TOP CANDIDATES]")
    preview_cols = [
        "valid_pass", "name", "train_count", "train_rate", "valid_count",
        "valid_rate", "rate_gap", "stability_score", "conds",
    ]
    preview_df = make_report_df(candidates[:30], [])
    if not preview_df.empty:
        print(preview_df[preview_cols].to_string(index=False))

    selected, selected_scenario, scenario_summary = select_best_scenario(
        candidates=candidates,
        train=train,
        valid=valid,
        y_train=y_train,
        y_valid=y_valid,
        scenarios=APPLIED_SCENARIOS,
    )

    print("\n[SCENARIO SUMMARY]")
    if not scenario_summary.empty:
        print(scenario_summary.to_string(index=False))

    print(f"[SELECTED_SCENARIO] {selected_scenario['name'] if selected_scenario else 'NONE'}")
    print(f"[SELECT] selected={len(selected)}")

    train_final_mask = combined_mask(train, selected)
    valid_final_mask = combined_mask(valid, selected)

    print("\n[COMBINED EVAL]")
    train_eval = eval_mask(train, train_final_mask, "TRAIN")
    valid_eval = eval_mask(valid, valid_final_mask, "VALID")
    print_eval(train_eval)
    print_eval(valid_eval)
    warn_if_needed(valid_eval)

    report_df = make_report_df(candidates, selected)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[REPORT SAVED] {Path(report_path).resolve()}")

    scenario_report_path = Path(report_path).with_name(SCENARIO_REPORT_PATH.name)
    scenario_summary.to_csv(scenario_report_path, index=False, encoding="utf-8-sig")
    print(f"[SCENARIO REPORT SAVED] {scenario_report_path.resolve()}")

    header_comment = (
        "# auto-generated: lowscan bad avoid rules\n"
        "# source: 8-1_find_stop_avoid_rules.py\n"
        f"# target: {TARGET_COL} == {TARGET_VALUE}\n"
        f"# split_date: {pd.to_datetime(split_date).date()}\n"
        f"# applied_scenario: {selected_scenario['name'] if selected_scenario else 'NONE'}\n"
        f"# valid_rate: {valid_eval['target_rate']:.6f}\n"
        f"# valid_coverage: {valid_eval['target_coverage']:.6f}\n"
        f"# valid_matched: {valid_eval['matched_count']}\n"
        f"# valid_target: {valid_eval['matched_target']}\n"
        "# purpose: exclude candidates likely to hit stop before target within 10 days\n# target preference: valid_rate >= 68%, valid_coverage >= 13.5% when possible\n"
        "# usage:\n"
        "#    import numpy as np\n"
        "#    import lowscan_stop_before_target_10_rules as lowscan_rules\n"
        "#    avoid_conditions = lowscan_rules.build_conditions(df)\n"
        "#\n"
        "#    avoid_mask = np.zeros(len(df), dtype=bool)\n"
        "#    for cond in avoid_conditions.values():\n"
        "#        avoid_mask |= cond\n"
        "#\n"
        "#    df = df[~avoid_mask].copy()\n"
    )

    write_rule_file(Path(out_path), selected, header_comment=header_comment)
    print(f"[SAVED] {Path(out_path).resolve()}")

    return selected, report_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH, help="input csv path")
    parser.add_argument("--out", default=str(OUT_PATH), help="output python rule file path")
    parser.add_argument("--report", default=str(REPORT_PATH), help="output csv rule report path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    find_stop_before_target_10_rules(
        csv_path=args.csv,
        out_path=Path(args.out),
        report_path=Path(args.report),
    )

    try:
        import winsound
        winsound.Beep(1500, 500)
        winsound.Beep(1000, 500)
    except ImportError:
        pass
