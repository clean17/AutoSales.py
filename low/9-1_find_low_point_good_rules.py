"""
target_before_stop_7 == 1 대량 추출 룰셋 생성기 - broad OR 버전

목표
- target_before_stop_7 == 1인 데이터를 최대한 많이 뽑는 룰셋 생성
- 기존 고확률/저커버리지 룰 대신, valid에서 실제 표본 수가 충분한 broad rule을 우선 선택
- train / valid 2분할
- 여러 시나리오를 동시에 실행해 Pareto 후보 비교

실행
    python find_target_before_stop7_rules_broad_or.py --csv csv/low_result_7_desc.csv

출력
    lowscan_target_before_stop7_rules_broad_or.py
    lowscan_target_before_stop7_rules_broad_or.report.csv
    lowscan_target_before_stop7_rules_broad_or.scenario_summary.csv
    lowscan_target_before_stop7_rules_broad_or.monthly_report.csv
"""
from __future__ import annotations

import argparse
import heapq
import math
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = "../csv/low_result_7_desc.csv"
OUT_PATH = Path("lowscan_target_before_stop7_rules_broad_or.py")
DATE_COL = "today"
TARGET_COL = "target_before_stop_7"
TARGET_VALUE = 1
TARGET_NAME = "target_before_stop7"
VALID_RATIO = 0.20

N_QUANTILES = 18
MIN_UNIQUE_VALUES = 8
MIN_RULE_COUNT = 250
MAX_DEPTH = 3
BEAM = 40000
TOP_N = 50000

# train 후보는 넓게 허용한다. 최종 선택은 valid에서 결정한다.
TRAIN_MIN_RATE_UP = 0.005
TRAIN_MIN_LIFT = 1.00
TRAIN_MIN_WILSON_LOW = 0.30
EXPAND_MIN_RATE_UP = [-0.08, -0.04, -0.02]
EXPAND_MIN_LIFT = [0.80, 0.88, 0.94]

# valid 후보 pool. 너무 작은 valid_n을 버리는 것이 핵심.
VALID_MIN_COUNT = 80
VALID_MIN_RATE_UP = -0.005
VALID_MIN_LIFT = 0.985
VALID_MIN_WILSON_LOW = 0.28

MAX_RULES = 200
MIN_VALID_ADD_COUNT_PRIMARY = 20
MIN_VALID_ADD_COUNT_FILL = 10
MIN_VALID_ADD_POS_PRIMARY = 8
MIN_VALID_ADD_POS_FILL = 4

SCENARIOS = [
    dict(name="precision80_attempt", target_precision=0.80, floor_precision=0.72,
         target_positive_coverage=0.20, soft_positive_coverage=0.08,
         min_added_precision_primary=0.70, min_added_precision_fill=0.66,
         tp_weight=18.0, selected_weight=0.20, coverage_weight=500.0,
         precision_weight=260.0, fp_penalty=7.0, under_target_penalty=450.0),
    dict(name="precision70_coverage", target_precision=0.70, floor_precision=0.63,
         target_positive_coverage=0.45, soft_positive_coverage=0.20,
         min_added_precision_primary=0.62, min_added_precision_fill=0.58,
         tp_weight=16.0, selected_weight=0.45, coverage_weight=900.0,
         precision_weight=150.0, fp_penalty=4.5, under_target_penalty=300.0),
    dict(name="precision60_coverage", target_precision=0.60, floor_precision=0.555,
         target_positive_coverage=0.65, soft_positive_coverage=0.35,
         min_added_precision_primary=0.54, min_added_precision_fill=0.51,
         tp_weight=14.0, selected_weight=0.75, coverage_weight=1250.0,
         precision_weight=90.0, fp_penalty=2.8, under_target_penalty=200.0),
    dict(name="precision55_mass", target_precision=0.55, floor_precision=0.515,
         target_positive_coverage=0.80, soft_positive_coverage=0.50,
         min_added_precision_primary=0.50, min_added_precision_fill=0.485,
         tp_weight=12.0, selected_weight=1.10, coverage_weight=1550.0,
         precision_weight=55.0, fp_penalty=2.0, under_target_penalty=120.0),
    dict(name="max_profit_like", target_precision=0.0, floor_precision=0.0,
         target_positive_coverage=0.85, soft_positive_coverage=0.55,
         min_added_precision_primary=0.44, min_added_precision_fill=0.415,
         tp_weight=10.0, selected_weight=0.50, coverage_weight=1200.0,
         precision_weight=20.0, fp_penalty=1.15, under_target_penalty=0.0),
    dict(name="max_recall_soft", target_precision=0.0, floor_precision=0.0,
         target_positive_coverage=0.90, soft_positive_coverage=0.65,
         min_added_precision_primary=0.405, min_added_precision_fill=0.395,
         tp_weight=8.0, selected_weight=1.40, coverage_weight=1800.0,
         precision_weight=10.0, fp_penalty=0.85, under_target_penalty=0.0),
]


def get_exclude_columns(df: pd.DataFrame | None = None) -> set[str]:
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
            if (c.startswith("validation_") or c.startswith("day_to_")
                    or c.startswith("target_before_stop_")
                    or c.startswith("stop_before_target_")
                    or c.startswith("target_stop_same_day_")
                    or c.startswith("no_target_no_stop_")
                    or c.startswith("fast_success_")
                    or c.startswith("slow_success_")
                    or c.startswith("fail_success_")):
                exclude.add(c)
    return exclude


def get_features(df: pd.DataFrame) -> list[str]:
    exclude = get_exclude_columns(df)
    features = []
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


def get_feature_groups():
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

        "market_today_pct": "MARKET",
        "market_5d_pct": "MARKET",
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
        "MARKET": 1,
    }

    return feature_groups, group_limits


def split_train_valid(df: pd.DataFrame, valid_ratio=VALID_RATIO, date_col=DATE_COL):
    def row_split(work):
        n = len(work)
        cut = max(1, min(int(n * (1 - valid_ratio)), n - 1))
        return work.iloc[:cut].copy(), work.iloc[cut:].copy(), None

    if date_col not in df.columns:
        return row_split(df.reset_index(drop=True))
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values(date_col).reset_index(drop=True)
    dates = work[date_col].dropna().sort_values().unique()
    if len(dates) < 2:
        return row_split(work)
    cut_idx = max(1, min(int(len(dates) * (1 - valid_ratio)), len(dates) - 1))
    valid_start = pd.Timestamp(dates[cut_idx])
    train = work[work[date_col] < valid_start].copy()
    valid = work[work[date_col] >= valid_start].copy()
    if len(train) == 0 or len(valid) == 0:
        return row_split(work)
    return train.reset_index(drop=True), valid.reset_index(drop=True), valid_start


def positive_mask(df: pd.DataFrame) -> np.ndarray:
    return (pd.to_numeric(df[TARGET_COL], errors="coerce") == TARGET_VALUE).to_numpy(dtype=bool)


def wilson_lower_bound(success: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = success / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def thresholds_for_series(s: pd.Series):
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return []
    qs = np.linspace(0.035, 0.965, N_QUANTILES)
    return sorted({round(float(x), 10) for x in vals.quantile(qs).to_numpy() if np.isfinite(x)})


def build_literals(df: pd.DataFrame, features: list[str]):
    literals, masks = [], []
    for feat in features:
        arr = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        finite = np.isfinite(arr)
        for th in thresholds_for_series(df[feat]):
            literals.append((feat, "<=", th)); masks.append(finite & (arr <= th))
            literals.append((feat, ">=", th)); masks.append(finite & (arr >= th))
    return literals, masks


def make_mask(df: pd.DataFrame, conds) -> np.ndarray:
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


def class_counts(df: pd.DataFrame, mask: np.ndarray):
    if "target_class" not in df.columns:
        return {0: 0, 1: 0, 2: 0, 3: 0}
    sub = df[mask]
    return {cls: int((sub["target_class"] == cls).sum()) for cls in [0, 1, 2, 3]}


def class_totals(df: pd.DataFrame):
    if "target_class" not in df.columns:
        return {0: 0, 1: 0, 2: 0, 3: 0}
    return {cls: int((df["target_class"] == cls).sum()) for cls in [0, 1, 2, 3]}


def eval_mask(df: pd.DataFrame, mask: np.ndarray, label=""):
    total = len(df)
    selected = int(mask.sum())
    pos = positive_mask(df)
    pos_total = int(pos.sum())
    pos_count = int((mask & pos).sum())
    base = pos_total / total if total else 0.0
    rate = pos_count / selected if selected else 0.0
    c = class_counts(df, mask)
    ct = class_totals(df)
    out = {
        "label": label, "total_count": total,
        "selected_count": selected, "selected_rate": selected / total if total else 0.0,
        "positive_total": pos_total, "positive_count": pos_count,
        "base_positive_rate": base, "positive_rate": rate,
        "positive_lift": rate / base if base else 0.0,
        "positive_coverage": pos_count / pos_total if pos_total else 0.0,
        "positive_wilson_low": wilson_lower_bound(pos_count, selected),
        "class0_count": c[0], "class1_count": c[1], "class2_count": c[2], "class3_count": c[3],
        "class0_rate": c[0] / selected if selected else 0.0,
        "class1_rate": c[1] / selected if selected else 0.0,
        "class2_rate": c[2] / selected if selected else 0.0,
        "class3_rate": c[3] / selected if selected else 0.0,
        "class23_rate": (c[2] + c[3]) / selected if selected else 0.0,
    }
    for cls in [0, 1, 2, 3]:
        out[f"class{cls}_coverage"] = c[cls] / ct[cls] if ct[cls] else 0.0
    return out


def print_eval(e):
    print(f"\n[{e['label']}]")
    print(f"total_count           : {e['total_count']}")
    print(f"selected_count        : {e['selected_count']} ({e['selected_rate']*100:.2f}%)")
    print(f"target_col            : {TARGET_COL}")
    print(f"target_value          : {TARGET_VALUE}")
    print(f"positive_count        : {e['positive_count']} / {e['positive_total']}")
    print(f"positive_rate         : {e['positive_rate']*100:.2f}%")
    print(f"base_positive_rate    : {e['base_positive_rate']*100:.2f}%")
    print(f"positive_lift         : {e['positive_lift']:.3f}x")
    print(f"positive_coverage     : {e['positive_coverage']*100:.2f}%")
    print(f"positive_wilson_low   : {e['positive_wilson_low']*100:.2f}%")
    print(f"class0_rate           : {e['class0_rate']*100:.2f}%")
    print(f"class1_rate           : {e['class1_rate']*100:.2f}%")
    print(f"class2_rate           : {e['class2_rate']*100:.2f}%")
    print(f"class3_rate           : {e['class3_rate']*100:.2f}%")
    print(f"class23_rate          : {e['class23_rate']*100:.2f}%")


def _by_depth(values, depth):
    return values[min(depth, len(values) - 1)]


def rule_score(cnt, pos_cnt, rate, base):
    if cnt <= 0 or base <= 0:
        return 0.0
    lift = rate / base
    wilson = wilson_lower_bound(pos_cnt, cnt)
    rate_up = rate - base
    return (max(rate_up + 0.06, 0.001) ** 1.0
            * max(lift, 0.001) ** 0.6
            * max(wilson, 0.001) ** 1.0
            * np.log1p(cnt) ** 1.8
            * np.log1p(pos_cnt) ** 1.2)


def mine_rules(train, literals, literal_masks, pos):
    base = float(pos.mean())
    groups, limits = get_feature_groups()
    beams = [(np.ones(len(train), dtype=bool), [])]
    good = {}
    print(f"\n[{TARGET_NAME.upper()} RULE MINING]")
    print("target:", f"{TARGET_COL} == {TARGET_VALUE}")
    print("base_positive_rate:", round(base, 4))
    print("beam:", BEAM, "top_n:", TOP_N, "min_count:", MIN_RULE_COUNT, "max_depth:", MAX_DEPTH)

    for depth in range(MAX_DEPTH):
        print("----------------------------------")
        print(f"[{TARGET_NAME}] depth", depth)
        heap, uid = [], count()
        exp_rate = base + _by_depth(EXPAND_MIN_RATE_UP, depth)
        exp_lift = _by_depth(EXPAND_MIN_LIFT, depth)
        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}
            group_used = {}
            for f in used_feats:
                g = groups.get(f)
                if g:
                    group_used[g] = group_used.get(g, 0) + 1
            for lit, lmask in zip(literals, literal_masks):
                feat = lit[0]
                if feat in used_feats:
                    continue
                g = groups.get(feat)
                if g and group_used.get(g, 0) >= limits.get(g, 99):
                    continue
                m = base_mask & lmask
                cnt = int(m.sum())
                if cnt < MIN_RULE_COUNT:
                    continue
                pos_cnt = int((m & pos).sum())
                rate = pos_cnt / cnt
                lift = rate / base if base else 0.0
                wilson = wilson_lower_bound(pos_cnt, cnt)
                score = rule_score(cnt, pos_cnt, rate, base)
                if rate >= base + TRAIN_MIN_RATE_UP and lift >= TRAIN_MIN_LIFT and wilson >= TRAIN_MIN_WILSON_LOW:
                    key = tuple(sorted((f, op, round(float(th), 6)) for f, op, th in conds + [lit]))
                    old = good.get(key)
                    if old is None or score > old[6]:
                        good[key] = (cnt, pos_cnt, rate, lift, wilson, conds + [lit], score)
                if rate >= exp_rate and lift >= exp_lift:
                    k = (score, wilson, rate, lift, pos_cnt, cnt)
                    item = (k, next(uid), m, conds + [lit], rate, lift, wilson, cnt)
                    if len(heap) < BEAM:
                        heapq.heappush(heap, item)
                    elif k > heap[0][0]:
                        heapq.heapreplace(heap, item)
        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print(f"[{TARGET_NAME}] new", len(new))
        if not new:
            break
        tail = new[-1]
        print(f"[{TARGET_NAME}] tail rate:", round(tail[4], 3), "lift:", round(tail[5], 2),
              "wilson:", round(tail[6], 3), "cnt:", tail[7], "conds:", tail[3])
        beams = [(m, conds) for _, _, m, conds, _, _, _, _ in new]
    out = sorted(good.values(), key=lambda x: (-x[6], -x[4], -x[2], -x[1], -x[0]))
    return out[:TOP_N]


def pass_valid_pool(va, tr, valid_base):
    rate_up = va["positive_rate"] - valid_base
    gap = tr["positive_rate"] - va["positive_rate"]
    return (va["selected_count"] >= VALID_MIN_COUNT
            and rate_up >= VALID_MIN_RATE_UP
            and va["positive_lift"] >= VALID_MIN_LIFT
            and va["positive_wilson_low"] >= VALID_MIN_WILSON_LOW
            and gap <= 0.90)


def scenario_score(ev, scenario, valid_base):
    rate = ev["positive_rate"]
    rate_up = rate - valid_base
    selected = ev["selected_count"]
    tp = ev["positive_count"]
    fp = selected - tp
    score = (tp * scenario["tp_weight"]
             + selected * scenario["selected_weight"]
             + ev["positive_coverage"] * scenario["coverage_weight"]
             + rate * scenario["precision_weight"]
             + max(rate_up, 0.0) * 350.0
             - fp * scenario["fp_penalty"])
    if scenario["target_precision"] > 0:
        score -= max(0.0, scenario["target_precision"] - rate) * scenario["under_target_penalty"]
        score -= max(0.0, scenario["floor_precision"] - rate) * scenario["under_target_penalty"] * 3.0
    if ev["positive_coverage"] >= scenario["soft_positive_coverage"]:
        score += 15000
    if ev["positive_coverage"] >= scenario["target_positive_coverage"]:
        score += 30000
    if selected >= 1000:
        score += 5000
    if selected >= 3000:
        score += 10000
    if selected >= 8000:
        score += 20000
    return score


def select_rules(train, valid, rules):
    train_pos, valid_pos = positive_mask(train), positive_mask(valid)
    train_base, valid_base = float(train_pos.mean()), float(valid_pos.mean())
    print(f"\n[BASE RATE] train={train_base*100:.2f}% valid={valid_base*100:.2f}%")
    evaluated = []
    for i, (cnt, pos_cnt, rate, lift, wilson, conds, score) in enumerate(rules, start=1):
        name = f"{TARGET_NAME}_{i:05d}_s{score:.4f}_tr{rate:.3f}_wl{wilson:.3f}_n{cnt}"
        tr_mask, va_mask = make_mask(train, conds), make_mask(valid, conds)
        tr, va = eval_mask(train, tr_mask, name), eval_mask(valid, va_mask, name)
        if not pass_valid_pool(va, tr, valid_base):
            continue
        rate_up = max(0.0, va["positive_rate"] - valid_base)
        gap = max(0.0, tr["positive_rate"] - va["positive_rate"])
        stable_score = score * (1.0 + rate_up * 5.0) * (1.0 + np.log1p(va["selected_count"]) / 8.0) * (1.0 - min(gap, 0.85))
        evaluated.append(dict(name=name, conds=conds, train_mask=tr_mask, valid_mask=va_mask,
                              score=score, stable_score=stable_score, train_eval=tr, valid_eval=va))
    print("\n[CANDIDATES AFTER BROAD VALID POOL]", len(evaluated))
    if not evaluated:
        return [], np.zeros(len(train), bool), np.zeros(len(valid), bool), "none", []

    valid_total_pos = int(valid_pos.sum())

    def run_scenario(scenario):
        selected, selected_names = [], set()
        tr_mask = np.zeros(len(train), bool)
        va_mask = np.zeros(len(valid), bool)
        checkpoints = []

        def save_cp(stage):
            tr_ev, va_ev = eval_mask(train, tr_mask, "TRAIN"), eval_mask(valid, va_mask, "VALID")
            checkpoints.append(dict(stage=stage, selected=list(selected), train_mask=tr_mask.copy(),
                                    valid_mask=va_mask.copy(), train_eval=tr_ev, valid_eval=va_ev,
                                    score=scenario_score(va_ev, scenario, valid_base)))

        def choose(stage):
            cur = eval_mask(valid, va_mask, "VALID")
            best = None
            for cand in evaluated:
                if cand["name"] in selected_names:
                    continue
                new_tr, new_va = cand["train_mask"] & ~tr_mask, cand["valid_mask"] & ~va_mask
                va_n = int(new_va.sum())
                if va_n < (MIN_VALID_ADD_COUNT_PRIMARY if stage == "primary" else MIN_VALID_ADD_COUNT_FILL):
                    continue
                va_pos = int((new_va & valid_pos).sum())
                tr_pos = int((new_tr & train_pos).sum())
                tr_n = int(new_tr.sum())
                if va_pos < (MIN_VALID_ADD_POS_PRIMARY if stage == "primary" else MIN_VALID_ADD_POS_FILL):
                    continue
                va_rate = va_pos / va_n if va_n else 0.0
                tr_rate = tr_pos / tr_n if tr_n else 0.0
                min_add = scenario["min_added_precision_primary"] if stage == "primary" else scenario["min_added_precision_fill"]
                if va_rate < min_add:
                    continue
                next_mask = va_mask | new_va
                next_ev = eval_mask(valid, next_mask, "NEXT")
                next_rate = next_ev["positive_rate"]
                next_cov = next_ev["positive_coverage"]
                if scenario["target_precision"] > 0:
                    if next_rate < scenario["floor_precision"]:
                        continue
                else:
                    if next_rate < max(0.0, valid_base - 0.015):
                        continue
                if stage == "fill" and next_cov >= scenario["target_positive_coverage"] * 1.05:
                    continue
                fp = va_n - va_pos
                rate_up = next_rate - valid_base
                score = (va_pos * scenario["tp_weight"]
                         + va_n * scenario["selected_weight"]
                         + next_cov * scenario["coverage_weight"]
                         + next_rate * scenario["precision_weight"]
                         + max(rate_up, 0.0) * 250.0
                         + tr_pos * 0.15
                         - fp * scenario["fp_penalty"]
                         + cand["stable_score"] * 0.02)
                if scenario["target_precision"] > 0:
                    score -= max(0.0, scenario["target_precision"] - next_rate) * scenario["under_target_penalty"]
                    score -= max(0.0, scenario["floor_precision"] - next_rate) * scenario["under_target_penalty"] * 3.0
                key = (score, next_cov, next_rate, va_pos, va_n, cand["stable_score"])
                if best is None or key > best["key"]:
                    best = dict(key=key, score=score, cand=cand, valid_added_total=va_n,
                                valid_added_pos=va_pos, valid_added_rate=va_rate,
                                train_added_total=tr_n, train_added_pos=tr_pos, train_added_rate=tr_rate,
                                next_rate=next_rate, next_cov=next_cov)
            return best

        for stage in ["primary", "fill"]:
            while True:
                if MAX_RULES is not None and len(selected) >= MAX_RULES:
                    break
                cur = eval_mask(valid, va_mask, "VALID")
                if stage == "fill" and cur["positive_coverage"] >= scenario["target_positive_coverage"]:
                    print(f"[{scenario['name']}] [FILL STOP] reached target positive coverage.")
                    break
                best = choose(stage)
                if best is None:
                    print(f"[{scenario['name']}] [{stage.upper()} STOP] no acceptable rule.")
                    break
                cand = best["cand"]
                tr_mask |= cand["train_mask"]
                va_mask |= cand["valid_mask"]
                selected.append((cand["name"], cand["conds"]))
                selected_names.add(cand["name"])
                ev = eval_mask(valid, va_mask, "VALID")
                print(f"[SELECT:{scenario['name']}:{stage.upper()}]", f"{len(selected):03d}", cand["name"],
                      f"valid_add={best['valid_added_pos']}/{best['valid_added_total']} ({best['valid_added_rate']*100:.2f}%)",
                      f"combined_rate={ev['positive_rate']*100:.2f}%", f"coverage={ev['positive_coverage']*100:.2f}%",
                      f"selected={ev['selected_count']}", f"class23={ev['class23_rate']*100:.2f}%")
                save_cp(stage)
        if checkpoints:
            return max(checkpoints, key=lambda x: x["score"])
        return dict(stage="none", selected=selected, train_mask=tr_mask, valid_mask=va_mask,
                    train_eval=eval_mask(train, tr_mask, "TRAIN"), valid_eval=eval_mask(valid, va_mask, "VALID"), score=-1e18)

    results = []
    for scenario in SCENARIOS:
        print(f"\n[SCENARIO RUN] {scenario['name']}")
        res = run_scenario(scenario)
        res["scenario"] = scenario
        results.append(res)
        va, tr = res["valid_eval"], res["train_eval"]
        print(f"[SCENARIO RESULT] {scenario['name']} rules={len(res['selected'])} valid_selected={va['selected_count']} "
              f"valid_pos={va['positive_count']} valid_rate={va['positive_rate']*100:.2f}% "
              f"valid_coverage={va['positive_coverage']*100:.2f}% valid_lift={va['positive_lift']:.3f} "
              f"train_rate={tr['positive_rate']*100:.2f}% score={res['score']:.2f}")

    print("\n[SCENARIO SUMMARY]")
    rows = []
    for res in sorted(results, key=lambda x: x["score"], reverse=True):
        va, tr, sc = res["valid_eval"], res["train_eval"], res["scenario"]
        rows.append(dict(scenario=sc["name"], rules=len(res["selected"]),
                         train_selected=tr["selected_count"], train_pos=tr["positive_count"], train_rate=tr["positive_rate"], train_coverage=tr["positive_coverage"],
                         valid_selected=va["selected_count"], valid_pos=va["positive_count"], valid_rate=va["positive_rate"], valid_coverage=va["positive_coverage"],
                         valid_lift=va["positive_lift"], valid_wilson=va["positive_wilson_low"], valid_class23_rate=va["class23_rate"], score=res["score"]))
        print(f"{sc['name']:24s} rules={len(res['selected']):3d} valid_selected={va['selected_count']:5d} valid_pos={va['positive_count']:5d} "
              f"valid_rate={va['positive_rate']*100:6.2f}% valid_cov={va['positive_coverage']*100:6.2f}% "
              f"lift={va['positive_lift']:5.3f} class23={va['class23_rate']*100:6.2f}% score={res['score']:12.2f}")

    feasible_80 = [r for r in results if r["valid_eval"]["positive_rate"] >= 0.78 and r["valid_eval"]["selected_count"] >= 500]
    best = max(feasible_80, key=lambda x: x["valid_eval"]["positive_coverage"]) if feasible_80 else max(results, key=lambda x: x["score"])
    print(f"\n[SCENARIO SELECTED] {best['scenario']['name']}")
    return best["selected"], best["train_mask"], best["valid_mask"], best["scenario"]["name"], rows


def cond_to_python_expr(conds, df_name="df"):
    return " & ".join(f"({df_name}[{feat!r}] {op} {float(th)!r})" for feat, op, th in conds) or "np.ones(len(df), dtype=bool)"


def write_rule_file(out_path, selected, header_comment):
    lines = [header_comment.rstrip(), "", "import numpy as np", "", f"TARGET_COL = {TARGET_COL!r}", f"TARGET_VALUE = {TARGET_VALUE!r}", ""]
    lines.append("def build_conditions(df):")
    lines.append("    conditions = {}")
    for name, conds in selected:
        lines.append(f"    conditions[{name!r}] = ({cond_to_python_expr(conds)}).to_numpy(dtype=bool)")
    lines.append("    return conditions")
    lines.append("")
    lines.append("def build_mask(df):")
    lines.append("    mask = np.zeros(len(df), dtype=bool)")
    lines.append("    for cond in build_conditions(df).values():")
    lines.append("        mask |= cond")
    lines.append("    return mask")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def save_selected_rule_report(out_path, selected, train, valid):
    rows = []
    for name, conds in selected:
        tr = eval_mask(train, make_mask(train, conds), "train")
        va = eval_mask(valid, make_mask(valid, conds), "valid")
        row = {"name": name, "conds": " AND ".join(f"{f} {op} {th:.8g}" for f, op, th in conds)}
        for prefix, ev in [("train", tr), ("valid", va)]:
            row.update({
                f"{prefix}_selected": ev["selected_count"],
                f"{prefix}_positive_rate": ev["positive_rate"],
                f"{prefix}_positive_lift": ev["positive_lift"],
                f"{prefix}_positive_wilson_low": ev["positive_wilson_low"],
                f"{prefix}_positive_coverage": ev["positive_coverage"],
                f"{prefix}_class0_rate": ev["class0_rate"],
                f"{prefix}_class1_rate": ev["class1_rate"],
                f"{prefix}_class2_rate": ev["class2_rate"],
                f"{prefix}_class3_rate": ev["class3_rate"],
                f"{prefix}_class23_rate": ev["class23_rate"],
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_monthly_report(out_path, df, mask):
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
        ev = eval_mask(g.drop(columns=["_selected", "_month"]), m, month)
        rows.append({"month": month, **{k: ev[k] for k in ["total_count", "selected_count", "selected_rate", "base_positive_rate", "positive_rate", "positive_lift", "positive_wilson_low", "positive_coverage", "class0_rate", "class1_rate", "class2_rate", "class3_rate", "class23_rate"]}})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def interpret_final_result(train_eval, valid_eval):
    print("\n[INTERPRETATION]")
    print(f"train_positive_rate     : {train_eval['positive_rate']*100:.2f}%")
    print(f"train_positive_coverage : {train_eval['positive_coverage']*100:.2f}%")
    print(f"train_selected_count    : {train_eval['selected_count']}")
    print(f"valid_positive_rate     : {valid_eval['positive_rate']*100:.2f}%")
    print(f"valid_positive_coverage : {valid_eval['positive_coverage']*100:.2f}%")
    print(f"valid_selected_count    : {valid_eval['selected_count']}")
    gap = train_eval["positive_rate"] - valid_eval["positive_rate"]
    rate_up = valid_eval["positive_rate"] - valid_eval["base_positive_rate"]
    print(f"valid_rate_up_vs_base   : {rate_up*100:.2f}%p")
    print(f"train_valid_rate_gap    : {gap*100:.2f}%p")
    if valid_eval["selected_count"] == 0:
        print("type: no-valid-coverage")
        print("해석: 룰셋이 아무 것도 선택하지 못했습니다.")
    elif valid_eval["positive_rate"] >= 0.78 and valid_eval["selected_count"] >= 500:
        print("type: high precision broad candidate")
        print("해석: 80%에 가까운 성공률과 의미 있는 표본 수를 동시에 확보했습니다.")
    elif valid_eval["positive_rate"] >= 0.65:
        print("type: medium-high precision candidate")
        print("해석: 성공률은 높지만 목표 80%에는 부족합니다. coverage와 trade-off를 봐야 합니다.")
    elif valid_eval["positive_rate"] >= valid_eval["base_positive_rate"] + 0.05:
        print("type: useful rate-up candidate")
        print("해석: base 대비 상승폭은 있습니다. 대량 추출 후보로 검토할 수 있습니다.")
    elif valid_eval["positive_rate"] >= valid_eval["base_positive_rate"]:
        print("type: mass extraction weak edge")
        print("해석: base보다 낮지는 않지만 edge가 약합니다. 단독 사용보다는 다른 필터와 결합하세요.")
    else:
        print("type: no reliable edge")
        print("해석: 현재 feature/rule 조합으로는 target_before_stop_7 == 1 대량 선별이 어렵습니다.")


def find_target_before_stop7_rules_broad_or(csv_path=CSV_PATH, out_path=OUT_PATH):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = pd.read_csv(csv_path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 없습니다.")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    if "target_class" in df.columns:
        df = df[df["target_class"].notna()].copy()
        df["target_class"] = df["target_class"].astype(int)
    train, valid, valid_start = split_train_valid(df)
    print("[DATA]")
    print("csv:", csv_path)
    print("rows:", len(df), "train:", len(train), "valid:", len(valid))
    if valid_start is not None:
        print("valid_start:", pd.to_datetime(valid_start).date())
    print("\n[TARGET DISTRIBUTION]")
    for name, part in [("train", train), ("valid", valid)]:
        print(name)
        print(part[TARGET_COL].value_counts(normalize=True).sort_index().round(4))
        print("positive_rate", round(float((part[TARGET_COL] == TARGET_VALUE).mean()), 4))
        if "target_class" in part.columns:
            print("target_class")
            print(part["target_class"].value_counts(normalize=True).sort_index().round(4))
    features = get_features(train)
    print("\n[FEATURES]", len(features), features)
    literals, literal_masks = build_literals(train, features)
    print("[LITERALS]", len(literals))
    rules = mine_rules(train, literals, literal_masks, positive_mask(train))
    print(f"\n[{TARGET_NAME}] 후보 룰 개수: {len(rules)}")
    selected, train_mask, valid_mask, scenario_name, scenario_rows = select_rules(train, valid, rules)
    print(f"\n[{TARGET_NAME}] 최종 통과 룰 개수: {len(selected)} / {len(rules)}")
    print("[SELECTED_SCENARIO]", scenario_name)
    split_comment = f"# valid_start: {pd.to_datetime(valid_start).date()}\n" if valid_start is not None else "# split: row-ratio train/valid\n"
    write_rule_file(out_path, selected, "# auto-generated: broad OR rules for target_before_stop_7 == 1\n" + split_comment)
    report_path = out_path.with_suffix(".report.csv")
    scenario_path = out_path.with_suffix(".scenario_summary.csv")
    monthly_path = out_path.with_suffix(".monthly_report.csv")
    save_selected_rule_report(report_path, selected, train, valid)
    pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False, encoding="utf-8-sig")
    save_monthly_report(monthly_path, valid, valid_mask)
    tr_ev = eval_mask(train, train_mask, "TRAIN COMBINED")
    va_ev = eval_mask(valid, valid_mask, "VALID COMBINED")
    print_eval(tr_ev)
    print_eval(va_ev)
    interpret_final_result(tr_ev, va_ev)
    print("\n[OUTPUT]")
    print("rule_file:", out_path)
    print("report:", report_path)
    print("scenario_summary:", scenario_path)
    print("monthly_report:", monthly_path)
    return selected, train_mask, valid_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--out", default=str(OUT_PATH))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    find_target_before_stop7_rules_broad_or(args.csv, args.out)
    try:
        import winsound
        winsound.Beep(1500, 500)
        winsound.Beep(1000, 500)
    except Exception:
        pass

# 실행방법
"""
균형
python stable_rule_miner_final.py \
  --csv csv/low_result_7_desc.csv \
  --out stable_rule_miner_final_out \
  --date-col today \
  --max-depth 6 \
  --beam-width 1500 \
  --top-k 150 \
  --min-train-count 60 \
  --min-valid-count 40 \
  --min-train-precision 0.55 \
  --min-valid-precision 0.62 \
  --min-train-lift 1.25 \
  --min-valid-lift 1.50 \
  --max-precision-gap 0.18 \
  --min-month-count 5 \
  --min-valid-pass-month-rate 0.20 \
  --valid-count-weight 7 \
  --simplify
  
70% 확률 우선
python stable_rule_miner_final.py \
  --csv csv/low_result_7_desc.csv \
  --out stable_rule_miner_final_precision70 \
  --date-col today \
  --max-depth 6 \
  --beam-width 2000 \
  --top-k 200 \
  --min-train-count 50 \
  --min-valid-count 30 \
  --min-train-precision 0.55 \
  --min-valid-precision 0.68 \
  --min-train-lift 1.25 \
  --min-valid-lift 1.60 \
  --max-precision-gap 0.22 \
  --min-month-count 5 \
  --min-valid-pass-month-rate 0.20 \
  --valid-count-weight 5 \
  --simplify
  
종목 수 우선
python stable_rule_miner_final.py \
  --csv csv/low_result_7_desc.csv \
  --out stable_rule_miner_final_more_count \
  --date-col today \
  --max-depth 6 \
  --beam-width 2000 \
  --top-k 200 \
  --min-train-count 80 \
  --min-valid-count 60 \
  --min-train-precision 0.52 \
  --min-valid-precision 0.60 \
  --min-train-lift 1.20 \
  --min-valid-lift 1.45 \
  --max-precision-gap 0.18 \
  --min-month-count 5 \
  --min-valid-pass-month-rate 0.20 \
  --valid-count-weight 12 \
  --simplify
"""