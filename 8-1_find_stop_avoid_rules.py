"""
stop_before_target_7 == 1 룰 생성 스크립트 - 적용 버전

이번 결과에서 가장 좋았던 시나리오를 기본 적용한다.

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
    python find_stop_before_target_7_rules_applied.py
    python find_stop_before_target_7_rules_applied.py --csv csv/low_result_7_desc.csv

생성:
    lowscan_stop_before_target_7_rules.py
    lowscan_stop_before_target_7_rule_report.csv
"""

import argparse
import heapq
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 기본 설정
# =============================================================================

CSV_PATH = "csv/low_result_7_desc.csv"
OUT_PATH = Path("lowscan_stop_before_target_7_rules.py")
REPORT_PATH = Path("lowscan_stop_before_target_7_rule_report.csv")

TARGET_COL = "stop_before_target_7"
TARGET_VALUE = 1
DATE_COL = "today"

VALID_RATIO = 0.20

# 후보 탐색 파라미터: 기존 안정형 유지
BEAM = 10000
TOP_N = 3000
MIN_CNT = 80
MAX_DEPTH = 3

# 후보 룰 기준: 기존 안정형 유지
MIN_TARGET_RATE = 0.58
MIN_LIFT = 1.25

# valid 후보 통과 기준: 너무 빡세게 올리면 selected=0 위험
VALID_MIN_RATE = 0.57
VALID_MIN_CNT = 30
MAX_RATE_GAP = 0.18
VALID_MIN_LIFT = 1.10

# literal 생성 설정
N_QUANTILES = 10
MAX_UNIQUE_FOR_EQ = 8

# stability 점수 가중치
VALID_WEIGHT = 1.25
RATE_GAP_PENALTY_POWER = 2.0

# =============================================================================
# 최종 적용 시나리오: coverage_keep_rate_loose
# =============================================================================
# 이번 결과에서 둘 다 개선된 시나리오:
#   valid target_rate      61.54% -> 62.58%
#   valid target_coverage  9.15%  -> 10.97%

APPLIED_SCENARIO = {
    "name": "coverage_keep_rate_loose_applied",
    "max_rules": 12,

    # 신규 추가분 최소 크기
    "min_added_total": 15,
    "min_added_target": 8,
    "min_added_valid_total": 10,
    "min_added_valid_target": 5,

    # 신규 추가분 품질 기준
    "min_train_added_rate": 0.57,
    "min_valid_added_rate": 0.565,
    "max_added_gap": 0.20,

    # 누적 valid 성능 방어선
    "min_next_valid_rate": 0.595,
    "max_rate_drop": 0.012,

    # coverage 우선이지만 rate가 무너지지 않도록 균형
    "objective": "coverage",
}

# 너무 작은 최종 결과 방지용 경고 기준
WARN_MIN_VALID_MATCHED = 300
WARN_MIN_VALID_COVERAGE = 0.08
WARN_MIN_VALID_RATE = 0.60


# =============================================================================
# 데이터 / 피쳐 유틸
# =============================================================================

def get_exclude_columns(df=None):
    exclude = {
        "ticker", "stock_name", "today", "idx",
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


def get_features(df):
    exclude = get_exclude_columns(df)
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def get_feature_groups():
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

    if op == "<=":
        return f'(pd.to_numeric(df[{feat!r}], errors="coerce") <= {val:.10g})'
    if op == ">=":
        return f'(pd.to_numeric(df[{feat!r}], errors="coerce") >= {val:.10g})'
    if op == "==":
        return f'(pd.to_numeric(df[{feat!r}], errors="coerce") == {val:.10g})'

    raise ValueError(f"지원하지 않는 op: {op}")


def write_rule_file(path, selected, header_comment=""):
    lines = []
    lines.append(header_comment.rstrip())
    lines.append("")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("")
    lines.append("def build_conditions(df):")
    lines.append("    conditions = {}")

    if not selected:
        lines.append("    return conditions")
    else:
        for rule in selected:
            name = rule["name"]
            conds = rule["conds"]
            expr = " & ".join(_cond_to_code(c) for c in conds)
            lines.append(f"    conditions[{name!r}] = {expr}")

        lines.append("    return conditions")

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

def select_rules_applied(candidates, y_train, y_valid, scenario=APPLIED_SCENARIO, verbose=True):
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

            # coverage_keep_rate_loose 결과를 재현하는 점수식.
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

def find_stop_before_target_7_rules(csv_path=CSV_PATH, out_path=OUT_PATH, report_path=REPORT_PATH):
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
    print(f"[APPLIED_SCENARIO] {APPLIED_SCENARIO['name']}")

    features = get_features(train)
    print(f"[FEATURES] {len(features)} features")
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

    selected = select_rules_applied(
        candidates=candidates,
        y_train=y_train,
        y_valid=y_valid,
        scenario=APPLIED_SCENARIO,
        verbose=True,
    )

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

    header_comment = (
        "# auto-generated: applied coverage_keep_rate_loose rules for stop_before_target_7 == 1\n"
        f"# target: {TARGET_COL} == {TARGET_VALUE}\n"
        f"# split_date: {pd.to_datetime(split_date).date()}\n"
        f"# applied_scenario: {APPLIED_SCENARIO['name']}\n"
        "# expected from previous run: valid target_rate around 62.58%, target_coverage around 10.97%\n"
        "# usage:\n"
        "#   import numpy as np\n"
        "#   import lowscan_stop_before_target_7_rules\n"
        "#   conditions = lowscan_stop_before_target_7_rules.build_conditions(df)\n"
        "#   rule_mask = np.zeros(len(df), dtype=bool)\n"
        "#   for cond in conditions.values():\n"
        "#       rule_mask |= cond\n"
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

    find_stop_before_target_7_rules(
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
