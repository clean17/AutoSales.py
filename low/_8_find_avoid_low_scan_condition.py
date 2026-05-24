"""
생성한 csv 파일을 가지고 저점 조건을 찾는 스크립트
binning 기반 rule mining
beam search 기반 조건 조합 모델

사용할 때는

import lowscan_avoid_class0_rules
import lowscan_avoid_stop_first_rules

avoid_class0_conditions = lowscan_avoid_class0_rules.build_conditions(df)
avoid_stop_conditions = lowscan_avoid_stop_first_rules.build_conditions(df)

avoid_class0_mask = np.zeros(len(df), dtype=bool)
for cond in avoid_class0_conditions.values():
    avoid_class0_mask |= cond

avoid_stop_mask = np.zeros(len(df), dtype=bool)
for cond in avoid_stop_conditions.values():
    avoid_stop_mask |= cond

final_avoid_mask = avoid_class0_mask | avoid_stop_mask
final_buy_mask = buy_mask & ~final_avoid_mask
"""

import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count

from utils import split_train_valid_by_date_ratio, build_literals

CSV_PATH = "../csv/low_result_7_desc.csv"

DEPTH0_FEATURES = {
    "vol5",
    "ATR_pct",
    "today_pct",
    "max_drop_7d",
    "dist_from_low_20d",
    "vol_ratio_5_15",
    "three_m_cur_max_chg_rate",
}

# ============================================================
# CLASS0 AVOID RULE SETTINGS
# 목적: target_class == 0 제거
# 보호: class3 최우선, class2도 강하게 보호
# ============================================================
AVOID_OUT_PATH = Path("lowscan_avoid_class0_rules.py")
AVOID_EVAL_PATH = Path("csv/class0_avoid_eval.csv")

MIN_CNT_AVOID = 80
MAX_DEPTH_AVOID = 5

AVOID_BEAM = 30000
AVOID_TOP_N = 3000

AVOID_EXPAND_BAD_RATIO = [0.32, 0.45, 0.55, 0.65, 0.70]
AVOID_EXPAND_MAX_PROTECT_RATE = [0.65, 0.55, 0.42, 0.32, 0.22]
AVOID_EXPAND_MAX_STRONG_RATE = [0.50, 0.40, 0.30, 0.20, 0.12]

MIN_BAD_RATE = 0.68
MAX_PROTECT_RATE = 0.18
MAX_STRONG_RATE = 0.08

CLASS_2_SCORE = 3.0
CLASS_3_SCORE = 50.0
CLASS_3_LOSS_LIMIT = 0.010  # 0.01 (1%)

VALID_RATIO = 0.10



def is_literal_allowed_at_depth(feat, depth):
    """
    모든 룰 마이닝에서 depth 0은 상위 피처만 허용.
    depth 1 이상에서는 전체 피처 허용.
    """
    if depth == 0:
        return feat in DEPTH0_FEATURES

    return True


def get_exclude_columns(df=None):
    """
    제외할 피쳐 Set

    df를 넘기면 패턴 기반으로 실제 존재하는 컬럼까지 자동 제외.
    df를 안 넘겨도 기본 제외 컬럼 set 반환.
    """
    exclude = {
        # 식별자 / 메타
        "ticker",
        "stock_name",
        "predict_str",
        "today",
        "idx",

        # stop / target / label
        "stop_loss",
        "stop_day",
        "target_pct",
        "target_class",

        # 과거 실험용 / raw 후보
        "_close_pos_20d",
        "_tr_value_ratio",
        "_tr_value_ratio_5d",
        "_dist_to_high_20d",
        "_BB_perc",
        "_UltimateOsc",
        "_CCI14",
        "_ADX14",
        "_gap_pct",
        "_vol_ratio_15_60",
        "_RSI_rebound",
        "_rebound_power",
        "_MACD_hist_1d",
        "_MACD_acc",
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


def get_features(df):
    exclude = get_exclude_columns(df)

    features = [
        c for c in df.columns
        if c not in exclude
           and np.issubdtype(df[c].dtype, np.number)
    ]

    return features


def get_feature_groups():
    feature_groups = {
        # 가격 / 당일 반등 강도
        "today_pct": "PRICE",

        # 최근 눌림 강도
        "max_drop_7d": "DROP",

        # 위치 / 회복 계열
        "dist_from_low_20d": "POSITION1",
        "three_m_cur_max_chg_rate": "POSITION2",
        "dist_to_ma5": "POSITION3",
        "dist_to_ma20": "POSITION4",
        "pct_vs_lastweek": "POSITION5",

        # 추세 개선
        "ma5_ma20_gap_chg_1d": "TREND",

        # 갭
        "gap_pct": "GAP",

        # 변동성 / 거래 확장 계열
        "vol5": "VOLATILITY1",
        "ATR_pct": "VOLATILITY2",
        "vol_ratio_5_15": "VOLATILITY3",
    }

    group_limits = {
        "PRICE": 1,
        "DROP": 1,
        "POSITION1": 1,
        "POSITION2": 1,
        "POSITION3": 1,
        "POSITION4": 1,
        "POSITION5": 1,
        "TREND": 1,
        "GAP": 1,
        "VOLATILITY": 2,
        "VOLATILITY": 2,
    }

    return feature_groups, group_limits


def mine_avoid_rules(
        df,
        literals,
        literal_masks,
        bad,
        protect,
        strong,
        min_count=25,
        max_depth=4,
        beam=30000,
        top_n=1000,
        feature_groups=None,
        group_limits=None,
):
    beams = [(np.ones(len(df), dtype=bool), [])]
    good = {}

    print(
        "\nbeam", beam,
        "\ntop_n", top_n,
        "\nmin_count", min_count,
        "\nmax_depth", max_depth,
        "\nexpand_bad_ratio", AVOID_EXPAND_BAD_RATIO,
        "\navoid_expand_max_protect_rate", AVOID_EXPAND_MAX_PROTECT_RATE,
        "\nexpand_max_strong_rate", AVOID_EXPAND_MAX_STRONG_RATE,
        "\nmin_bad_rate", MIN_BAD_RATE,
        "\nmax_protect_rate", MAX_PROTECT_RATE,
        "\nmax_strong_rate", MAX_STRONG_RATE,
        "\nclass_3_loss_limit", CLASS_3_LOSS_LIMIT,
        "\nclass_2_score", CLASS_2_SCORE,
        "\nclass_3_score", CLASS_3_SCORE,
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
        print("[AVOID] depth", depth)

        heap = []
        uid = count()

        save_stats = {
            "cnt_ok": 0,
            "bad_ok": 0,
            "protect_ok": 0,
            "strong_ok": 0,
            "save_ok": 0,
        }

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

            group_used = {}
            for f in used_feats:
                g = get_group(f)
                if g is not None:
                    group_used[g] = group_used.get(g, 0) + 1

            for lit, lmask in zip(literals, literal_masks):
                feat = lit[0]

                # depth 0에서는 상위 피처만 시작 조건으로 허용
                if not is_literal_allowed_at_depth(feat, depth):
                    continue

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

                bad_cnt = int((m & bad).sum())
                protect_cnt = int((m & protect).sum())
                strong_cnt = int((m & strong).sum())

                bad_rate = bad_cnt / cnt
                protect_rate = protect_cnt / cnt
                strong_rate = strong_cnt / cnt

                save_stats["cnt_ok"] += 1

                if bad_rate >= MIN_BAD_RATE:
                    save_stats["bad_ok"] += 1

                if protect_rate <= MAX_PROTECT_RATE:
                    save_stats["protect_ok"] += 1

                if strong_rate <= MAX_STRONG_RATE:
                    save_stats["strong_ok"] += 1

                if (
                        bad_rate >= MIN_BAD_RATE
                        and protect_rate <= MAX_PROTECT_RATE
                        and strong_rate <= MAX_STRONG_RATE
                ):
                    save_stats["save_ok"] += 1

                bad_cnt = int((m & bad).sum())
                protect_cnt = int((m & protect).sum())
                strong_cnt = int((m & strong).sum())

                bad_rate = bad_cnt / cnt
                protect_rate = protect_cnt / cnt
                strong_rate = strong_cnt / cnt

                score = (
                        (bad_rate ** 2.5)
                        * np.log1p(cnt)
                        * ((1 - protect_rate) ** 1.5)
                        * ((1 - strong_rate) ** 2.0)
                )

                if (
                        bad_rate >= MIN_BAD_RATE
                        and protect_rate <= MAX_PROTECT_RATE
                        and strong_rate <= MAX_STRONG_RATE
                ):
                    key2 = tuple(sorted(
                        (c[0], c[1], round(float(c[2]), 6))
                        for c in (conds + [lit])
                    ))

                    prev = good.get(key2)
                    if prev is None or score > prev[8]:
                        good[key2] = (
                            cnt,
                            bad_cnt,
                            bad_rate,
                            protect_cnt,
                            protect_rate,
                            strong_cnt,
                            strong_rate,
                            conds + [lit],
                            score,
                        )

                if (
                        bad_rate >= AVOID_EXPAND_BAD_RATIO[depth]
                        and protect_rate <= AVOID_EXPAND_MAX_PROTECT_RATE[depth]
                        and strong_rate <= AVOID_EXPAND_MAX_STRONG_RATE[depth]
                ):
                    k = (score, bad_rate, -strong_rate, -protect_rate, cnt)
                    item = (k, next(uid), m, conds + [lit], bad_rate, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        if k <= heap[0][0]:
                            continue
                        heapq.heapreplace(heap, item)

        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("[AVOID] new", len(new))
        print("[AVOID] save_stats:", save_stats)

        if not new:
            print("[AVOID] no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("[AVOID] tail bad_rate:", round(tail[4], 3), "cnt:", tail[5], "conds:", tail[3])

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    def out_key(x):
        cnt, bad_cnt, bad_rate, protect_cnt, protect_rate, strong_cnt, strong_rate, conds, score = x
        return (-score, -bad_rate, strong_rate, protect_rate, -cnt)

    out = sorted(good.values(), key=out_key)

    if top_n is not None:
        out = out[:top_n]

    return out


def make_mask_from_conds(df, conds):
    mask = np.ones(len(df), dtype=bool)

    for f, op, thr in conds:
        if op == "<=":
            mask &= (df[f] <= thr)
        else:
            mask &= (df[f] > thr)

    return mask



def test_avoid_condition(name, cond, df, min_count=25, verbose=False):
    sub = df[cond]

    if len(sub) == 0:
        return False

    cnt = len(sub)

    class0_cnt = int((sub["target_class"] == 0).sum())
    class2_cnt = int((sub["target_class"] == 2).sum())
    class3_cnt = int((sub["target_class"] == 3).sum())

    class0_rate = class0_cnt / cnt
    protect_rate = (class2_cnt + class3_cnt) / cnt
    class3_rate = class3_cnt / cnt

    # 전체 class별 총량 대비 얼마나 제거했는지도 중요
    total_class0 = int((df["target_class"] == 0).sum())
    total_class2 = int((df["target_class"] == 2).sum())
    total_class3 = int((df["target_class"] == 3).sum())

    class0_remove_rate = class0_cnt / total_class0 if total_class0 else 0
    class2_loss_rate = class2_cnt / total_class2 if total_class2 else 0
    class3_loss_rate = class3_cnt / total_class3 if total_class3 else 0

    # 핵심: class_0은 많이 제거하고 class_2/3은 적게 잃는 룰
    score = (
            class0_remove_rate
            - class2_loss_rate
            - class3_loss_rate * 1.5
    )

    if cnt < min_count:
        return False

    # 룰 subset 내부가 class_0 위주여야 함
    if class0_rate < MIN_BAD_RATE:
        return False

    # 좋은 종목이 너무 많이 섞이면 제외 룰로 부적합
    if protect_rate > MAX_PROTECT_RATE:
        return False

    # class_3 손실은 특히 제한
    if class3_rate > MAX_STRONG_RATE:
        return False

    # 전체 기준으로도 의미 있는 제거 성능이 있어야 함
    if score <= 0:
        return False

    if verbose:
        print(f"\n=== {name} ===")
        print(f"선택된 행 수: {cnt}")
        print(f"class_0 비율: {class0_rate:.3f}")
        print(f"class_2+3 비율: {protect_rate:.3f}")
        print(f"class_0 제거율: {class0_remove_rate:.3f}")
        print(f"class_2 손실률: {class2_loss_rate:.3f}")
        print(f"class_3 손실률: {class3_loss_rate:.3f}")
        print(f"score: {score:.3f}")

    return True


def rule_to_code(name, conds, thr_round=3):
    lines = [f'    "{name}":']
    parts = []

    for f, op, thr in conds:
        thr = float(np.round(thr, thr_round))

        if op == "<=":
            parts.append(f'(df["{f}"] <= {thr})')
        else:
            parts.append(f'(df["{f}"] > {thr})')  # ">": ge로 출력

    joined = " &\n        ".join(parts)
    lines.append(f"        {joined},")

    return "\n".join(lines)


def write_rule_file(out_path, selected, header_comment):
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header_comment + "\n")
        f.write("import numpy as np\n\n")

        f.write("RULE_NAMES = [\n")
        for name, _ in selected:
            f.write(f'    "{name}",\n')
        f.write("]\n\n")

        f.write("def build_conditions(df):\n")
        f.write("    conditions = {\n")

        for name, conds in selected:
            code = rule_to_code(name, conds)
            lines = code.splitlines()

            f.write("        " + lines[0] + "\n")
            for line in lines[1:]:
                f.write("        " + line + "\n")

        f.write("    }\n")
        f.write("    return conditions\n")

    print(f"saved to: {out_path.resolve()}")



def save_combined_avoid_eval(df, selected, out_csv="combined_avoid_eval.csv"):
    avoid_mask = np.zeros(len(df), dtype=bool)

    for name, conds in selected:
        avoid_mask |= make_mask_from_conds(df, conds)

    removed = df[avoid_mask]
    remain = df[~avoid_mask]

    total0 = (df["target_class"] == 0).sum()
    total1 = (df["target_class"] == 1).sum()
    total2 = (df["target_class"] == 2).sum()
    total3 = (df["target_class"] == 3).sum()

    c0 = (removed["target_class"] == 0).sum()
    c1 = (removed["target_class"] == 1).sum()
    c2 = (removed["target_class"] == 2).sum()
    c3 = (removed["target_class"] == 3).sum()

    summary = pd.DataFrame([{
        "total_count": len(df),
        "removed_count": len(removed),
        "remain_count": len(remain),
        "removed_rate": round(len(removed) / len(df) * 100, 1),

        "removed_class0": int(c0),
        "removed_class1": int(c1),
        "removed_class2": int(c2),
        "removed_class3": int(c3),

        "class0_remove_rate": round(c0 / total0 * 100, 1) if total0 else 0,
        "class1_remove_rate": round(c1 / total1 * 100, 1) if total1 else 0,
        "class2_loss_rate": round(c2 / total2 * 100, 1) if total2 else 0,
        "class3_loss_rate": round(c3 / total3 * 100, 1) if total3 else 0,

        "removed_class0_rate": round((removed["target_class"] == 0).mean() * 100, 1),
        "removed_class1_rate": round((removed["target_class"] == 1).mean() * 100, 1),
        "removed_class2_rate": round((removed["target_class"] == 2).mean() * 100, 1),
        "removed_class3_rate": round((removed["target_class"] == 3).mean() * 100, 1),

        "remain_class0_rate": round((remain["target_class"] == 0).mean() * 100, 1),
        "remain_class1_rate": round((remain["target_class"] == 1).mean() * 100, 1),
        "remain_class2_rate": round((remain["target_class"] == 2).mean() * 100, 1),
        "remain_class3_rate": round((remain["target_class"] == 3).mean() * 100, 1),
    }])

    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(summary.T)


def find_avoid_rule():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    if "target_class" not in df.columns:
        raise ValueError("target_class 컬럼이 없습니다. 데이터 생성 시 target_class를 먼저 저장하세요.")

    train, valid, split_date = split_train_valid_by_date_ratio(df, valid_ratio=VALID_RATIO)

    features = get_features(train)
    literals, literal_masks = build_literals(train, features)

    feature_groups, group_limits = get_feature_groups()

    train_bad = train["target_class"].to_numpy() == 0
    train_protect = train["target_class"].to_numpy() >= 2
    train_strong = train["target_class"].to_numpy() == 3

    rules = mine_avoid_rules(
        df=train,
        literals=literals,
        literal_masks=literal_masks,
        bad=train_bad,
        protect=train_protect,
        strong=train_strong,
        min_count=MIN_CNT_AVOID,
        max_depth=MAX_DEPTH_AVOID,
        beam=AVOID_BEAM,
        top_n=AVOID_TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    selected, train_avoid_mask = select_avoid_rules_max_class0_under_class3_limit(
        df=train,
        rules=rules,
        class3_loss_limit=CLASS_3_LOSS_LIMIT,
        min_count=MIN_CNT_AVOID,
        max_rules=None,
        verbose=True,
    )

    print(f"[AVOID] 통과 룰 개수: {len(selected)} / {len(rules)}")

    write_rule_file(
        AVOID_OUT_PATH,
        selected,
        header_comment=(
            "# auto-generated: lowscan avoid rules for class_0\n"
            f"# split_date: {pd.to_datetime(split_date).date()}\n"
            "# generated on train only\n"
        )
    )

    print("[TRAIN]")
    save_combined_avoid_eval(train, selected, "csv/class0_avoid_train_eval.csv")

    print("[VALID]")
    save_combined_avoid_eval(valid, selected, "csv/class0_avoid_valid_eval.csv")

def select_avoid_rules_max_class0_under_class3_limit(
        df,
        rules,
        class3_loss_limit=0.025,
        min_count=40,
        max_rules=None,
        verbose=True,
):
    total_class3 = int((df["target_class"] == 3).sum())
    class3_limit_count = int(total_class3 * class3_loss_limit)

    avoid_mask = np.zeros(len(df), dtype=bool)
    selected = []

    used_class3 = 0

    # 후보 룰을 먼저 만들어둠
    candidates = []

    for i, (
            cnt,
            bad_cnt,
            bad_rate,
            protect_cnt,
            protect_rate,
            strong_cnt,
            strong_rate,
            conds,
            score,
    ) in enumerate(rules, start=1):

        name = (
            f"avoid_{i:03d}"
            f"_s{score:.2f}"
            f"_n{cnt}"
            f"_bad{bad_rate:.3f}"
            f"_p{protect_rate:.3f}"
            f"_strong{strong_rate:.3f}"
        )

        rule_mask = make_mask_from_conds(df, conds)

        if not test_avoid_condition(name, rule_mask, df, min_count=min_count, verbose=False):
            continue

        candidates.append({
            "name": name,
            "conds": conds,
            "rule_mask": rule_mask,
            "score": score,
            "bad_rate": bad_rate,
            "protect_rate": protect_rate,
            "strong_rate": strong_rate,
        })

    # 매번 "현재 avoid_mask 기준으로" 추가 효율이 가장 좋은 룰을 고름
    while True:
        best = None

        reject_stats = {
            "no_new_remove": 0,
            "added_total_small": 0,
            "no_class0": 0,
            "class3_limit": 0,
            "class0_rate": 0,
            "protect_rate": 0,
            "class3_rate": 0,
            "candidate_ok": 0,
        }

        for cand in candidates:
            rule_mask = cand["rule_mask"]
            new_remove = rule_mask & ~avoid_mask

            if not new_remove.any():
                reject_stats["no_new_remove"] += 1
                continue

            added_class0 = int(((df["target_class"] == 0) & new_remove).sum())
            added_class2 = int(((df["target_class"] == 2) & new_remove).sum())
            added_class3 = int(((df["target_class"] == 3) & new_remove).sum())
            added_total = int(new_remove.sum())

            if added_total < 5:
                reject_stats["added_total_small"] += 1
                continue

            if added_class0 <= 0:
                reject_stats["no_class0"] += 1
                continue

            if used_class3 + added_class3 > class3_limit_count:
                reject_stats["class3_limit"] += 1
                continue

            added_class0_rate = added_class0 / added_total
            added_class2_rate = added_class2 / added_total
            added_class3_rate = added_class3 / added_total
            added_protect_rate = (added_class2 + added_class3) / added_total

            if added_class0_rate < 0.68:
                reject_stats["class0_rate"] += 1
                continue

            if added_protect_rate > 0.22:
                reject_stats["protect_rate"] += 1
                continue

            if added_class3_rate > 0.05:
                reject_stats["class3_rate"] += 1
                continue

            reject_stats["candidate_ok"] += 1

            efficiency = (
                    added_class0
                    - CLASS_3_SCORE * added_class3
                    - CLASS_2_SCORE * added_class2
            )

            key = (
                added_class0_rate,
                -added_class3_rate,
                -added_protect_rate,
                efficiency,
                added_class0,
                cand["score"],
            )

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "cand": cand,
                    "added_class0": added_class0,
                    "added_class2": added_class2,
                    "added_class3": added_class3,
                    "added_total": added_total,
                    "added_class0_rate": added_class0_rate,
                    "added_class3_rate": added_class3_rate,
                    "added_protect_rate": added_protect_rate,
                    "efficiency": efficiency,
                }

        print("[SELECT] reject_stats:", reject_stats)

        if best is None:
            break

        cand = best["cand"]

        avoid_mask |= cand["rule_mask"]
        selected.append((cand["name"], cand["conds"]))

        used_class3 += best["added_class3"]

        print(
            "[SELECT] pick:",
            cand["name"],
            "added_total=", best["added_total"],
            "added_c0=", best["added_class0"],
            "added_c2=", best["added_class2"],
            "added_c3=", best["added_class3"],
            "added_c0_rate=", round(best["added_class0_rate"], 3),
            "added_protect_rate=", round(best["added_protect_rate"], 3),
            "added_c3_rate=", round(best["added_class3_rate"], 3),
        )

        if max_rules is not None and len(selected) >= max_rules:
            break

    if verbose:
        removed = df[avoid_mask]
        total0 = int((df["target_class"] == 0).sum())
        total2 = int((df["target_class"] == 2).sum())

        c0 = int((removed["target_class"] == 0).sum())
        c2 = int((removed["target_class"] == 2).sum())
        c3 = int((removed["target_class"] == 3).sum())

        print(
            f"[CLASS3_LIMIT] selected={len(selected)}, "
            f"class0_remove_rate={c0 / total0 * 100:.2f}%, "
            f"class2_loss_rate={c2 / total2 * 100:.2f}%, "
            f"class3_loss_rate={c3 / total_class3 * 100:.2f}% "
            f"({c3}/{class3_limit_count})"
        )

    return selected, avoid_mask


if __name__ == "__main__":
    find_avoid_rule()

    try:
        import winsound
        winsound.Beep(1500, 500)  # 1000Hz, 0.5초
        winsound.Beep(1000, 500)  # 1000Hz, 0.5초
    except ImportError:
        pass
