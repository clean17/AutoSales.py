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

from utils import build_literals, split_train_valid_by_date_ratio, write_rule_file, make_mask_from_conds

CSV_PATH = "../csv/low_result_7_desc.csv"

DEPTH0_FEATURES = {
    "ATR_pct",
    "vol5",
    "today_pct",
    "max_drop_7d",
    "gap_pct",
    "pct_vs_lastweek",
    "dist_to_ma20",
}

# ============================================================
# CLASS23 RULE SETTINGS
# 목적: target_class == 2 or 3 탐색
# ============================================================
GOOD_OUT_PATH = Path("lowscan_rules.py")
GOOD_EXPAND_RATIO = [0.32, 0.55, 0.73, 0.82, 0.85]
MIN_CNT = 120
MAX_DEPTH = 4
# MIN_RATE = GOOD_EXPAND_RATIO[(MAX_DEPTH-1)]
MIN_RATE = 0.75
BEAM = 30000
TOP_N = 3000

VALID_MIN_RATE = 0.63
VALID_MIN_CNT = 6
VALID_RATIO = 0.15

redule_rule = 1  # 중복 룰 제거 조건




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
    return [c for c in df.columns if c not in exclude]


def get_feature_groups():
    feature_groups = {
        "today_pct": "PRICE",

        "max_drop_7d": "DROP",

        "dist_to_ma5": "MA5_POSITION",
        "dist_to_ma20": "MA20_POSITION",

        "pct_vs_lastweek": "WEEK_POSITION",

        "ma5_ma20_gap_chg_1d": "TREND",

        "gap_pct": "GAP",

        "today_tr_val_eok": "VOLUME",
        "tr_val_rank_20d": "VOLUME",
        "tr_value_ratio_5d": "VOLUME",

        "vol5": "VOLATILITY",
        "ATR_pct": "VOLATILITY",
        "vol_ratio_5_15": "VOLATILITY",
    }

    group_limits = {
        "PRICE": 1,
        "DROP": 1,
        "MA5_POSITION": 1,
        "MA20_POSITION": 1,
        "WEEK_POSITION": 1,
        "TREND": 1,
        "GAP": 1,
        "VOLATILITY": 2,
        "VOLUME": 1,
    }

    return feature_groups, group_limits


def mine_good_rules(
        df,
        literals,
        literal_masks,
        target,
        min_ratio=0.83,
        min_count=25,
        max_depth=4,
        beam=30000,
        top_n=1000,
        feature_groups=None,
        group_limits=None,
        cnt_priority_ratio=0.85,
):
    """
    beam : 단계별 상위 수량만 다음 뎁스로 가져간다, 한 depth에서 유지할 후보 개수
    expand_ratio : 각 후보의 성능이 이 값보다 커야 다음 뎁스로 진행, 다음 depth로 확장할 최소 기준
    min_ratio : “좋은 룰”로 저장할 기준
    cnt_priority_ratio : 넘기면 ratio 보다 cnt 를 더 중요시
    - ratio >= cnt_priority_ratio 이면 cnt 우선(그 다음 ratio)
    - ratio <  cnt_priority_ratio 이면 ratio 우선(그 다음 cnt)

    권장 depth
    | depth | 적당한 new 개수 | 해석              |
    | ----: | -------------: | ---------------- |
    |     0 |       50 ~ 500 | 단일 조건 후보     |
    |     1 |    500 ~ 5,000 | 2개 조건 조합      |
    |     2 | 1,000 ~ 10,000 | 3개 조건 조합      |
    |     3 | 1,000 ~ 10,000 | 4개 조건 조합      |
    |     4 |    100 ~ 3,000 | 5개 조건 최종 후보 |

    :returns (count, ratio, up_cnt, conds) 리스트 반환
    """

    # df 길이 만큼의 True 배열(불리언 마스크), 빈 배열의 튜플 리스트 > "데이터프레임의 모든 행을 선택한다"는 상태
    beams = [(np.ones(len(df), dtype=bool), [])]
    good = {}

    print(
        "\nbeam", beam,
        "\nmin_ratio", min_ratio,
        "\nmin_count", min_count,
        "\nmax_depth", max_depth,
        "\nexpand_ratio", GOOD_EXPAND_RATIO,
        "\ntop_n", top_n,
        "\nvalid_min_rate", VALID_MIN_RATE,
        "\nvalid_min_cnt", VALID_MIN_CNT,
        "\n"
    )

    if feature_groups is None:
        feature_groups = {}
    if group_limits is None:
        group_limits = {}

    def get_group(feat_name):
        return feature_groups.get(feat_name)

    def beam_key(ratio, cnt):
        """
        beam(확장 후보)용 비교키
        "좋은 후보"가 더 큰 key를 갖도록 설계 (나중에 key 비교로 worst 교체)
        """
        if ratio >= cnt_priority_ratio:
            return (1, cnt, ratio)  # cnt 우선
        return (0, ratio, cnt)      # ratio 우선

    for depth in range(max_depth):
        print("----------------------------------")
        print("[GOOD] depth", depth)

        # heap item: (key, uid, mask, conds, ratio, cnt)
        # heap[0] = 가장 "나쁜" 후보 (key가 가장 작음)
        heap = []
        uid = count()  # 카운트 제네레이터 (itertools 모듈)

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

            # 현재 rule에서 그룹 사용량 계산
            group_used = {}
            for f in used_feats:
                g = get_group(f)
                if g is None:
                    continue
                group_used[g] = group_used.get(g, 0) + 1

            for (lit, lmask) in zip(literals, literal_masks):  # zip() 이용한 동시 순회
                feat = lit[0]

                # depth 0에서는 상위 피처만 시작 조건으로 허용
                if not is_literal_allowed_at_depth(feat, depth):
                    continue

                # 동일 feature 중복 금지
                if feat in used_feats:
                    continue

                # 그룹 제약 체크
                g = get_group(feat)
                if g is not None:
                    limit = group_limits.get(g, None)
                    if limit is not None:
                        if group_used.get(g, 0) >= limit:
                            continue

                # 새로운 subset
                m = base_mask & lmask
                cnt = int(m.sum())

                if cnt < min_count:
                    continue

                up = int((m & target).sum())
                ratio = up / cnt
                # score = ratio * np.log1p(cnt)
                # score = (ratio ** 2) * np.log1p(cnt)
                score = (ratio ** 3) * np.log1p(cnt)  # ratio에 더 강하게 보상

                # good 저장
                if ratio >= min_ratio:
                    key2 = tuple(sorted(
                        (c[0], c[1], round(float(c[2]), 6))
                        for c in (conds + [lit])
                    ))

                    prev = good.get(key2)
                    if prev is None or score > prev[4]:
                        good[key2] = (cnt, ratio, up, conds + [lit], score)

                """
                new ≈ beam의 30~70%가 되도록 expand_ratio 조절 필요

                depth 0 >> new가 beam의 1~20%여도 괜찮음
                depth 1 >> beam의 20~70%
                depth 2+ > beam의 30~70%
                """
                # 확장 후보
                if ratio >= GOOD_EXPAND_RATIO[depth]:
                    k = beam_key(ratio, cnt)
                    item = (k, next(uid), m, conds + [lit], ratio, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        if k <= heap[0][0]:
                            continue
                        heapq.heapreplace(heap, item)

        # 다음 depth로 넘길 후보들(좋은 순): key 내림차순
        # key는 (group, primary, secondary)인데 primary/secondary는 큰 게 좋게 만들어 둠
        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("[GOOD] new", len(new))

        if not new:
            print("[GOOD] no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("[GOOD] tail ratio:", round(tail[4], 3), "cnt:", tail[5], "conds:", tail[3])  # 최악 후보 출력 (디버깅용)

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    # 점수 내림차순 정렬
    def out_key(x):
        cnt, ratio, up, conds, score = x
        return (-score, -ratio, -cnt, len(conds))

    out = sorted(good.values(), key=out_key)

    if top_n is not None:
        out = out[:top_n]

    return out



def test_good_condition(name, cond, df, min_count=20, verbose=False):
    """
    룰 검증
    """
    # 조건을 만족한 종목/날짜만
    sub = df[cond]

    if len(sub) == 0:
        return False

    up_cnt = int((sub["target_before_stop_7"] == 1).sum())
    ratio = up_cnt / len(sub)

    """
    confidence: 성공률에 “표본 수 신뢰도 할인”을 적용한 값
    ratio = 0.70 일 때
    # 표본 20개
    confidence = 0.70 - (1 / sqrt(20))
               = 0.70 - 0.224
               = 0.476
    
    # 표본 200개
    confidence = 0.70 - (1 / sqrt(200))
               = 0.70 - 0.071
               = 0.629
    """
    confidence = ratio - (1 / np.sqrt(len(sub)))

    if confidence < 0.55:
        return False

    # 표본이 너무 적으면 과적합 가능성이 큼
    if len(sub) < min_count:
        return False

    if verbose:
        print(f"\n=== {name} ===")
        print(f"선택된 행 수: {len(sub)}")
        print(f"성공 개수   : {up_cnt}")
        print(f"성공률      : {ratio:.3f}")
        print(f"confidence : {confidence:.3f}")

    return True



def find_good_rule(m_ratio, m_count):
    df = pd.read_csv(CSV_PATH, low_memory=False)

    df["today"] = pd.to_datetime(df["today"], errors="coerce")
    df = df.dropna(subset=["today"])

    train, valid, split_date = split_train_valid_by_date_ratio(df, valid_ratio=VALID_RATIO)

    # 룰 생성은 train으로만 한다
    features = get_features(train)
    literals, literal_masks = build_literals(train, features)

    feature_groups, group_limits = get_feature_groups()
    target = train["target_before_stop_7"].to_numpy() == 1

    rules = mine_good_rules(
        df=train,
        literals=literals,
        literal_masks=literal_masks,
        target=target,
        min_ratio=m_ratio,
        min_count=m_count,
        max_depth=MAX_DEPTH,
        beam=BEAM,
        top_n=TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    fail_reason = {
        "train_count": 0,
        "train_ratio": 0,
        "valid_count": 0,
        "valid_ratio": 0,
        "selected": 0,
    }

    valid_counts = []
    debug_candidates = []
    passed = []

    for i, (cnt, ratio, up, conds, score) in enumerate(rules, start=1):
        train_mask = make_mask_from_conds(train, conds)
        valid_mask = make_mask_from_conds(valid, conds)

        train_sub = train[train_mask]
        valid_sub = valid[valid_mask]

        train_cnt = len(train_sub)
        valid_cnt = len(valid_sub)

        valid_counts.append(valid_cnt)

        if train_cnt > 0:
            train_up = int((train_sub["target_before_stop_7"] == 1).sum())
            train_ratio = train_up / train_cnt
        else:
            train_up = 0
            train_ratio = 0.0

        if valid_cnt > 0:
            valid_up = int((valid_sub["target_before_stop_7"] == 1).sum())
            valid_ratio = valid_up / valid_cnt
        else:
            valid_up = 0
            valid_ratio = 0.0

        debug_candidates.append({
            "train_count": train_cnt,
            "train_success_cnt": train_up,
            "train_success_rate": round(train_ratio * 100, 1),
            "valid_count": valid_cnt,
            "valid_success_cnt": valid_up,
            "valid_success_rate": round(valid_ratio * 100, 1),
            "score": round(float(score), 6),
            "conds": conds,
        })

        if train_cnt < m_count:
            fail_reason["train_count"] += 1
            continue

        if train_ratio < m_ratio:
            fail_reason["train_ratio"] += 1
            continue

        if valid_cnt < VALID_MIN_CNT:
            fail_reason["valid_count"] += 1
            continue

        if valid_ratio < VALID_MIN_RATE:
            fail_reason["valid_ratio"] += 1
            continue

        fail_reason["selected"] += 1

        passed.append({
            "conds": conds,
            "train_count": train_cnt,
            "train_success_cnt": train_up,
            "train_ratio": train_ratio,
            "valid_count": valid_cnt,
            "valid_success_cnt": valid_up,
            "valid_ratio": valid_ratio,
            "score": score,
        })

    # ============================================================
    # DEBUG 출력
    # ============================================================
    print("valid_count max:", max(valid_counts) if valid_counts else 0)
    print("valid_count p95:", np.percentile(valid_counts, 95) if valid_counts else 0)
    print("valid_count p90:", np.percentile(valid_counts, 90) if valid_counts else 0)
    print("valid_count p50:", np.percentile(valid_counts, 50) if valid_counts else 0)
    print("valid_count >= 10:", sum(c >= 10 for c in valid_counts))
    print("valid_count >= 15:", sum(c >= 15 for c in valid_counts))

    debug_df = pd.DataFrame(debug_candidates)

    if len(debug_df):
        print("\n[DEBUG] top valid_count candidates")
        tmp = debug_df.sort_values(
            ["valid_count", "valid_success_rate", "train_success_rate"],
            ascending=[False, False, False],
        )
        print(
            tmp.head(30)[[
                "train_count",
                "train_success_rate",
                "valid_count",
                "valid_success_rate",
                "conds",
            ]].to_string(index=False)
        )

        print("\n[DEBUG] valid_count >= VALID_MIN_CNT candidates")
        tmp2 = debug_df[debug_df["valid_count"] >= VALID_MIN_CNT].copy()

        if len(tmp2):
            tmp2 = tmp2.sort_values(
                ["valid_success_rate", "valid_count", "train_success_rate"],
                ascending=[False, False, False],
            )

            print(
                tmp2.head(30)[[
                    "train_count",
                    "train_success_rate",
                    "valid_count",
                    "valid_success_rate",
                    "conds",
                ]].to_string(index=False)
            )

            print("\n[DEBUG] valid_count >= VALID_MIN_CNT summary")
            print("candidate_count:", len(tmp2))
            print("valid_success_rate max:", tmp2["valid_success_rate"].max())
            print("valid_success_rate p90:", np.percentile(tmp2["valid_success_rate"], 90))
            print("valid_success_rate p50:", np.percentile(tmp2["valid_success_rate"], 50))
            print("valid_success_rate >= 50:", int((tmp2["valid_success_rate"] >= 50).sum()))
            print("valid_success_rate >= 55:", int((tmp2["valid_success_rate"] >= 55).sum()))
            print("valid_success_rate >= 60:", int((tmp2["valid_success_rate"] >= 60).sum()))
            print("valid_success_rate >= 65:", int((tmp2["valid_success_rate"] >= 65).sum()))
            print("valid_success_rate >= 70:", int((tmp2["valid_success_rate"] >= 70).sum()))
            print("valid_success_rate >= 75:", int((tmp2["valid_success_rate"] >= 75).sum()))
        else:
            print("없음")

    print("[GOOD] fail reason:", fail_reason)
    print(f"\n[GOOD] valid 통과 룰 개수: {len(passed)} / {len(rules)}")

    # ============================================================
    # valid 성능순으로 통과 룰 정렬
    # ============================================================
    passed = sorted(
        passed,
        key=lambda x: (
            x["valid_ratio"],
            x["valid_count"],
            x["train_ratio"],
            x["train_count"],
        ),
        reverse=True,
    )

    selected = []

    for row in passed:
        name = f"rule_{len(selected) + 1:03d}"
        selected.append((name, row["conds"]))

    # ============================================================
    # 중복 룰 제거
    # ============================================================
    if redule_rule == 1:
        print(f"[GOOD] before reduce: {len(selected)}")
        selected = reduce_rules_by_new_rows(valid, selected, min_new_rows=3)
        print(f"[GOOD] after reduce: {len(selected)}")

    # reduce 이후 이름 재정렬
    selected = [
        (f"rule_{i + 1:03d}", conds)
        for i, (_, conds) in enumerate(selected)
    ]


    # ============================================================
    # 최종 선택 룰 평가 / 저장
    # ============================================================
    eval_combined_good_rules(train, selected, title="TRAIN")
    eval_combined_good_rules(valid, selected, title="VALID")

    write_rule_file(
        GOOD_OUT_PATH,
        selected,
        header_comment=(
            "# auto-generated: lowscan good buy rules\n"
            "# train/valid split applied\n"
            f"# split_date: {pd.to_datetime(split_date).date()}\n"
            f"# train_min_rate: {m_ratio}\n"
            f"# train_min_count: {m_count}\n"
            f"# valid_min_rate: {VALID_MIN_RATE}\n"
            f"# valid_min_count: {VALID_MIN_CNT}\n"
            "# usage:\n"
            "#    import lowscan_rules\n"
            "#    buy_conditions = lowscan_rules.build_conditions(df)\n"
            "#    \n"
            "#    buy_mask = np.zeros(len(df), dtype=bool)\n"
            "#    for cond in buy_conditions.values():\n"
            "#        buy_mask |= cond\n"
            "#    \n"
            "#    df = df[buy_mask].copy()\n"
        )
    )



def eval_combined_good_rules(df, selected, title=""):
    buy_mask = np.zeros(len(df), dtype=bool)

    for name, conds in selected:
        buy_mask |= make_mask_from_conds(df, conds)

    sub = df[buy_mask]

    print(f"\n[{title}] combined good rules")
    print("rule_count:", len(selected))
    print("row_count:", len(sub))

    if len(sub) == 0:
        return

    print("target_before_stop_7:", round((sub["target_before_stop_7"] == 1).mean() * 100, 2), "%")
    print("target_before_stop_12:", round((sub["target_before_stop_12"] == 1).mean() * 100, 2), "%")

    if "target_class" in sub.columns:
        print("class0:", round((sub["target_class"] == 0).mean() * 100, 2), "%")
        print("class1:", round((sub["target_class"] == 1).mean() * 100, 2), "%")
        print("class2:", round((sub["target_class"] == 2).mean() * 100, 2), "%")
        print("class3:", round((sub["target_class"] == 3).mean() * 100, 2), "%")

    if "validation_high_rate_max" in sub.columns:
        print("avg_high:", round(sub["validation_high_rate_max"].mean(), 2))
        print("median_high:", round(sub["validation_high_rate_max"].median(), 2))

    if "validation_low_rate_min" in sub.columns:
        print("avg_low:", round(sub["validation_low_rate_min"].mean(), 2))
        print("median_low:", round(sub["validation_low_rate_min"].median(), 2))


def reduce_rules_by_new_rows(df, selected, min_new_rows=2):
    final = []
    used_mask = np.zeros(len(df), dtype=bool)

    for name, conds in selected:
        rule_mask = make_mask_from_conds(df, conds)
        new_rows = rule_mask & ~used_mask
        new_cnt = int(new_rows.sum())
        total_cnt = int(rule_mask.sum())

        if new_cnt >= min_new_rows:
            final.append((name, conds))
            used_mask |= rule_mask
            print(
                f"[REDUCE KEEP] {name} total_cnt={total_cnt} new_cnt={new_cnt} conds={conds}"
            )
        else:
            print(
                f"[REDUCE DROP] {name} total_cnt={total_cnt} new_cnt={new_cnt} conds={conds}"
            )

    print("[REDUCE] final_rule_count:", len(final))
    print("[REDUCE] final_row_count:", int(used_mask.sum()))

    return final



if __name__ == "__main__":
    for i in range(1):
        for_cnt = MIN_CNT + (i*1)

        for j in range(1):
            for_rate = MIN_RATE + (0.01*j)
            find_good_rule(for_rate, for_cnt)

    try:
        import winsound
        winsound.Beep(1500, 500)  # 1000Hz, 0.5초
        winsound.Beep(1000, 500)  # 1000Hz, 0.5초
    except ImportError:
        pass

