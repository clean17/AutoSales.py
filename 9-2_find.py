"""
생성한 csv 파일을 가지고 저점 조건을 찾는 스크립트

사용할 때는

buy_conditions = lowscan_rules.build_conditions(df)
avoid_conditions = lowscan_avoid_rules.build_conditions(df)

buy_mask = np.zeros(len(df), dtype=bool)
for cond in buy_conditions.values():
    buy_mask |= cond

avoid_mask = np.zeros(len(df), dtype=bool)
for cond in avoid_conditions.values():
    avoid_mask |= cond

final_buy_mask = buy_mask & ~avoid_mask
"""

import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count

from utils import split_train_valid_by_date_ratio, build_literals, make_mask_from_conds, write_rule_file

CSV_PATH = "csv/low_result_7_desc.csv"
GOOD_OUT_PATH = Path("lowscan_rules.py")

# 고정
GOOD_EXPAND_RATIO = [0.33, 0.55, 0.73, 0.80, 0.85]

MIN_RATE        = 0.75
MIN_CNT         = 120
MAX_DEPTH       = 4
TOP_N = 3000

VALID_MIN_RATE = 0.65
VALID_MIN_CNT = 6
VALID_RATIO = 0.10



def get_exclude_columns(df=None):
    """
    제외할 피쳐 Set

    df를 넘기면 패턴 기반으로 실제 존재하는 컬럼까지 자동 제외.
    df를 안 넘겨도 기본 제외 컬럼 set 반환.
    """
    exclude = {
        # 식별자 / 메타
        "ticker", "stock_name", "today", "idx",

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

        "dist_from_low_20d": "POSITION",
        "three_m_cur_max_chg_rate": "POSITION",  #
        "dist_to_ma5": "POSITION",  #
        "dist_to_ma20": "POSITION",  #

        "pct_vs_lastweek": "WEEK_POSITION",  #

        "ma5_ma20_gap_chg_1d": "TREND",

        "gap_pct": "GAP",  #

        "today_tr_val_eok": "VOLUME",
        "tr_val_rank_20d": "VOLUME",
        "tr_value_ratio_5d": "VOLUME",  #

        "MACD_hist_3d": "MACD",

        "vol5": "VOLATILITY",  #
        "ATR_pct": "VOLATILITY",
        "vol_ratio_5_15": "VOLATILITY",  #
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

                # 동일 feature 중복 금지 (기존)
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
    # df = pd.read_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH, low_memory=False)

    df["today"] = pd.to_datetime(df["today"], errors="coerce")
    df = df.dropna(subset=["today"])

    train, valid, split_date = split_train_valid_by_date_ratio(
        df,
        valid_ratio=VALID_RATIO,
    )

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
        beam=10000,
        top_n=TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    selected = []

    eval_rows = []

    for i, (cnt, ratio, up, conds, score) in enumerate(rules, start=1):
        train_mask = make_mask_from_conds(train, conds)
        valid_mask = make_mask_from_conds(valid, conds)

        train_sub = train[train_mask]
        valid_sub = valid[valid_mask]

        if len(train_sub) < m_count:
            continue

        train_up = int((train_sub["target_before_stop_7"] == 1).sum())
        train_ratio = train_up / len(train_sub)

        if train_ratio < m_ratio:
            continue

        if len(valid_sub) < VALID_MIN_CNT:
            continue

        valid_up = int((valid_sub["target_before_stop_7"] == 1).sum())
        valid_ratio = valid_up / len(valid_sub)

        if valid_ratio < VALID_MIN_RATE:
            continue

        name = (
            f"rule_{len(selected) + 1:03d}"
            # f"_trn{len(train_sub)}_trr{train_ratio:.3f}"
            # f"_van{len(valid_sub)}_var{valid_ratio:.3f}"
        )

        selected.append((name, conds))

        eval_rows.append({
            "rule": name,
            "train_count": len(train_sub),
            "train_success_cnt": train_up,
            "train_success_rate": round(train_ratio * 100, 1),
            "valid_count": len(valid_sub),
            "valid_success_cnt": valid_up,
            "valid_success_rate": round(valid_ratio * 100, 1),
            "split_date": pd.to_datetime(split_date).date(),
            "conds": str(conds),
        })

    # 진단용으로 사용, 실제 룰 저장에서는 제외
    # print(f"[GOOD] before reduce: {len(selected)}")
    # selected = reduce_rules_by_new_rows(valid, selected, min_new_rows=1)
    # print(f"[GOOD] after reduce: {len(selected)}")

    print(f"\n[GOOD] valid 통과 룰 개수: {len(selected)} / {len(rules)}")

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